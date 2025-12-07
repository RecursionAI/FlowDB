import os
import numpy as np
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.params import Body
from pydantic import BaseModel
from fastmcp import FastMCP
from dotenv import load_dotenv  # <--- NEW IMPORT

# Load .env file immediately
load_dotenv()

# Import your core engine
from flowdb.core.engine import FlowDB

# --- Shared Database State ---
# We keep the database instance global so both FastAPI and FastMCP can use it.
db_instance: Optional[FlowDB] = None
DB_PATH = os.getenv("FLOWDB_PATH", "./flow_data")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_instance
    print(f"--- FlowDB Starting at {DB_PATH} ---")
    db_instance = FlowDB(storage_path=DB_PATH)
    yield
    print("--- FlowDB Shutting Down ---")
    # In a real app, explicit closing would happen here


# --- 1. The FastAPI App (For Humans/Web Apps) ---
app = FastAPI(title="FlowDB", lifespan=lifespan)


# Helper Models
class GenericRecord(BaseModel):
    id: str
    data: Dict[str, Any]
    vector: Optional[List[float]] = None


def get_db():
    if not db_instance:
        raise HTTPException(500, "DB not initialized")
    return db_instance


# Standard REST Endpoints
@app.post("/v1/{collection_name}/upsert")
def rest_put(collection_name: str, payload: GenericRecord):
    col = get_db().collection(collection_name, GenericRecord)
    vec = np.array(payload.vector, dtype=np.float32) if payload.vector else None
    col.upsert(payload.id, payload, vector=vec)
    return {"status": "success", "id": payload.id}


@app.get("/v1/{collection_name}/read/{key}")
def rest_get(collection_name: str, key: str):
    col = get_db().collection(collection_name, GenericRecord)
    res = col.read(key)
    if not res: raise HTTPException(404, "Not found")
    return res


@app.get("/v1/{collection_name}/all")
def rest_list(collection_name: str, limit: int = 20, skip: int = 0):
    col = get_db().collection(collection_name, GenericRecord)
    return col.all(limit=limit, skip=skip)


@app.post("/v1/{collection_name}/search")
def rest_search(collection_name: str, query_text: str = Body(..., embed=True), limit: int = 5):
    """
    Standard Search: Vector similarity lookup.
    """
    col = get_db().collection(collection_name, GenericRecord)

    # This triggers the 'search' method in engine.py we just fixed
    results = col.search(query_text=query_text, limit=limit)

    # Convert Pydantic models to plain dicts for JSON response
    return [item.model_dump() for item in results]


# --- 2. The FastMCP Server (For AI Agents) ---
# We define this *separately* so we can curate exactly what the AI sees.
# We don't want the AI to see internal API details, just the high-level tools.

mcp = FastMCP("FlowDB Agent Interface")

mcp_asgi = mcp.sse_app()


@mcp.tool()
def flowdb_upsert(collection: str, key: str, data: Dict[str, Any], vector: List[float] = None):
    """
    Create or Update a record in the database.
    Use this to save data or fix typos in existing records.
    """
    # FastMCP handles the JSON serialization automatically!
    col = db_instance.collection(collection, GenericRecord)

    # Wrap data into our internal format
    record = GenericRecord(id=key, data=data, vector=vector)

    vec_np = np.array(vector, dtype=np.float32) if vector else None
    col.upsert(key, record, vector=vec_np)
    return f"Successfully saved record {key} to collection {collection}"


@mcp.tool()
def flowdb_read(collection: str, key: str) -> str:
    """
    Retrieve a full record by its ID.
    Returns the JSON representation of the data.
    """
    col = db_instance.collection(collection, GenericRecord)
    res = col.read(key)
    if not res:
        return "Error: Record not found."
    return str(res.data)


@mcp.tool()
def flowdb_search(collection: str, query_vector: List[float]) -> str:
    """
    Semantic search. Finds records with similar vector embeddings.
    Useful when you don't know the ID but know the 'meaning'.
    """
    col = db_instance.collection(collection, GenericRecord)
    vec_np = np.array(query_vector, dtype=np.float32)
    results = col.search(vector=vec_np, limit=3)

    # Return a simplified string summary for the AI
    summary = [f"ID: {r.id} | Data: {r.data}" for r in results]
    return "\n".join(summary)


# --- 3. The Merge (Mounting) ---
# We mount the MCP server endpoints onto the FastAPI app.
# Access via: http://localhost:8000/mcp/sse (for configuration)

app.mount("/mcp", mcp_asgi)


def start():
    uvicorn.run("flowdb.server.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    start()