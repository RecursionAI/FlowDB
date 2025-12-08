import os
import numpy as np
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.params import Body
from pydantic import BaseModel
from fastmcp import FastMCP
from dotenv import load_dotenv
from flowdb.core.engine import FlowDB

load_dotenv()

db_instance: Optional[FlowDB] = None
DB_PATH = os.getenv("FLOWDB_PATH", "./flow_data")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_instance
    print(f"--- FlowDB Starting at {DB_PATH} ---")
    db_instance = FlowDB(storage_path=DB_PATH)
    yield
    print("--- FlowDB Shutting Down ---")


app = FastAPI(title="FlowDB", lifespan=lifespan)

api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)):
    real_key = os.getenv("FLOWDB_API_KEY")

    if not real_key:
        return True

    if api_key != f"Bearer {real_key}":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key",
        )
    return True


# Helper Models
class GenericRecord(BaseModel):
    id: str
    data: Dict[str, Any]
    vector: Optional[List[float]] = None


def get_db():
    if not db_instance:
        raise HTTPException(500, "DB not initialized")
    return db_instance


@app.post("/v1/{collection_name}/upsert", dependencies=[Depends(verify_api_key)])
def rest_put(collection_name: str, payload: GenericRecord):
    col = get_db().collection(collection_name, GenericRecord)
    vec = np.array(payload.vector, dtype=np.float32) if payload.vector else None
    col.upsert(payload.id, payload, vector=vec)
    return {"status": "success", "id": payload.id}


@app.get("/v1/{collection_name}/read/{key}", dependencies=[Depends(verify_api_key)])
def rest_get(collection_name: str, key: str):
    col = get_db().collection(collection_name, GenericRecord)
    res = col.read(key)
    if not res: raise HTTPException(404, "Not found")
    return res


@app.get("/v1/{collection_name}/all", dependencies=[Depends(verify_api_key)])
def rest_list(collection_name: str, limit: int = 20, skip: int = 0):
    col = get_db().collection(collection_name, GenericRecord)
    return col.all(limit=limit, skip=skip)


@app.post("/v1/{collection_name}/search", dependencies=[Depends(verify_api_key)])
def rest_search(collection_name: str, query_text: str = Body(..., embed=True), limit: int = 5):
    """
    Standard Search: Vector similarity lookup.
    """
    col = get_db().collection(collection_name, GenericRecord)

    # This triggers the 'search' method in engine.py we just fixed
    results = col.search(query_text=query_text, limit=limit)

    # Convert Pydantic models to plain dicts for JSON response
    return [item.model_dump() for item in results]


@app.delete("/v1/{collection_name}/delete/{key}", dependencies=[Depends(verify_api_key)])
def rest_delete(collection_name: str, key: str):
    """
    Delete a record in a collection by key.
    :param collection_name: Name of the collection
    :param key: Unique identifier of the record to delete
    :return: JSONResponse confirming record deletion
    """
    col = get_db().collection(collection_name, GenericRecord)
    deleted = col.delete(key)
    if not deleted:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"status": "deleted", "id": key}


@app.get("/v1/collections", dependencies=[Depends(verify_api_key)])
def rest_list_collections():
    """
    Returns a list of all active collections (tables).
    """
    return {"collections": get_db().list_collections()}


mcp = FastMCP("FlowDB Agent Interface")

mcp_asgi = mcp.sse_app()


@mcp.tool()
def flowdb_upsert(collection: str, key: str, data: Dict[str, Any], vector: List[float] = None):
    """
    Create or Update a record in the database.
    """
    col = db_instance.collection(collection, GenericRecord)

    record = GenericRecord(id=key, data=data, vector=vector)

    vec_np = np.array(vector, dtype=np.float32) if vector else None
    col.upsert(key, record, vector=vec_np)
    return f"Successfully saved record {key} to collection {collection}"


@mcp.tool()
def flowdb_read(collection: str, key: str) -> str:
    """
    Read data from the database.
    :returns a single record in a collection
    """
    col = db_instance.collection(collection, GenericRecord)
    res = col.read(key)
    if not res:
        return "Error: Record not found."
    return str(res.data)


@mcp.tool()
def flowdb_search(collection: str, query: str) -> str:
    """
    Semantic search. Finds records by meaning (e.g. "Who is the admin?").
    """
    col = db_instance.collection(collection, GenericRecord)

    # FIX 3: Accept 'query' string and let the Engine vectorize it!
    # This fixes the "Dimensions doesn't match" error.
    results = col.search(query_text=query, limit=3)

    summary = [f"ID: {r.id} | Data: {r.data}" for r in results]
    return "\n".join(summary)


@mcp.tool()
def flowdb_list(collection: str, limit: int = 20, skip: int = 0) -> str:
    """
    List records in a collection.
    :returns a list of objects or 'records' in a collection, up to the specified limit.
    Utilize skip and limit for pagination
    """
    col = db_instance.collection(collection, GenericRecord)
    results = col.all(limit=limit, skip=skip)

    if not results:
        return "No records found."

    # Return a concise summary for the AI
    summary = [f"ID: {r.id} | Data: {r.data}" for r in results]
    return "\n".join(summary)


@mcp.tool()
def flowdb_list_collections() -> str:
    """
    List all available collections.
    """
    names = db_instance.list_collections()
    if not names:
        return "No collections found."
    return "Available Collections:\n- " + "\n- ".join(names)


@mcp.tool()
def flowdb_delete(collection: str, key: str) -> str:
    """
    Delete a record by ID.
    """
    col = db_instance.collection(collection, GenericRecord)
    success = col.delete(key)
    if success:
        return f"Successfully deleted record {key} from {collection}."
    else:
        return f"Record {key} was not found."


app.mount("/mcp", mcp_asgi)


def start():
    uvicorn.run("flowdb.server.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
