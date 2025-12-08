import os
import base64
import binascii
import numpy as np
import uvicorn
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Body, Request, Response
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
    if db_instance:
        db_instance.close()


app = FastAPI(title="FlowDB", lifespan=lifespan)


class SecurityMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in ["/docs", "/openapi.json", "/favicon.ico"]:
            await self.app(scope, receive, send)
            return

        real_key = os.getenv("FLOWDB_API_KEY")
        if not real_key:
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        query_string = scope.get("query_string", b"").decode()

        input_key = None

        auth_header = headers.get(b"authorization", b"").decode()

        if auth_header.startswith("Bearer "):
            input_key = auth_header.split(" ")[1]
        elif auth_header.startswith("Basic "):
            try:
                encoded_creds = auth_header.split(" ")[1]
                decoded_bytes = base64.b64decode(encoded_creds)
                decoded_str = decoded_bytes.decode("utf-8")
                if ":" in decoded_str:
                    _, password = decoded_str.split(":", 1)
                    input_key = password
                else:
                    input_key = decoded_str
            except Exception:
                pass

        if not input_key:
            for param in query_string.split("&"):
                if param.startswith("api_key="):
                    input_key = param.split("=")[1]
                    break

        if input_key != real_key:
            await send({
                "type": "http.response.start",
                "status": 403,
                "headers": [(b"content-type", b"text/plain")],
            })
            await send({
                "type": "http.response.body",
                "body": b"Unauthorized: Invalid FlowDB API Key",
            })
            return

        await self.app(scope, receive, send)


app.add_middleware(SecurityMiddleware)


class GenericRecord(BaseModel):
    id: str
    data: Dict[str, Any]
    vector: Optional[List[float]] = None


def get_db():
    if not db_instance:
        raise HTTPException(500, "DB not initialized")
    return db_instance


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
    col = get_db().collection(collection_name, GenericRecord)
    results = col.search(query_text=query_text, limit=limit)
    return [item.model_dump() for item in results]


@app.delete("/v1/{collection_name}/delete/{key}")
def rest_delete(collection_name: str, key: str):
    col = get_db().collection(collection_name, GenericRecord)
    deleted = col.delete(key)
    if not deleted:
        raise HTTPException(status_code=404, detail="Record not found")
    return {"status": "deleted", "id": key}


@app.get("/v1/collections")
def rest_list_collections():
    return {"collections": get_db().list_collections()}


# --- 2. The FastMCP Server ---
mcp = FastMCP("FlowDB Agent Interface")
mcp_asgi = mcp.sse_app()


@mcp.tool()
def flowdb_upsert(collection: str, key: str, data: Dict[str, Any], vector: List[float] = None):
    """Create or Update a record in the database."""
    col = db_instance.collection(collection, GenericRecord)
    record = GenericRecord(id=key, data=data, vector=vector)
    vec_np = np.array(vector, dtype=np.float32) if vector else None
    col.upsert(key, record, vector=vec_np)
    return f"Successfully saved record {key} to collection {collection}"


@mcp.tool()
def flowdb_read(collection: str, key: str) -> str:
    """Read data from the database."""
    col = db_instance.collection(collection, GenericRecord)
    res = col.read(key)
    if not res: return "Error: Record not found."
    return str(res.data)


@mcp.tool()
def flowdb_search(collection: str, query: str) -> str:
    """Semantic search. Finds records by meaning."""
    col = db_instance.collection(collection, GenericRecord)
    results = col.search(query_text=query, limit=3)
    summary = [f"ID: {r.id} | Data: {r.data}" for r in results]
    return "\n".join(summary)


@mcp.tool()
def flowdb_list(collection: str, limit: int = 20, skip: int = 0) -> str:
    """List records in a collection."""
    col = db_instance.collection(collection, GenericRecord)
    results = col.all(limit=limit, skip=skip)
    if not results: return "No records found."
    summary = [f"ID: {r.id} | Data: {r.data}" for r in results]
    return "\n".join(summary)


@mcp.tool()
def flowdb_list_collections() -> str:
    """List all available collections."""
    names = db_instance.list_collections()
    if not names: return "No collections found."
    return "Available Collections:\n- " + "\n- ".join(names)


@mcp.tool()
def flowdb_delete(collection: str, key: str) -> str:
    """Delete a record by ID."""
    col = db_instance.collection(collection, GenericRecord)
    success = col.delete(key)
    if success:
        return f"Successfully deleted record {key} from {collection}."
    return f"Record {key} was not found."


app.mount("/mcp", mcp_asgi)


def start():
    uvicorn.run("flowdb.server.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
