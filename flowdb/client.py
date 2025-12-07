import os
import requests
from typing import List, Dict, Any, Optional, Type, TypeVar, Union, Generic
from pydantic import BaseModel

# Generic Type for user models
T = TypeVar("T", bound=BaseModel)


class FlowDBError(Exception):
    """Custom exception for database errors"""
    pass


class CollectionClient(Generic[T]):
    """
    Handles interactions for a specific collection (e.g., 'users').
    """

    def __init__(self, base_url: str, name: str, model: Type[T]):
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.model = model

    def _url(self, path: str) -> str:
        return f"{self.base_url}/v1/{self.name}/{path}"

    def upsert(self, record: T, vector: Optional[List[float]] = None) -> str:
        """
        Saves a record.
        """
        # 1. auto-detect ID from the model
        if not hasattr(record, "id"):
            raise FlowDBError("Model must have an 'id' field.")

        record_id = str(getattr(record, "id"))

        # 2. Prepare payload
        payload = {
            "id": record_id,
            "data": record.model_dump(),  # Convert Pydantic to Dict
            "vector": vector
        }

        # 3. Send Request
        resp = requests.post(self._url("upsert"), json=payload)

        if resp.status_code != 200:
            raise FlowDBError(f"Put failed: {resp.text}")

        return record_id

    def read(self, key: str) -> Optional[T]:
        """
        Retrieves a record by ID and converts it back to Pydantic.
        """
        resp = requests.get(self._url(f"read/{key}"))

        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise FlowDBError(f"Get failed: {resp.text}")

        # server returns: {"id": "...", "data": {...}, "vector": ...}
        # We extract "data" and validate it into the user's model
        wrapper = resp.json()
        return self.model.model_validate(wrapper["data"])

    def search(self, query: Union[str, List[float]], limit: int = 5) -> List[T]:
        """
        Semantic search. Accepts text OR a vector list.
        """
        endpoint = self._url(f"search?limit={limit}")

        # Handle Text Query vs Vector Query
        if isinstance(query, str):
            payload = {"query_text": query}
        else:
            # Future proofing if we add raw vector search back to API
            # For now, we assume text, but let's stick to the text API we built
            payload = {"query_text": str(query)}

        resp = requests.post(endpoint, json=payload)

        if resp.status_code != 200:
            raise FlowDBError(f"Search failed: {resp.text}")

        # Convert list of dicts back to Pydantic objects
        # Server returns: [{"id":..., "data":...}, ...]
        results = []
        for item in resp.json():
            obj = self.model.model_validate(item["data"])
            results.append(obj)

        return results

    def all(self, limit: int = 100, skip: int = 0) -> List[T]:
        """
        List records.
        """
        params = {"limit": limit, "skip": skip}
        resp = requests.get(self._url("all"), params=params)

        if resp.status_code != 200:
            raise FlowDBError(f"List failed: {resp.text}")

        results = []
        for item in resp.json():
            obj = self.model.model_validate(item["data"])
            results.append(obj)

        return results

    def delete(self, key: str) -> bool:
        """
        Deletes a record by ID.
        """
        resp = requests.delete(self._url(f"delete/{key}"))

        if resp.status_code == 404:
            return False
        if resp.status_code != 200:
            raise FlowDBError(f"Delete failed: {resp.text}")

        return True


class FlowDB:
    """
    The Main Client. Connects to your running FlowDB server.
    """

    def __init__(self, url: Optional[str] = None):
        # Auto-detect URL from env or default to localhost
        self.base_url = url or os.getenv("FLOWDB_URL", "http://localhost:8000")

    def collection(self, name: str, model: Type[T]) -> CollectionClient[T]:
        return CollectionClient(self.base_url, name, model)
