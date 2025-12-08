import os
import requests
from typing import List, Dict, Any, Optional, Type, TypeVar, Union, Generic
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class FlowDBError(Exception):
    pass


class CollectionClient(Generic[T]):
    def __init__(self, session: requests.Session, base_url: str, name: str, model: Type[T]):
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.model = model

    def _url(self, path: str) -> str:
        return f"{self.base_url}/v1/{self.name}/{path}"

    def upsert(self, record: T, vector: Optional[List[float]] = None) -> str:
        if not hasattr(record, "id"):
            raise FlowDBError("Model must have an 'id' field.")

        record_id = str(getattr(record, "id"))
        payload = {
            "id": record_id,
            "data": record.model_dump(),
            "vector": vector
        }

        resp = self.session.post(self._url("upsert"), json=payload)

        if resp.status_code != 200:
            raise FlowDBError(f"Upsert failed: {resp.text}")

        return record_id

    def read(self, key: str) -> Optional[T]:
        resp = self.session.get(self._url(f"read/{key}"))

        if resp.status_code == 404:
            return None
        if resp.status_code != 200:
            raise FlowDBError(f"Read failed: {resp.text}")

        wrapper = resp.json()
        return self.model.model_validate(wrapper["data"])

    def search(self, query: Union[str, List[float]], limit: int = 5) -> List[T]:
        endpoint = self._url(f"search?limit={limit}")
        payload = {"query_text": query} if isinstance(query, str) else {"query_text": str(query)}

        resp = self.session.post(endpoint, json=payload)

        if resp.status_code != 200:
            raise FlowDBError(f"Search failed: {resp.text}")

        results = []
        for item in resp.json():
            obj = self.model.model_validate(item["data"])
            results.append(obj)
        return results

    def all(self, limit: int = 100, skip: int = 0) -> List[T]:
        params = {"limit": limit, "skip": skip}
        resp = self.session.get(self._url("all"), params=params)

        if resp.status_code != 200:
            raise FlowDBError(f"List failed: {resp.text}")

        results = []
        for item in resp.json():
            obj = self.model.model_validate(item["data"])
            results.append(obj)
        return results

    def delete(self, key: str) -> bool:
        resp = self.session.delete(self._url(f"delete/{key}"))
        if resp.status_code == 404:
            return False
        if resp.status_code != 200:
            raise FlowDBError(f"Delete failed: {resp.text}")
        return True


class FlowDB:
    def __init__(self, url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = url or os.getenv("FLOWDB_URL", "http://localhost:8000")
        self.api_key = api_key or os.getenv("FLOWDB_API_KEY")

        # Create a session that persists headers
        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})

    def collection(self, name: str, model: Type[T]) -> CollectionClient[T]:
        return CollectionClient(self.session, self.base_url, name, model)

    def list_collections(self) -> List[str]:
        resp = self.session.get(f"{self.base_url}/v1/collections")
        if resp.status_code != 200:
            raise FlowDBError(f"Failed to list collections: {resp.text}")
        return resp.json()["collections"]
