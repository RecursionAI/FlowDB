import os
import lmdb
import struct
import numpy as np
import glob  # <--- NEW IMPORT
from typing import Type, List, Optional, Generic, TypeVar
from pydantic import BaseModel
from usearch.index import Index

from flowdb.core.vectorizer import Vectorizer

T = TypeVar("T", bound=BaseModel)


class FlowDBError(Exception):
    """Base exception for FlowDB"""
    pass


class Collection(Generic[T]):
    """
    Manages a single 'table' of data.
    Handles the synchronization between LMDB (Storage) and USearch (Vectors).
    """

    def __init__(self, name: str, model_cls: Type[T], db_path: str, vector_dim: int = 1536):
        self.name = name
        self.model_cls = model_cls
        self.db_path = db_path
        self.vector_dim = vector_dim

        # Ensure directories exist
        os.makedirs(db_path, exist_ok=True)

        # Initialize Vectorizer (auto-detects OPENAI_API_KEY)
        self.vectorizer = Vectorizer(provider="auto")

        # Setup LMDB Environment
        self.env = lmdb.open(
            os.path.join(db_path, "data.lmdb"),
            max_dbs=10,
            map_size=10 * 1024 * 1024 * 1024
        )

        # Open specific DBs
        self.main_db = self.env.open_db(f"{name}:main".encode(), create=True)
        self.id_map_db = self.env.open_db(f"{name}:idmap".encode(), create=True)
        self.meta_db = self.env.open_db(f"{name}:meta".encode(), create=True)

        # Setup USearch Index
        self.index_path = os.path.join(db_path, f"{name}.usearch")
        self.index = Index(ndim=vector_dim)

        if os.path.exists(self.index_path):
            self.index.load(self.index_path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _get_next_doc_id(self) -> int:
        """Atomic counter to generate unique integer IDs for USearch"""
        with self.env.begin(db=self.meta_db, write=True) as txn:
            cursor_key = b"next_doc_id"
            current = txn.get(cursor_key)
            if current is None:
                next_id = 1
            else:
                next_id = int(struct.unpack('>Q', current)[0]) + 1

            txn.put(cursor_key, struct.pack('>Q', next_id))
            return next_id

    def upsert(self, key: str, record: T, vector: Optional[np.ndarray] = None):
        """
        Writes data.
        If 'vector' is None, it AUTO-GENERATES one from the record's JSON.
        """
        json_data = record.model_dump_json()

        # Auto-Vectorize if missing
        if vector is None:
            vector = self.vectorizer.embed(json_data)

        # SAFETY CAST: Ensure it is ALWAYS a float32 numpy array
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        else:
            if vector.dtype != np.float32:
                vector = vector.astype(np.float32)

        # Get unique ID
        doc_id = self._get_next_doc_id()

        # Write Transaction (LMDB)
        with self.env.begin(write=True) as txn:
            txn.put(key.encode(), json_data.encode(), db=self.main_db)
            txn.put(struct.pack('>Q', doc_id), key.encode(), db=self.id_map_db)

        # Add to Vector Index (USearch)
        self.index.add(doc_id, vector)
        self.index.save(self.index_path)

    def read(self, key: str) -> Optional[T]:
        """Retrieves a record by primary key."""
        with self.env.begin(db=self.main_db, write=False) as txn:
            data = txn.get(key.encode())
            if not data:
                return None
            return self.model_cls.model_validate_json(data)

    def all(self, limit: int = 100, skip: int = 0) -> List[T]:
        """Scans records with pagination."""
        results = []
        with self.env.begin(db=self.main_db, write=False) as txn:
            cursor = txn.cursor()

            if skip > 0:
                for _ in range(skip):
                    if not cursor.next(): return []

            if skip == 0 and not cursor.first():
                return []

            count = 0
            for key, value in cursor.iternext(keys=True, values=True):
                if count >= limit: break
                obj = self.model_cls.model_validate_json(value)
                results.append(obj)
                count += 1

        return results

    def search(self, vector: Optional[np.ndarray] = None, query_text: Optional[str] = None, limit: int = 5) -> List[T]:
        """
        Performs vector search.
        """
        # Convert Text -> Vector
        if vector is None and query_text is not None:
            vector = self.vectorizer.embed(query_text)

        if vector is None:
            vector = self.vectorizer._mock_embed()

        if not isinstance(vector, np.ndarray):
            vector = np.array(vector, dtype=np.float32)
        else:
            if vector.dtype != np.float32:
                vector = vector.astype(np.float32)

        matches = self.index.search(vector, limit)

        results = []
        with self.env.begin(write=False) as txn:
            for match in matches:
                doc_id = match.key

                key_bytes = txn.get(struct.pack('>Q', doc_id), db=self.id_map_db)
                if not key_bytes:
                    continue

                data_bytes = txn.get(key_bytes, db=self.main_db)
                if data_bytes:
                    obj = self.model_cls.model_validate_json(data_bytes)
                    results.append(obj)

        return results

    def delete(self, key: str) -> bool:
        """
        Deletes a record.
        """
        with self.env.begin(write=True) as txn:
            return txn.delete(key.encode(), db=self.main_db)

    def close(self):
        """Closes connections cleanly."""
        self.env.close()
        self.index.save(self.index_path)


class FlowDB:
    """Main entry point."""

    def __init__(self, storage_path: str = "./flowdb_data"):
        self.storage_path = storage_path
        self.collections = {}

    def collection(self, name: str, model: Type[T]) -> Collection[T]:
        if name not in self.collections:
            self.collections[name] = Collection(name, model, self.storage_path)
        return self.collections[name]

    def list_collections(self) -> List[str]:
        """
        Scans the storage directory to find all active collections.
        Returns a list of names like ['users', 'orders'].
        """
        # Look for .usearch files (one per collection)
        pattern = os.path.join(self.storage_path, "*.usearch")
        files = glob.glob(pattern)

        collections = []
        for f in files:
            filename = os.path.basename(f)
            # Remove extension to get collection name
            name = os.path.splitext(filename)[0]
            collections.append(name)

        return sorted(collections)

    def close(self):
        """
        Closes all open collections and their underlying LMDB/USearch environments.
        This fixes the AttributeError on shutdown.
        """
        for name, col in self.collections.items():
            try:
                col.close()
                print(f"Closed collection: {name}")
            except Exception as e:
                print(f"Error closing collection {name}: {e}")
        self.collections.clear()
