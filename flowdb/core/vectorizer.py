import os
import random
from typing import List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class Vectorizer:
    def __init__(self, provider: str = "mock", model: str = "text-embedding-3-small"):
        self.provider = provider
        self.model = model
        self.client = None

        if provider == "openai":
            if not OpenAI:
                raise ImportError("Run `pip install openai` to use OpenAI vectorizer")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                print("Warning: OPENAI_API_KEY not found. Falling back to mock vectors.")
                self.provider = "mock"
            else:
                self.client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> List[float]:
        """
        Turns text into a list of floats (vector).
        """
        if self.provider == "openai":
            try:
                # Real Embedding
                response = self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                return response.data[0].embedding
            except Exception as e:
                print(f"Vectorization Error: {e}. Falling back to mock.")
                return self._mock_embed()

        return self._mock_embed()

    def _mock_embed(self) -> List[float]:
        # Generates a random vector of size 1536 (OpenAI standard)
        return [random.random() for _ in range(1536)]