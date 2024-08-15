import os
from pinecone import Pinecone
from typing import Any, Dict, List
from community.tools import BaseTool
from sentence_transformers import SentenceTransformer


class ArxivConnector(BaseTool):
    NAME = "arxiv_connector"

    def __init__(self):
        self.model = SentenceTransformer("thenlper/gte-large")
        self.index = (Pinecone(api_key= os.getenv("PINECONE_API_KEY"))).Index(os.getenv("PINECONE_INDEX"))

    @classmethod
    def is_available(cls) -> bool:
        return True

    async def call(self, parameters: dict, ctx: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        # Extract the preprocessed user query.
        query = parameters.get("query", "")

        # Embed the preprocessed user query to allow for retrieval.
        vec = self.model.encode(query)

        # Determine the k closest sources that may be relevant to the query.
        pinecone_search_result = self.index.query(
            namespace="ns1",
            vector=vec.tolist(),
            top_k=3,
            include_values=False,
            include_metadata=True,
        )

        final_result = []

        for match in pinecone_search_result['matches']:
            id = match['id']

            abstract = match['metadata']['abstract']
            title = match['metadata']['title']
            url = f'https://arxiv.org/abs/{id}'

            final_result.append(
                {
                    'text': abstract,
                    'title': title,
                    'url': url
                }
            )

        return final_result
