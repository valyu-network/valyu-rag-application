import os
import boto3
from pinecone import Pinecone
from typing import Any, Dict, List
from community.tools import BaseTool
from sentence_transformers import SentenceTransformer


class ArxivConnector(BaseTool):
    NAME = "arxiv_connector"

    def __init__(self):
        self.client = boto3.client('s3', region_name='eu-west-2')
        self.bucket_name = 'valyu-arxiv-articles'
        self.model = SentenceTransformer('src/community/other_models/Alibaba-NLP/gte-large-en-v1.5', trust_remote_code=True)
        self.index = (Pinecone(api_key= os.getenv("PINECONE_API_KEY"))).Index(os.getenv("PINECONE_INDEX"))
        self.top_sensitivity = 0.5
        self.bottom_sensitivity = 0.65

    @classmethod
    def is_available(cls) -> bool:
        return True

    async def call(self, parameters: dict, ctx: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        # Extract the preprocessed user query.
        query = parameters.get("query", "")

        # Embed the preprocessed user query to allow for retrieval.
        vec = self.model.encode(query)

        # Determine the k closest sources that may be relevant to the query.
        top_search_result = self.index.query(
            namespace="main",
            vector=vec.tolist(),
            top_k=2,
            include_values=False,
            include_metadata=True,
        )

        print(top_search_result)

        final_result = []

        for top_match in top_search_result['matches']:
            if top_match['score'] < self.top_sensitivity:
                continue

            id = top_match['id']

            # Determine the k closest sources that may be relevant to the query.
            bottom_search_result = self.index.query(
                namespace=id,
                vector=vec.tolist(),
                top_k=5,
                include_values=False,
                include_metadata=True,
            )

            print(bottom_search_result)

            for bottom_match in bottom_search_result['matches']:
                if bottom_match['score'] < self.bottom_sensitivity:
                    continue

                text = bottom_match['metadata']['raw_text']
                title = bottom_match['metadata']['title']
                url = f'https://arxiv.org/abs/{id}'

                final_result.append(
                    {
                        'text': text,
                        'title': title,
                        'url': url
                    }
                )

        if not final_result:
            return [
                {
                    'text': "No matches found."
                }
            ]
        else:
            return final_result