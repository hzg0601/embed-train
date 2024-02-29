

import os
import time
import tiktoken
import torch
import uvicorn

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from sse_starlette.sse import EventSourceResponse

import torch
# Set up limit request time
EventSourceResponse.DEFAULT_PING_INTERVAL = 1000

# set Embedding Model path
EMBEDDING_PATH = os.environ.get('EMBEDDING_PATH', '/home/pinming/models/bge-large-zh-v1.5-finetune-v1.0')



@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

## for Embedding
class EmbeddingRequest(BaseModel):
    input: Union[str,List[str]]
    model: str = "text-embedding-ada-002"


class EmbeddingResponse(BaseModel):
    data: list
    model: str
    object: str
    usage: dict


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def get_embeddings(request: EmbeddingRequest):
    embeddings = [embedding_model.encode(text) for text in request.input]
    embeddings = [embedding.tolist() for embedding in embeddings]

    def num_tokens_from_string(string: str) -> int:
        """
        Returns the number of tokens in a text string.
        use cl100k_base tokenizer
        """
        encoding = tiktoken.get_encoding('cl100k_base')
        num_tokens = len(encoding.encode(string))
        return num_tokens

    response = {
        "data": [
            {
                "object": "embedding",
                "embedding": embedding,
                "index": index
            }
            for index, embedding in enumerate(embeddings)
        ],
        "model": request.model,
        "object": "list",
        "usage": {
            "prompt_tokens": sum(len(text.split()) for text in request.input),  # how many characters in prompt
            "total_tokens": sum(num_tokens_from_string(text) for text in request.input),  # how many tokens (encoding)
        },
    }
    return response


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(
        id="text-embedding-ada-002"
    )
    return ModelList(
        data=[model_card]
    )



if __name__ == "__main__":

    # load Embedding
    embedding_model = SentenceTransformer(EMBEDDING_PATH, device="cuda")
    uvicorn.run(app, host='172.16.13.199', port=8090, workers=1)

