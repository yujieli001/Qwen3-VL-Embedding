#!/usr/bin/env python3
"""
OpenAI-compatible proxy for Qwen3-VL Embedding and Reranker services.
This allows Dify to use the models via standard OpenAI API format.

Port: 10013
"""

import os
import sys
import logging
from typing import List, Optional, Dict, Union, Any
from typing_extensions import Annotated
from functools import lru_cache

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from src.models.qwen3_vl_reranker import Qwen3VLReranker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Models cache
_models_cache: Dict[str, object] = {}


def get_embedding_model() -> Qwen3VLEmbedder:
    """Lazy load embedding model"""
    if "embedding" not in _models_cache:
        logger.info("Loading Embedding model on CPU...")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        _models_cache["embedding"] = Qwen3VLEmbedder(
            model_name_or_path="/model/Qwen3-VL-Embedding-2B",
            use_cpu=True,
            torch_dtype=torch.float32,
        )
        logger.info("Embedding model loaded successfully.")
    return _models_cache["embedding"]


def get_reranker_model() -> Qwen3VLReranker:
    """Lazy load reranker model"""
    if "reranker" not in _models_cache:
        logger.info("Loading Reranker model on CPU...")
        _models_cache["reranker"] = Qwen3VLReranker(
            model_name_or_path="/model/Qwen3-VL-Reranker-2B",
            use_cpu=True,
            torch_dtype=torch.float32,
        )
        logger.info("Reranker model loaded successfully.")
    return _models_cache["reranker"]


# ============== Pydantic Models (OpenAI-compatible) ==============

class EmbeddingRequest(BaseModel):
    input: Union[str, List[str], List[int], List[List[int]]]
    model: str = "qwen3-vl-embedding-2b"
    user: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "input": ["Hello world", "Test embedding"],
                "model": "qwen3-vl-embedding-2b"
            }
        }


class EmbeddingResponseData(BaseModel):
    index: int
    object: str = "embedding"
    embedding: List[float]


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingResponseData]
    model: str
    usage: dict


class RerankerRequest(BaseModel):
    query: str
    documents: List[str]
    model: str = "qwen3-vl-reranker-2b"


class RerankerResult(BaseModel):
    index: int
    document: str
    score: float


class RerankerResponse(BaseModel):
    results: List[RerankerResult]
    model: str


# ============== FastAPI App ==============

app = FastAPI(
    title="Qwen3-VL OpenAI-compatible API",
    description="OpenAI-compatible API for Qwen3-VL Embedding and Reranker",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Endpoints ==============

@app.get("/health")
async def health():
    return {"status": "healthy", "models": ["qwen3-vl-embedding-2b", "qwen3-vl-reranker-2b"]}


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint"""
    try:
        # Convert to format expected by embedder
        # Handle different input types: str, List[str], List[int], List[List[int]]
        if isinstance(request.input, str):
            inputs = [{"text": request.input}]
        elif isinstance(request.input, list):
            if all(isinstance(x, str) for x in request.input):
                inputs = [{"text": t} for t in request.input]
            elif all(isinstance(x, int) for x in request.input):
                # Token IDs - convert to text placeholder
                inputs = [{"text": "token_input"}]
            elif all(isinstance(x, list) and all(isinstance(i, int) for i in x) for x in request.input):
                # List of token ID lists
                inputs = [{"text": "token_input"} for _ in request.input]
            else:
                inputs = [{"text": str(x)} for x in request.input]
        else:
            inputs = [{"text": str(request.input)}]

        # Generate embeddings
        model = get_embedding_model()
        embeddings = model.process(inputs, normalize=True)

        # Build response
        data = []
        for i, emb in enumerate(embeddings.cpu()):
            data.append(EmbeddingResponseData(
                index=i,
                object="embedding",
                embedding=emb.tolist()
            ))

        # Calculate token usage (approximate)
        def count_tokens(inp):
            if isinstance(inp, str):
                return len(inp.split())
            elif isinstance(inp, list):
                if all(isinstance(x, int) for x in inp):
                    return len(inp)
                return sum(count_tokens(x) for x in inp)
            return 1

        token_count = count_tokens(request.input)

        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": token_count,
                "total_tokens": token_count
            }
        )
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rerank")
async def rerank(request: RerankerRequest):
    """Rerank endpoint (similar to Cohere rerank API)"""
    try:
        inputs = {
            "query": {"text": request.query},
            "documents": [{"text": doc} for doc in request.documents]
        }
        model = get_reranker_model()
        scores = model.process(inputs)

        results = [
            RerankerResult(
                index=i,
                document=doc,
                score=scores[i]
            )
            for i, doc in enumerate(request.documents)
        ]

        return RerankerResponse(results=results, model=request.model)
    except Exception as e:
        logger.error(f"Error reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "qwen3-vl-embedding-2b",
                "object": "model",
                "owned_by": "qwen3-vl",
                "permission": [{"allowed": True}]
            },
            {
                "id": "qwen3-vl-reranker-2b",
                "object": "model",
                "owned_by": "qwen3-vl",
                "permission": [{"allowed": True}]
            }
        ]
    }


if __name__ == "__main__":
    logger.info("Starting OpenAI-compatible proxy server on port 10013...")
    uvicorn.run(app, host="0.0.0.0", port=10013)
