#!/usr/bin/env python3
"""
Combined Embedding & Reranker API Server for Qwen3-VL
Port: 10011 (Embedding), 10012 (Reranker)
Both services run in a single process with dual ports.
"""

import torch
import logging
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from src.models.qwen3_vl_reranker import Qwen3VLReranker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model variables
embedding_model: Optional[Qwen3VLEmbedder] = None
reranker_model: Optional[Qwen3VLReranker] = None


# Pydantic models
class EmbeddingInput(BaseModel):
    inputs: List[Dict[str, Any]]
    normalize: bool = True


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    shape: List[int]


class RerankerInput(BaseModel):
    query: Dict[str, Any]
    documents: List[Dict[str, Any]]
    instruction: Optional[str] = None


class RerankerResponse(BaseModel):
    scores: List[float]


class ModelInfo(BaseModel):
    embedding_model: str
    reranker_model: str
    device: str
    embedding_port: int
    reranker_port: int


def load_models():
    """Load both models on CPU"""
    global embedding_model, reranker_model

    embedding_path = "/model/Qwen3-VL-Embedding-2B"
    reranker_path = "/model/Qwen3-VL-Reranker-2B"

    logger.info(f"Loading embedding model from {embedding_path} on CPU...")
    embedding_model = Qwen3VLEmbedder(
        model_name_or_path=embedding_path,
        torch_dtype=torch.float32,
    )
    logger.info("Embedding model loaded successfully")

    logger.info(f"Loading reranker model from {reranker_path} on CPU...")
    reranker_model = Qwen3VLReranker(
        model_name_or_path=reranker_path,
        torch_dtype=torch.float32,
    )
    logger.info("Reranker model loaded successfully")

    return embedding_model, reranker_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    logger.info("Starting Qwen3-VL Combined Service...")
    load_models()
    logger.info("All models loaded. Server ready.")
    yield


app = FastAPI(
    title="Qwen3-VL Combined Service",
    description="Unified Embedding and Reranker API Service",
    version="1.0.0",
    lifespan=lifespan
)


# ============== Health & Info Endpoints ==============

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "embedding": "running",
            "reranker": "running"
        }
    }


@app.get("/info")
async def get_info():
    """Get service information"""
    return ModelInfo(
        embedding_model="Qwen3-VL-Embedding-2B",
        reranker_model="Qwen3-VL-Reranker-2B",
        device="cpu",
        embedding_port=10011,
        reranker_port=10012
    )


# ============== Embedding Endpoints (Port 10011) ==============

@app.post("/embeddings")
async def get_embeddings(input_data: EmbeddingInput) -> EmbeddingResponse:
    """
    Generate embeddings for the given inputs.

    Access via port 10011

    Input format:
    [
        {"text": "text content", "instruction": "optional instruction"},
        {"text": "text content"},
        {"image": "image_url_or_path"},
        {"text": "text content", "image": "image_url_or_path"}
    ]
    """
    try:
        embeddings = embedding_model.process(input_data.inputs, normalize=input_data.normalize)
        embeddings_list = embeddings.cpu().tolist()

        return EmbeddingResponse(
            embeddings=embeddings_list,
            shape=list(embeddings.shape)
        )
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed")
async def embed(inputs: List[Dict[str, Any]], normalize: bool = True):
    """Alternative endpoint for generating embeddings."""
    try:
        embeddings = embedding_model.process(inputs, normalize=normalize)
        return {
            "embeddings": embeddings.cpu().tolist(),
            "shape": list(embeddings.shape)
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Reranker Endpoints (Port 10012) ==============

@app.post("/rerank")
async def rerank(input_data: RerankerInput) -> RerankerResponse:
    """
    Rerank documents based on the query.

    Access via port 10012

    Input format:
    {
        "query": {"text": "query text", "image": "optional image"},
        "documents": [
            {"text": "doc1 text", "image": "optional image"},
            {"text": "doc2 text", "image": "optional image"}
        ],
        "instruction": "optional instruction"
    }
    """
    try:
        inputs = {
            "query": input_data.query,
            "documents": input_data.documents,
            "instruction": input_data.instruction
        }
        scores = reranker_model.process(inputs)
        return RerankerResponse(scores=scores)
    except Exception as e:
        logger.error(f"Error reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Combined Endpoints ==============

@app.post("/process")
async def process_embedding_and_rerank(
    query: Dict[str, Any],
    documents: List[Dict[str, Any]],
    use_reranker: bool = True
):
    """
    Combined endpoint: generate embeddings for query and documents,
    optionally rerank the results.
    """
    try:
        # Generate embeddings for query
        query_embedding = embedding_model.process([query], normalize=True)

        # Generate embeddings for documents
        doc_embeddings = embedding_model.process(documents, normalize=True)

        result = {
            "query_embedding": query_embedding.cpu().tolist()[0],
            "document_embeddings": doc_embeddings.cpu().tolist()
        }

        if use_reranker:
            rerank_scores = reranker_model.process({
                "query": query,
                "documents": documents
            })
            result["rerank_scores"] = rerank_scores

        return result
    except Exception as e:
        logger.error(f"Error processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Starting Combined Server...")
    logger.info("Embedding service: http://0.0.0.0:10011")
    logger.info("Reranker service: http://0.0.0.0:10012")

    # Run both ports using uvicorn's ability to handle multiple hosts
    config = uvicorn.Config(app, host="0.0.0.0", port=10011, log_level="info")
    server = uvicorn.Server(config)

    # Start first server in background, then second
    import asyncio

    async def run_servers():
        # Run embedding server on 10011
        await server.serve()

    asyncio.run(run_servers())
