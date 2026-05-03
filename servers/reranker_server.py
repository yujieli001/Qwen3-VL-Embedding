#!/usr/bin/env python3
"""
Reranker API Server for Qwen3-VL-Reranker
Port: 10012
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from src.models.qwen3_vl_reranker import Qwen3VLReranker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen3-VL Reranker Service", version="1.0.0")

# Global model variable
model: Optional[Qwen3VLReranker] = None


class RerankerInput(BaseModel):
    query: Dict[str, Any]
    documents: List[Dict[str, Any]]
    instruction: Optional[str] = None
    batch_size: int = Field(default=1, gt=0)


class RerankerResponse(BaseModel):
    scores: List[float]


def load_model():
    """Load the reranker model on CPU"""
    global model
    model_path = "/model/Qwen3-VL-Reranker-2B"

    logger.info(f"Loading reranker model from {model_path} on CPU...")

    model = Qwen3VLReranker(
        model_name_or_path=model_path,
        use_cpu=True,
        torch_dtype=torch.float32,  # Use float32 for CPU
    )

    logger.info("Reranker model loaded successfully")
    return model


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "reranker"}


@app.post("/rerank")
async def rerank(input_data: RerankerInput) -> RerankerResponse:
    """
    Rerank documents based on the query.

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
            "instruction": input_data.instruction,
            "batch_size": input_data.batch_size,
        }

        scores = model.process(inputs)

        return RerankerResponse(scores=scores)
    except Exception as e:
        logger.error(f"Error reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Starting Reranker Server on port 10012...")
    uvicorn.run(app, host="0.0.0.0", port=10012)
