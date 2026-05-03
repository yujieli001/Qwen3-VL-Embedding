#!/usr/bin/env python3
"""
Embedding API Server for Qwen3-VL-Embedding
Port: 10011
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen3-VL Embedding Service", version="1.0.0")

# Global model variable
model: Optional[Qwen3VLEmbedder] = None


class EmbeddingInput(BaseModel):
    inputs: List[Dict[str, Any]]
    normalize: bool = True


class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    shape: List[int]


def load_model():
    """Load the embedding model on CPU"""
    global model
    model_path = "/model/Qwen3-VL-Embedding-2B"

    logger.info(f"Loading embedding model from {model_path} on CPU...")

    model = Qwen3VLEmbedder(
        model_name_or_path=model_path,
        use_cpu=True,
        torch_dtype=torch.float32,  # Use float32 for CPU
    )

    logger.info("Embedding model loaded successfully")
    return model


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "embedding"}


@app.post("/embeddings")
async def get_embeddings(input_data: EmbeddingInput) -> EmbeddingResponse:
    """
    Generate embeddings for the given inputs.

    Input format:
    [
        {"text": "text content", "instruction": "optional instruction"},
        {"text": "text content"},
        {"image": "image_url_or_path"},
        {"text": "text content", "image": "image_url_or_path"}
    ]
    """
    try:
        embeddings = model.process(input_data.inputs, normalize=input_data.normalize)

        # Convert tensor to list of lists
        embeddings_list = embeddings.cpu().tolist()

        return EmbeddingResponse(
            embeddings=embeddings_list,
            shape=list(embeddings.shape)
        )
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode")
async def encode(inputs: List[Dict[str, Any]], normalize: bool = True):
    """
    Alternative endpoint for generating embeddings.
    """
    try:
        embeddings = model.process(inputs, normalize=normalize)
        return {
            "embeddings": embeddings.cpu().tolist(),
            "shape": list(embeddings.shape)
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info("Starting Embedding Server on port 10011...")
    uvicorn.run(app, host="0.0.0.0", port=10011)
