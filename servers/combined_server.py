#!/usr/bin/env python3
"""
Combined Embedding & Reranker API Server for Qwen3-VL
Port: 10011 (Embedding), 10012 (Reranker)
Both services run in a single process with dual ports.
"""

import os
import sys

# 在导入 torch 之前设置 GPU 配置
# 读取.env 文件中的配置
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_file = os.path.join(project_root, "embedding_reranker.env")
if os.path.exists(env_file):
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                key, value = key.strip(), value.strip()
                if key in ('USE_EMBEDDING_GPU', 'USE_RERANKER_GPU', 'EMBEDDING_CUDA_DEVICE', 'RERANKER_CUDA_DEVICE'):
                    os.environ.setdefault(key, value)

# 设置 CUDA_VISIBLE_DEVICES（在导入 torch 之前）
USE_EMBEDDING_GPU = os.environ.get("USE_EMBEDDING_GPU", "true").lower() in ("true", "1", "yes")
USE_RERANKER_GPU = os.environ.get("USE_RERANKER_GPU", "true").lower() in ("true", "1", "yes")
EMBEDDING_CUDA_DEVICE = os.environ.get("EMBEDDING_CUDA_DEVICE", "0")
RERANKER_CUDA_DEVICE = os.environ.get("RERANKER_CUDA_DEVICE", "0")

# 根据配置设置可见设备（优先使用 embedding 的设备）
if not USE_EMBEDDING_GPU and not USE_RERANKER_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
elif USE_EMBEDDING_GPU and USE_RERANKER_GPU:
    # 两个都使用 GPU，设置两个设备
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{EMBEDDING_CUDA_DEVICE},{RERANKER_CUDA_DEVICE}"
elif USE_EMBEDDING_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(EMBEDDING_CUDA_DEVICE)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(RERANKER_CUDA_DEVICE)

import logging
import torch
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from src.models.qwen3_vl_reranker import Qwen3VLReranker

# 从环境变量读取配置
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "/model/Qwen3-VL-Embedding-2B")
RERANKER_MODEL_PATH = os.environ.get("RERANKER_MODEL_PATH", "/model/Qwen3-VL-Reranker-2B")
MODEL_DTYPE_STR = os.environ.get("MODEL_DTYPE", "bfloat16")
MODEL_DTYPE = torch.bfloat16 if MODEL_DTYPE_STR == "bfloat16" else torch.float32


def get_device_info(model_type: str = "embedding"):
    """获取设备信息字符串"""
    use_gpu = USE_EMBEDDING_GPU if model_type == "embedding" else USE_RERANKER_GPU
    cuda_dev = EMBEDDING_CUDA_DEVICE if model_type == "embedding" else RERANKER_CUDA_DEVICE
    if use_gpu and torch.cuda.is_available():
        # CUDA_VISIBLE_DEVICES already set, so cuda:0 maps to the specified GPU
        return f"cuda:0 (physical GPU {cuda_dev})"
    return "cpu"

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
    batch_size: int = Field(default=1, gt=0)


class RerankerResponse(BaseModel):
    scores: List[float]


class ModelInfo(BaseModel):
    embedding_model: str
    reranker_model: str
    device: str
    embedding_port: int
    reranker_port: int


def load_models():
    """Load both models using environment config"""
    global embedding_model, reranker_model

    embedding_device = get_device_info("embedding")
    reranker_device = get_device_info("reranker")

    logger.info(f"Loading embedding model from {EMBEDDING_MODEL_PATH} on {embedding_device}...")
    embedding_model = Qwen3VLEmbedder(
        model_name_or_path=EMBEDDING_MODEL_PATH,
        use_cpu=not USE_EMBEDDING_GPU,
        # When CUDA_VISIBLE_DEVICES is set, use cuda:0 which maps to the specified GPU
        device_id=0 if USE_EMBEDDING_GPU else None,
        torch_dtype=MODEL_DTYPE,
    )
    logger.info(f"Embedding model loaded successfully on {embedding_device}")

    logger.info(f"Loading reranker model from {RERANKER_MODEL_PATH} on {reranker_device}...")
    reranker_model = Qwen3VLReranker(
        model_name_or_path=RERANKER_MODEL_PATH,
        use_cpu=not USE_RERANKER_GPU,
        # When CUDA_VISIBLE_DEVICES is set, use cuda:0 which maps to the specified GPU
        device_id=0 if USE_RERANKER_GPU else None,
        torch_dtype=MODEL_DTYPE,
    )
    logger.info(f"Reranker model loaded successfully on {reranker_device}")

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
        device=f"embedding:{get_device_info('embedding')}, reranker:{get_device_info('reranker')}",
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
            "instruction": input_data.instruction,
            "batch_size": input_data.batch_size,
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
