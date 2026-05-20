#!/usr/bin/env python3
"""
Embedding API Server for Qwen3-VL-Embedding
Port: configured by API_PORT
"""

import os

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
                if key in ('USE_EMBEDDING_GPU', 'EMBEDDING_CUDA_DEVICE', 'EMBEDDING_MODEL_PATH', 'MODEL_DTYPE', 'API_PORT'):
                    os.environ.setdefault(key, value)

# 设置 CUDA_VISIBLE_DEVICES（在导入 torch 之前）
USE_EMBEDDING_GPU = os.environ.get("USE_EMBEDDING_GPU", "true").lower() in ("true", "1", "yes")
EMBEDDING_CUDA_DEVICE = os.environ.get("EMBEDDING_CUDA_DEVICE", "0")

# 检查 GPU 可用性
if USE_EMBEDDING_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(EMBEDDING_CUDA_DEVICE)

import logging
import torch
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder

# 从环境变量读取配置
EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "/model/Qwen3-VL-Embedding-2B")
MODEL_DTYPE_STR = os.environ.get("MODEL_DTYPE", "bfloat16")
MODEL_DTYPE = torch.bfloat16 if MODEL_DTYPE_STR == "bfloat16" else torch.float32


def get_device_info():
    """获取设备信息字符串"""
    if USE_EMBEDDING_GPU and torch.cuda.is_available():
        # CUDA_VISIBLE_DEVICES already set, so cuda:0 maps to the specified GPU
        return f"cuda:0 (physical GPU {EMBEDDING_CUDA_DEVICE})"
    return "cpu"

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
    """Load the embedding model using environment config"""
    global model
    device_info = get_device_info()

    logger.info(f"Loading embedding model from {EMBEDDING_MODEL_PATH} on {device_info}...")

    model = Qwen3VLEmbedder(
        model_name_or_path=EMBEDDING_MODEL_PATH,
        use_cpu=not USE_EMBEDDING_GPU,
        # When CUDA_VISIBLE_DEVICES is set, use cuda:0 which maps to the specified GPU
        device_id=0 if USE_EMBEDDING_GPU else None,
        torch_dtype=MODEL_DTYPE,
    )

    logger.info(f"Embedding model loaded successfully on {device_info}")
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
    port = int(os.environ["API_PORT"])
    logger.info(f"Starting Embedding Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
