#!/usr/bin/env python3
"""
Reranker API Server for Qwen3-VL-Reranker
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
                if key in ('USE_RERANKER_GPU', 'RERANKER_CUDA_DEVICE', 'RERANKER_MODEL_PATH', 'MODEL_DTYPE', 'API_PORT'):
                    os.environ.setdefault(key, value)

# 设置 CUDA_VISIBLE_DEVICES（在导入 torch 之前）
USE_RERANKER_GPU = os.environ.get("USE_RERANKER_GPU", "true").lower() in ("true", "1", "yes")
RERANKER_CUDA_DEVICE = os.environ.get("RERANKER_CUDA_DEVICE", "0")

# 检查 GPU 可用性
if USE_RERANKER_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(RERANKER_CUDA_DEVICE)

import logging
import torch
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from src.models.qwen3_vl_reranker import Qwen3VLReranker

# 从环境变量读取配置
RERANKER_MODEL_PATH = os.environ.get("RERANKER_MODEL_PATH", "/model/Qwen3-VL-Reranker-2B")
MODEL_DTYPE_STR = os.environ.get("MODEL_DTYPE", "bfloat16")
MODEL_DTYPE = torch.bfloat16 if MODEL_DTYPE_STR == "bfloat16" else torch.float32


def get_device_info():
    """获取设备信息字符串"""
    if USE_RERANKER_GPU and torch.cuda.is_available():
        # CUDA_VISIBLE_DEVICES already set, so cuda:0 maps to the specified GPU
        return f"cuda:0 (physical GPU {RERANKER_CUDA_DEVICE})"
    return "cpu"

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
    """Load the reranker model using environment config"""
    global model
    device_info = get_device_info()

    logger.info(f"Loading reranker model from {RERANKER_MODEL_PATH} on {device_info}...")

    model = Qwen3VLReranker(
        model_name_or_path=RERANKER_MODEL_PATH,
        use_cpu=not USE_RERANKER_GPU,
        # When CUDA_VISIBLE_DEVICES is set, use cuda:0 which maps to the specified GPU
        device_id=0 if USE_RERANKER_GPU else None,
        torch_dtype=MODEL_DTYPE,
    )

    logger.info(f"Reranker model loaded successfully on {device_info}")
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
    port = int(os.environ["API_PORT"])
    logger.info(f"Starting Reranker Server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
