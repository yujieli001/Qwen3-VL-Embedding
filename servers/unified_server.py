#!/usr/bin/env python3
"""
Unified API server for Qwen3-VL Embedding and Reranker.
Port: configured by API_PORT
"""

import logging
import os
from typing import Any, Dict, List, Optional


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_file = os.path.join(project_root, "embedding_reranker.env")
if os.path.exists(env_file):
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

USE_EMBEDDING_GPU = os.environ.get("USE_EMBEDDING_GPU", "true").lower() in ("true", "1", "yes")
USE_RERANKER_GPU = os.environ.get("USE_RERANKER_GPU", "true").lower() in ("true", "1", "yes")
EMBEDDING_CUDA_DEVICE = os.environ.get("EMBEDDING_CUDA_DEVICE", "0")
RERANKER_CUDA_DEVICE = os.environ.get("RERANKER_CUDA_DEVICE", "0")


def _visible_devices() -> List[str]:
    devices: List[str] = []
    if USE_EMBEDDING_GPU:
        devices.append(str(EMBEDDING_CUDA_DEVICE))
    if USE_RERANKER_GPU and str(RERANKER_CUDA_DEVICE) not in devices:
        devices.append(str(RERANKER_CUDA_DEVICE))
    return devices


VISIBLE_CUDA_DEVICES = _visible_devices()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(VISIBLE_CUDA_DEVICES)

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.qwen3_vl_embedding import Qwen3VLEmbedder
from src.models.qwen3_vl_reranker import Qwen3VLReranker


EMBEDDING_MODEL_PATH = os.environ.get("EMBEDDING_MODEL_PATH", "/model/Qwen3-VL-Embedding-2B")
RERANKER_MODEL_PATH = os.environ.get("RERANKER_MODEL_PATH", "/model/Qwen3-VL-Reranker-2B")
MODEL_DTYPE_STR = os.environ.get("MODEL_DTYPE", "bfloat16")
MODEL_DTYPE = torch.bfloat16 if MODEL_DTYPE_STR == "bfloat16" else torch.float32

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen3-VL Embedding and Reranker Service", version="1.0.0")

embedding_model: Optional[Qwen3VLEmbedder] = None
reranker_model: Optional[Qwen3VLReranker] = None


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


class OpenAIEmbeddingRequest(BaseModel):
    input: Any
    model: str = "qwen3-vl-embedding-2b"
    user: Optional[str] = None


class OpenAIEmbeddingData(BaseModel):
    object: str = "embedding"
    index: int
    embedding: List[float]


class OpenAIEmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[OpenAIEmbeddingData]
    model: str
    usage: Dict[str, int]


class V1RerankerRequest(BaseModel):
    query: Any
    documents: List[Any]
    model: str = "qwen3-vl-reranker-2b"
    batch_size: int = Field(default=1, gt=0)


class V1RerankerResult(BaseModel):
    index: int
    document: Any
    score: float
    relevance_score: float


class V1RerankerResponse(BaseModel):
    results: List[V1RerankerResult]
    model: str
    scores: List[float]


def _text_input(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {"text": str(value)}


def _embedding_inputs_from_openai(request: OpenAIEmbeddingRequest) -> List[Dict[str, Any]]:
    value = request.input
    if isinstance(value, str):
        return [{"text": value}]
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return [{"text": item} for item in value]
        if all(isinstance(item, dict) for item in value):
            return value
        if all(isinstance(item, int) for item in value):
            raise HTTPException(status_code=400, detail="Token ID inputs are not supported.")
        return [{"text": str(item)} for item in value]
    return [{"text": str(value)}]


def _token_count(value: Any) -> int:
    if isinstance(value, str):
        return len(value.split())
    if isinstance(value, list):
        return sum(_token_count(item) for item in value)
    if isinstance(value, dict):
        return sum(_token_count(item) for item in value.values())
    return 1


def _device_id(enabled: bool, physical_device: str) -> Optional[int]:
    if not enabled:
        return None
    return VISIBLE_CUDA_DEVICES.index(str(physical_device))


def _device_info(enabled: bool, physical_device: str) -> str:
    if enabled and torch.cuda.is_available():
        return f"cuda:{_device_id(enabled, physical_device)} (physical GPU {physical_device})"
    return "cpu"


def load_models():
    global embedding_model, reranker_model

    embedding_device = _device_info(USE_EMBEDDING_GPU, EMBEDDING_CUDA_DEVICE)
    logger.info(f"Loading embedding model from {EMBEDDING_MODEL_PATH} on {embedding_device}...")
    embedding_model = Qwen3VLEmbedder(
        model_name_or_path=EMBEDDING_MODEL_PATH,
        use_cpu=not USE_EMBEDDING_GPU,
        device_id=_device_id(USE_EMBEDDING_GPU, EMBEDDING_CUDA_DEVICE),
        torch_dtype=MODEL_DTYPE,
    )
    logger.info(f"Embedding model loaded successfully on {embedding_device}")

    reranker_device = _device_info(USE_RERANKER_GPU, RERANKER_CUDA_DEVICE)
    logger.info(f"Loading reranker model from {RERANKER_MODEL_PATH} on {reranker_device}...")
    reranker_model = Qwen3VLReranker(
        model_name_or_path=RERANKER_MODEL_PATH,
        use_cpu=not USE_RERANKER_GPU,
        device_id=_device_id(USE_RERANKER_GPU, RERANKER_CUDA_DEVICE),
        torch_dtype=MODEL_DTYPE,
    )
    logger.info(f"Reranker model loaded successfully on {reranker_device}")


@app.on_event("startup")
async def startup_event():
    load_models()


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "embedding": "running",
            "reranker": "running",
        },
    }


@app.get("/embedding/health")
async def embedding_health_check():
    return {"status": "healthy", "service": "embedding"}


@app.get("/reranker/health")
async def reranker_health_check():
    return {"status": "healthy", "service": "reranker"}


@app.post("/embeddings")
@app.post("/embedding/embeddings")
async def get_embeddings(input_data: EmbeddingInput) -> EmbeddingResponse:
    try:
        embeddings = embedding_model.process(input_data.inputs, normalize=input_data.normalize)
        return EmbeddingResponse(
            embeddings=embeddings.cpu().tolist(),
            shape=list(embeddings.shape),
        )
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/encode")
@app.post("/embedding/encode")
async def encode(inputs: List[Dict[str, Any]], normalize: bool = True):
    try:
        embeddings = embedding_model.process(inputs, normalize=normalize)
        return {
            "embeddings": embeddings.cpu().tolist(),
            "shape": list(embeddings.shape),
        }
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank")
@app.post("/reranker/rerank")
async def rerank(input_data: RerankerInput) -> RerankerResponse:
    try:
        scores = reranker_model.process({
            "query": input_data.query,
            "documents": input_data.documents,
            "instruction": input_data.instruction,
            "batch_size": input_data.batch_size,
        })
        return RerankerResponse(scores=scores)
    except Exception as e:
        logger.error(f"Error reranking: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embeddings")
async def v1_embeddings(request: OpenAIEmbeddingRequest) -> OpenAIEmbeddingResponse:
    try:
        inputs = _embedding_inputs_from_openai(request)
        embeddings = embedding_model.process(inputs, normalize=True)
        token_count = _token_count(request.input)
        return OpenAIEmbeddingResponse(
            data=[
                OpenAIEmbeddingData(index=index, embedding=embedding.cpu().tolist())
                for index, embedding in enumerate(embeddings)
            ],
            model=request.model,
            usage={
                "prompt_tokens": token_count,
                "total_tokens": token_count,
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating v1 embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rerank")
async def v1_rerank(request: V1RerankerRequest) -> V1RerankerResponse:
    try:
        inputs = {
            "query": _text_input(request.query),
            "documents": [_text_input(document) for document in request.documents],
            "batch_size": request.batch_size,
        }
        scores = reranker_model.process(inputs)
        results = [
            V1RerankerResult(
                index=index,
                document=document,
                score=float(scores[index]),
                relevance_score=float(scores[index]),
            )
            for index, document in enumerate(request.documents)
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        return V1RerankerResponse(
            results=results,
            model=request.model,
            scores=[result.score for result in results],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reranking v1 request: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def v1_models():
    return {
        "object": "list",
        "data": [
            {"id": "qwen3-vl-embedding-2b", "object": "model", "owned_by": "qwen3-vl"},
            {"id": "qwen3-vl-reranker-2b", "object": "model", "owned_by": "qwen3-vl"},
        ],
    }

