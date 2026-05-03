#!/usr/bin/env python3
"""
OpenAI-compatible proxy for Qwen3-VL Embedding and Reranker services.
This allows Dify to use the models via standard OpenAI API format.

Port: 10013
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import base64
import binascii
import io
import sys
import logging
from typing import List, Optional, Dict, Union, Any
from typing_extensions import Annotated
from functools import lru_cache

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

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
    input: Optional[Any] = None
    messages: Optional[List[Dict[str, Any]]] = None
    model: str = "qwen3-vl-embedding-2b"
    user: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "input": ["Hello world", "Test embedding"],
                "model": "qwen3-vl-embedding-2b"
            }
        }


def _image_from_data_url(url: str) -> Image.Image:
    if "," not in url:
        raise HTTPException(status_code=400, detail="Invalid data URL image.")

    header, encoded = url.split(",", 1)
    if not header.lower().startswith("data:image/"):
        raise HTTPException(status_code=400, detail="Only data:image URLs are supported for inline images.")

    try:
        raw = base64.b64decode(encoded, validate=True)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (binascii.Error, OSError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {e}") from e


def _normalize_image_url(url: str) -> Union[str, Image.Image]:
    if url.startswith("data:image/"):
        return _image_from_data_url(url)
    if url.startswith(("http://", "https://")):
        return url
    raise HTTPException(status_code=400, detail="image_url.url must be an http(s) URL or data:image base64 URL.")


def _messages_to_embedding_input(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    instruction_parts: List[str] = []
    text_parts: List[str] = []
    images: List[Union[str, Image.Image]] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content", [])
        if isinstance(content, str):
            content_items = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            content_items = content
        else:
            raise HTTPException(status_code=400, detail="messages[].content must be a string or a list.")

        for item in content_items:
            item_type = item.get("type")
            if item_type == "text":
                text = item.get("text", "")
                if role == "system":
                    instruction_parts.append(text)
                else:
                    text_parts.append(text)
            elif item_type == "image_url":
                image_url = item.get("image_url", {})
                if isinstance(image_url, dict):
                    url = image_url.get("url")
                else:
                    url = image_url
                if not isinstance(url, str):
                    raise HTTPException(status_code=400, detail="image_url.url is required.")
                images.append(_normalize_image_url(url))
            elif item_type == "image":
                image = item.get("image")
                if isinstance(image, str):
                    images.append(_normalize_image_url(image))
                else:
                    raise HTTPException(status_code=400, detail="image content must be a URL string.")
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported message content type: {item_type}")

    model_input: Dict[str, Any] = {}
    if instruction_parts:
        model_input["instruction"] = "\n".join(part for part in instruction_parts if part)
    if text_parts:
        model_input["text"] = "\n".join(part for part in text_parts if part)
    if images:
        model_input["image"] = images if len(images) > 1 else images[0]

    if not model_input:
        raise HTTPException(status_code=400, detail="messages must contain text or image content.")
    return model_input


def _embedding_inputs_from_request(request: EmbeddingRequest) -> List[Dict[str, Any]]:
    if request.messages is not None:
        return [_messages_to_embedding_input(request.messages)]

    if request.input is None:
        raise HTTPException(status_code=400, detail="Either input or messages is required.")

    if isinstance(request.input, str):
        return [{"text": request.input}]
    if isinstance(request.input, list):
        if all(isinstance(x, str) for x in request.input):
            return [{"text": t} for t in request.input]
        if all(isinstance(x, int) for x in request.input):
            raise HTTPException(status_code=400, detail="Token ID inputs are not supported by this embedding proxy.")
        if all(isinstance(x, list) and all(isinstance(i, int) for i in x) for x in request.input):
            raise HTTPException(status_code=400, detail="Token ID inputs are not supported by this embedding proxy.")
        if all(isinstance(x, dict) for x in request.input):
            return request.input
        return [{"text": str(x)} for x in request.input]
    if isinstance(request.input, dict):
        return [request.input]
    return [{"text": str(request.input)}]


def _count_request_tokens(request: EmbeddingRequest) -> int:
    def count_tokens(inp):
        if isinstance(inp, str):
            return len(inp.split())
        if isinstance(inp, list):
            if all(isinstance(x, int) for x in inp):
                return len(inp)
            return sum(count_tokens(x) for x in inp)
        if isinstance(inp, dict):
            return sum(count_tokens(v) for v in inp.values())
        return 1

    if request.messages is not None:
        return count_tokens(request.messages)
    return count_tokens(request.input)


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
    batch_size: int = Field(default=1, gt=0)


class RerankerResult(BaseModel):
    index: int
    document: str
    score: float
    relevance_score: float


class RerankerResponse(BaseModel):
    results: List[RerankerResult]
    model: str
    scores: List[float]


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
        inputs = _embedding_inputs_from_request(request)

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

        token_count = _count_request_tokens(request)

        return EmbeddingResponse(
            data=data,
            model=request.model,
            usage={
                "prompt_tokens": token_count,
                "total_tokens": token_count
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/rerank")
async def rerank(request: RerankerRequest):
    """Rerank endpoint (similar to Cohere rerank API)"""
    try:
        inputs = {
            "query": {"text": request.query},
            "documents": [{"text": doc} for doc in request.documents],
            "batch_size": request.batch_size,
        }
        model = get_reranker_model()
        scores = model.process(inputs)

        results = [
            RerankerResult(
                index=i,
                document=doc,
                score=float(scores[i]),
                relevance_score=float(scores[i]),
            )
            for i, doc in enumerate(request.documents)
        ]
        results.sort(key=lambda result: result.score, reverse=True)
        sorted_scores = [result.score for result in results]

        return RerankerResponse(results=results, model=request.model, scores=sorted_scores)
    except HTTPException:
        raise
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
