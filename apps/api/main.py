import time
from fastapi import FastAPI
from pydantic import BaseModel
from packages.graph.rag_graph import run_rag
from contextlib import asynccontextmanager
from packages.graph.nodes.rerank import get_reranker
from packages.core.config import settings
import uuid
import logging
import json

@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.reranker_enabled:
        get_reranker()
    yield

app = FastAPI(title="yag-rag", lifespan=lifespan)

logger = logging.getLogger("rag")
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

class AskRequest(BaseModel):
    query: str

class SourceItem(BaseModel):
    id: int | str
    title: str


class AskResponse(BaseModel):
    query: str
    retrieved_docs: list[dict]
    prepared_context: str
    answer: str
    llm_meta: dict
    retrieval_meta: dict
    meta: dict
    sources: list[SourceItem]
        
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    if settings.log_queries:
        logger.info(
            "request_start request_id=%s query=%s",
            request_id,
            req.query,
        )
    else:
        logger.info(
            "request_start request_id=%s",
            request_id,
        )

    result = await run_rag({
        "query": req.query,
        "meta": {
            "request_id": request_id
        }
    })

    sources = []
    for doc in result.get("retrieved_docs", []):
        payload = doc.get("payload", {}) or {}
        sources.append({
            "id": payload.get("id", doc.get("id")),
            "title": payload.get("title", "unknown"),
        })

    total_latency_ms = int((time.perf_counter() - start) * 1000)

    logger.info(
        "request_done request_id=%s latency_ms=%s docs=%s source_count=%s",
        request_id,
        total_latency_ms,
        len(result.get("retrieved_docs", [])),
        len(sources),
    )

    result["meta"] = {
        "total_latency_ms": total_latency_ms,
        "request_id": request_id
    }
    result["sources"] = sources
    result["meta"]["source_count"] = len(sources)
    
    logger.info(
        "evaluation_log request_id=%s query=%s answer=%s context_len=%s docs=%s",
        request_id,
        req.query if settings.log_queries else "",
        result.get("answer", "")[:200], 
        len(result.get("prepared_context", "")),
        len(result.get("retrieved_docs", [])),
    )
    
    eval_record = {
        "request_id": request_id,
        "query": req.query if settings.log_queries else "",
        "answer": result.get("answer", ""),
        "contexts": [
            (doc.get("payload", {}) or {}).get("text", "")
            for doc in result.get("retrieved_docs", [])
        ],
        "sources": [
            (doc.get("payload", {}) or {}).get("title", "")
            for doc in result.get("retrieved_docs", [])
        ],
        "latency_ms": total_latency_ms,
    }

    with open("evaluation.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(eval_record, ensure_ascii=False) + "\n")

    return result