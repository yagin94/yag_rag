import time
import uuid
import logging
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from apps.api.schemas import AskRequest, AskResponse
from packages.graph.rag_graph import run_rag, run_rag_pre_llm
from packages.graph.nodes.generate import stream_generate_node
from packages.graph.nodes.rerank import get_reranker
from packages.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.reranker_enabled:
        get_reranker()
    yield


app = FastAPI(title="yag-rag", lifespan=lifespan)

logger = logging.getLogger("rag")
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

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

    with open(settings.eval_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(eval_record, ensure_ascii=False) + "\n")

    return result


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    if settings.log_queries:
        logger.info(
            "stream_request_start request_id=%s query=%s",
            request_id,
            req.query,
        )
    else:
        logger.info(
            "stream_request_start request_id=%s",
            request_id,
        )

    async def event_generator():
        try:
            pre_llm_state = await run_rag_pre_llm({
                "query": req.query,
                "meta": {
                    "request_id": request_id
                }
            })

            sources = []
            for doc in pre_llm_state.get("retrieved_docs", []):
                payload = doc.get("payload", {}) or {}
                sources.append({
                    "id": payload.get("id", doc.get("id")),
                    "title": payload.get("title", "unknown"),
                })

            initial_event = {
                "type": "meta",
                "data": {
                    "request_id": request_id,
                    "sources": sources,
                    "retrieval_meta": pre_llm_state.get("retrieval_meta", {}),
                },
            }
            yield f"data: {json.dumps(initial_event, ensure_ascii=False)}\n\n"

            final_payload = None

            async for event in stream_generate_node(pre_llm_state):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                if event.get("type") in {"final", "error"}:
                    final_payload = event.get("data", {})

            total_latency_ms = int((time.perf_counter() - start) * 1000)

            end_event = {
                "type": "done",
                "data": {
                    "request_id": request_id,
                    "total_latency_ms": total_latency_ms,
                    "source_count": len(sources),
                },
            }
            yield f"data: {json.dumps(end_event, ensure_ascii=False)}\n\n"

            logger.info(
                "stream_request_done request_id=%s latency_ms=%s docs=%s source_count=%s",
                request_id,
                total_latency_ms,
                len(pre_llm_state.get("retrieved_docs", [])),
                len(sources),
            )

            if final_payload:
                eval_record = {
                    "request_id": request_id,
                    "query": req.query if settings.log_queries else "",
                    "answer": final_payload.get("answer", ""),
                    "contexts": [
                        (doc.get("payload", {}) or {}).get("text", "")
                        for doc in pre_llm_state.get("retrieved_docs", [])
                    ],
                    "sources": [
                        (doc.get("payload", {}) or {}).get("title", "")
                        for doc in pre_llm_state.get("retrieved_docs", [])
                    ],
                    "latency_ms": total_latency_ms,
                }

                with open(settings.eval_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(eval_record, ensure_ascii=False) + "\n")

        except Exception as e:
            total_latency_ms = int((time.perf_counter() - start) * 1000)

            logger.error(
                "stream_request_error request_id=%s latency_ms=%s error=%s",
                request_id,
                total_latency_ms,
                str(e),
            )

            error_event = {
                "type": "error",
                "data": {
                    "answer": "Hệ thống stream đang tạm lỗi.",
                    "llm_meta": {
                        "reason": "stream_error",
                        "error": str(e),
                        "latency_ms": total_latency_ms,
                    },
                },
            }
            yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
