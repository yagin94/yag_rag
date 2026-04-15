import time
import logging

from apps.llm.factory import get_llm_client
from packages.rag.prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger("rag.generate")


async def generate_node(state: dict) -> dict:
    request_id = state.get("meta", {}).get("request_id", "no_request_id")

    question = state.get("query", "").strip()
    context = state.get("prepared_context", "").strip()

    if not context:
        logger.warning(f"[{request_id}] Empty context → skip generation")

        return {
            "answer": "Không tìm thấy đủ thông tin trong dữ liệu truy xuất để trả lời câu hỏi này.",
            "llm_meta": {
                "reason": "empty_context",
            },
            "retrieval_meta": state.get("retrieval_meta", {}),
        }

    llm = get_llm_client()

    start = time.perf_counter()

    try:
        logger.info(
            "llm_start request_id=%s question_len=%s context_len=%s",
            request_id,
            len(question),
            len(context),
        )

        result = await llm.generate(
            system_prompt=build_system_prompt(),
            user_prompt=build_user_prompt(
                question=question,
                context=context,
            ),
            temperature=0.1,
            max_tokens=512,
        )

        latency_ms = int((time.perf_counter() - start) * 1000)

        logger.info(
            "llm_done request_id=%s latency_ms=%s model=%s",
            request_id,
            latency_ms,
            result.get("model"),
        )

        answer = result.get("text", "").strip()
        if not answer:
            answer = "Mô hình không sinh ra câu trả lời hợp lệ."
            
        if "Không tìm thấy thông tin trong tài liệu" in answer:
            answer = "Không tìm thấy thông tin trong tài liệu."    

        return {
            "answer": answer,
            "llm_meta": {
                "model": result.get("model"),
                "usage": result.get("usage"),
                "finish_reason": result.get("finish_reason"),
                "latency_ms": latency_ms,
            },
            "retrieval_meta": state.get("retrieval_meta", {}),
        }

    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)

        logger.error(
            "llm_error request_id=%s latency_ms=%s error=%s",
            request_id,
            latency_ms,
            str(e),
        )

        return {
            "answer": "Hệ thống sinh câu trả lời đang tạm lỗi, nhưng bước truy xuất dữ liệu đã chạy thành công.",
            "llm_meta": {
                "reason": "llm_error",
                "error": str(e),
                "latency_ms": latency_ms,
            },
            "retrieval_meta": state.get("retrieval_meta", {}),
        }


async def stream_generate_node(state: dict):
    request_id = state.get("meta", {}).get("request_id", "no_request_id")

    question = state.get("query", "").strip()
    context = state.get("prepared_context", "").strip()

    if not context:
        logger.warning(f"[{request_id}] Empty context → skip streaming generation")

        yield {
            "type": "final",
            "data": {
                "answer": "Không tìm thấy đủ thông tin trong dữ liệu truy xuất để trả lời câu hỏi này.",
                "llm_meta": {
                    "reason": "empty_context",
                },
                "retrieval_meta": state.get("retrieval_meta", {}),
            },
        }
        return

    llm = get_llm_client()
    start = time.perf_counter()
    full_text = ""
    last_chunk = None

    try:
        logger.info(
            "llm_stream_start request_id=%s question_len=%s context_len=%s",
            request_id,
            len(question),
            len(context),
        )

        async for chunk in llm.stream_generate(
            system_prompt=build_system_prompt(),
            user_prompt=build_user_prompt(
                question=question,
                context=context,
            ),
            temperature=0.1,
            max_tokens=512,
        ):
            token_text = chunk.get("text", "")
            if token_text:
                full_text += token_text
                yield {
                    "type": "token",
                    "data": token_text,
                }

            last_chunk = chunk

        latency_ms = int((time.perf_counter() - start) * 1000)

        logger.info(
            "llm_stream_done request_id=%s latency_ms=%s model=%s",
            request_id,
            latency_ms,
            (last_chunk or {}).get("model"),
        )

        answer = full_text.strip()
        if not answer:
            answer = "Mô hình không sinh ra câu trả lời hợp lệ."

        if "Không tìm thấy thông tin trong tài liệu" in answer:
            answer = "Không tìm thấy thông tin trong tài liệu."

        yield {
            "type": "final",
            "data": {
                "answer": answer,
                "llm_meta": {
                    "model": (last_chunk or {}).get("model"),
                    "usage": (last_chunk or {}).get("usage"),
                    "finish_reason": (last_chunk or {}).get("finish_reason"),
                    "latency_ms": latency_ms,
                },
                "retrieval_meta": state.get("retrieval_meta", {}),
            },
        }

    except Exception as e:
        latency_ms = int((time.perf_counter() - start) * 1000)

        logger.error(
            "llm_stream_error request_id=%s latency_ms=%s error=%s",
            request_id,
            latency_ms,
            str(e),
        )

        yield {
            "type": "error",
            "data": {
                "answer": "Hệ thống sinh câu trả lời đang tạm lỗi, nhưng bước truy xuất dữ liệu đã chạy thành công.",
                "llm_meta": {
                    "reason": "llm_error",
                    "error": str(e),
                    "latency_ms": latency_ms,
                },
                "retrieval_meta": state.get("retrieval_meta", {}),
            },
        }