def prepare_context_node(state: dict) -> dict:
    docs = state.get("retrieved_docs", [])

    if not docs:
        return {
            "prepared_context": "",
            "retrieval_meta": state.get("retrieval_meta", {}),
        }

    docs = sorted(
        docs,
        key=lambda x: x.get("rerank_score", x.get("score", 0.0)),
        reverse=True,
    )

    seen = set()
    filtered = []

    for doc in docs:
        text = (doc.get("payload", {}) or {}).get("text", "").strip()
        if not text:
            continue

        text_key = hash(text)
        if text_key in seen:
            continue

        seen.add(text_key)
        filtered.append(doc)

    filtered = filtered[:5]

    parts = []
    max_chars = 6000
    total_chars = 0

    for i, doc in enumerate(filtered, start=1):
        payload = doc.get("payload", {}) or {}
        doc_id = payload.get("id", f"doc_{i}")
        title = payload.get("title", "unknown")
        text = payload.get("text", "").strip()
        score = doc.get("score", 0.0)

        block = f"[{doc_id} | score={score:.3f} | source={title}]\n{text}\n"

        if total_chars + len(block) > max_chars:
            break

        parts.append(block)
        total_chars += len(block)

    return {
    **state,  
    "prepared_context": "\n".join(parts),
    "retrieval_meta": {
        **state.get("retrieval_meta", {}),
        "context_doc_count": len(parts),
        "context_sort_key": "rerank_score_or_score",
    },
}