def build_system_prompt() -> str:
    return (
        "You are a retrieval-augmented assistant. "
        "Answer only from the provided context. "
        "If the context is insufficient, say clearly that the answer is not available in the retrieved documents. "
        "Do not fabricate facts."
    )


def build_user_prompt(question: str, context: str) -> str:
    return f"""Question:
{question}

Context:
{context}

Instructions:
- Use only the context above.
- Each statement in your answer MUST reference its source using square-bracket citations like [1].
- Do not use parentheses around citations.
- Do NOT answer without citing sources.
- If the answer is not in the context, say clearly: "Không tìm thấy thông tin trong tài liệu."
- Keep the answer concise and accurate.
"""