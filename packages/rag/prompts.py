def build_system_prompt() -> str:
    return (
        "You are a strict retrieval-augmented assistant.\n"
        "You MUST answer ONLY using the provided context.\n"
        "You are NOT allowed to use prior knowledge.\n"
        "If the context does not contain the answer, you MUST say so.\n"
        "Any statement not supported by context is strictly forbidden."
    )


def build_user_prompt(question: str, context: str) -> str:
    return f"""Question:
{question}

Context:
{context}

Rules (MANDATORY):
1. ONLY use information from the Context.
2. EVERY sentence MUST include at least one citation like [1]. Multiple citations are allowed if needed.
3. If ANY part of the question is NOT covered by the context, DO NOT GUESS.
4. If the context is insufficient, output EXACTLY this sentence and nothing else:
Không tìm thấy thông tin trong tài liệu.
5. DO NOT restate, repeat, or quote the question.
6. DO NOT add any introduction, explanation, or apology when context is insufficient.
7. DO NOT use outside knowledge under any circumstances.
8. The answer MUST be written in the SAME LANGUAGE as the question.

Answer:
"""