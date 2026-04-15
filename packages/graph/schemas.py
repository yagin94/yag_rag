from typing import List, TypedDict


class RAGState(TypedDict, total=False):
    query: str
    retrieved_docs: List[dict]
    prepared_context: str
    answer: str
    llm_meta: dict
    retrieval_meta: dict
    meta: dict
