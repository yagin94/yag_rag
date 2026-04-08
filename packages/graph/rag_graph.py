from typing import TypedDict, List
from langgraph.graph import StateGraph, END

from packages.rag.retriever import retrieve
from packages.graph.nodes.prepare_context import prepare_context_node
from packages.graph.nodes.generate import generate_node
from packages.graph.nodes.rerank import rerank_node
from packages.core.config import settings


class RAGState(TypedDict, total=False):
    query: str
    retrieved_docs: List[dict]
    prepared_context: str
    answer: str
    llm_meta: dict
    retrieval_meta: dict
    meta: dict


def retrieve_node(state: RAGState) -> RAGState:
    normalized_query = state["query"].strip().lower()
    request_id = state.get("meta", {}).get("request_id", "no_request_id")
    docs = retrieve(state["query"], request_id=request_id)

    return {
        **state,
        "retrieved_docs": docs,
        "retrieval_meta": {
            "hybrid_enabled": settings.hybrid_enabled if normalized_query else False,
            "retrieved_count": len(docs),
            "dense_weight": settings.hybrid_dense_weight if normalized_query else 0.0,
            "lexical_weight": settings.hybrid_lexical_weight if normalized_query else 0.0,
            "normalized_query": normalized_query,
        },
    }


graph = StateGraph(RAGState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("rerank", rerank_node)
graph.add_node("prepare_context", prepare_context_node)
graph.add_node("generate", generate_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "rerank")
graph.add_edge("rerank", "prepare_context")
graph.add_edge("prepare_context", "generate")
graph.add_edge("generate", END)

compiled_graph = graph.compile()


async def run_rag(initial_state: dict) -> dict:
    return await compiled_graph.ainvoke(initial_state)


async def run_rag_pre_llm(initial_state: dict) -> dict:
    state = dict(initial_state)
    state = retrieve_node(state)
    state = await rerank_node(state)
    state = prepare_context_node(state)
    return state