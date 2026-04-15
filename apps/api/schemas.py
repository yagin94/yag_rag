from pydantic import BaseModel


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
