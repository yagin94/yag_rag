import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"\w+", text)
    return [t for t in tokens if len(t) > 1]


def keyword_score(query: str, title: str, text: str) -> float:
    query_tokens = tokenize(query)
    if not query_tokens:
        return 0.0

    doc_tokens = tokenize(f"{title} {text}")
    if not doc_tokens:
        return 0.0

    q_counts = Counter(query_tokens)
    d_counts = Counter(doc_tokens)

    overlap = 0.0
    for token, q_count in q_counts.items():
        overlap += min(q_count, d_counts.get(token, 0))

    return overlap / max(len(query_tokens), 1)