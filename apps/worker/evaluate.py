import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import ContextUtilization
from openai import AsyncOpenAI
from ragas.llms import llm_factory

client = AsyncOpenAI(
    api_key="ollama",
    base_url="http://172.28.224.1:11434/v1",
)

llm = llm_factory(
    "llama3",
    provider="openai",
    client=client,
)

def load_eval_dataset(path: str = "evaluation.jsonl") -> Dataset:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    return Dataset.from_list(rows)


if __name__ == "__main__":
    ds = load_eval_dataset()

    client = AsyncOpenAI(
        api_key="ollama",
        base_url="http://172.28.224.1:11434/v1",
    )

    llm = llm_factory(
        "llama3",
        provider="openai",
        client=client,
    )

    scorer = ContextUtilization(llm=llm)

    scores = []
    for row in ds:
        result = scorer.score(
            user_input=row["query"],
            response=row["answer"],
            retrieved_contexts=row["contexts"],
        )
        print({
            "request_id": row["request_id"],
            "query": row["query"],
            "score": result.value,
        })
        scores.append(result.value)

    avg_score = sum(scores) / len(scores) if scores else 0.0
    print({"context_utilization_avg": avg_score, "num_rows": len(scores)})