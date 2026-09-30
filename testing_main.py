import os

# gpt-4o on this account is capped at 30,000 tokens per minute.
# DeepEval retries a 429 only twice, then gives up. Raise that before deepeval loads.
os.environ["DEEPEVAL_RETRY_MAX_ATTEMPTS"] = "8"
os.environ["DEEPEVAL_RETRY_INITIAL_SECONDS"] = "2"
os.environ["DEEPEVAL_RETRY_CAP_SECONDS"] = "30"

from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from retrieval import get_retrieved_docs, rag
import json
import time

#load golden dataset
golden_dataset = []
with open("evaluation/golden_dataset.jsonl","r",encoding = "utf-8") as f:
    for line in f:
        golden_dataset.append(json.loads(line))

#print(golden_dataset)

#Create deepeval test cases
test_cases = []

for item in golden_dataset:
    query, document_name, ideal_answer = item["query"], item["document_name"], item["ideal_answer"]
    
    actual_response, retrieved_chunks = rag(query,document_name) 

    test_case = LLMTestCase(
        input = query,
        actual_output = actual_response,
        expected_output = ideal_answer,
        retrieval_context = retrieved_chunks
    )

    test_cases.append(test_case)

#print(test_cases)

# One fresh set of metrics per question. async_mode=False keeps each metric's
# own judge calls in sequence. evaluate() would still run all five at once.
def make_metrics():
    shared = dict(
        threshold=0.7,
        include_reason=True,
        model="gpt-4o",
        async_mode=False,
    )
    return [
        ContextualRecallMetric(**shared),
        ContextualPrecisionMetric(**shared),
        ContextualRelevancyMetric(**shared),
        FaithfulnessMetric(**shared),
        AnswerRelevancyMetric(**shared),
    ]


# Pause between judge calls so five metrics on 13 questions stay under 30k TPM.
# About 13 questions x 5 metrics x 12 seconds, plus the model time itself.
PAUSE_SECONDS = 12

def save_records(records):
    with open("evaluation/eval_results.jsonl", "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# --------------------------------------------------
# 5. Run Evaluation
# --------------------------------------------------
records = []
total = len(test_cases)

for index, (item, test_case) in enumerate(zip(golden_dataset, test_cases), start=1):
    print(f"\nEvaluating {index}/{total}: {item['query']}")
    metric_rows = []

    for metric in make_metrics():
        print(f"  {metric.__class__.__name__}")
        metric.measure(test_case, _show_indicator=False)
        metric_rows.append({
            "name": metric.__name__,
            "score": metric.score,
            "passed": metric.is_successful(),
            "explanation": metric.reason,
        })
        time.sleep(PAUSE_SECONDS)

    records.append({
        "document_name": item["document_name"],
        "query": test_case.input,
        "ideal_answer": test_case.expected_output,
        "actual_answer": test_case.actual_output,
        "retrieved_chunks": test_case.retrieval_context,
        "passed": all(row["passed"] for row in metric_rows),
        "metrics": metric_rows,
    })
    save_records(records)

print(f"\nSaved {len(records)} cases to evaluation/eval_results.jsonl")


