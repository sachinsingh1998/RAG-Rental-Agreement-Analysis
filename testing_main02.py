from deepeval.metrics import (
    ContextualRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from retrieval import get_retrieved_docs, rag
from deepeval.evaluate import AsyncConfig
import json

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

#create metrics
contextual_relevancy = ContextualRelevancyMetric(
    threshold=0.7,
    include_reason=True,
    model="gpt-4o"
)
contextual_precision = ContextualPrecisionMetric(
    threshold=0.7,
    include_reason=True,
    model="gpt-4o"
)
contextual_recall = ContextualRecallMetric(
    threshold=0.7,
    include_reason=True,
    model="gpt-4o"
)
faithfulness = FaithfulnessMetric(
    threshold=0.7,
    include_reason=True,
    model="gpt-4o"
)
answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    include_reason=True,
    model="gpt-4o"
)


# --------------------------------------------------
# 5. Run Evaluation
# --------------------------------------------------
evaluate(
    test_cases=test_cases,
    metrics=[
    contextual_recall,
    contextual_precision,
    contextual_relevancy,
    faithfulness,
    answer_relevancy
    ],
    async_config= AsyncConfig(
        run_async=True,
        max_concurrent=5,  # Lower this if hitting rate limits
        #throttle_value=1    
        # Optional delay in seconds
    )
)



