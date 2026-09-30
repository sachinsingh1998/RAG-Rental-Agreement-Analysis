import json
import textwrap
from pathlib import Path

PATH = Path("evaluation/eval_results.jsonl")

SHORT_NAMES = {
    "Contextual Recall": "Recall",
    "Contextual Precision": "Prec",
    "Contextual Relevancy": "Relev",
    "Faithfulness": "Faith",
    "Answer Relevancy": "AnsRel",
}


def load_records(path):
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def clip(text, width):
    text = " ".join((text or "").split())
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def print_table(headers, rows):
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def format_row(row):
        return "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    print(format_row(headers))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(format_row(row))
    print()


def print_summary(records):
    metric_names = []
    for record in records:
        for metric in record.get("metrics", []):
            if metric["name"] not in metric_names:
                metric_names.append(metric["name"])

    headers = ["#", "Doc", "Pass", "Query"] + [
        SHORT_NAMES.get(name, name) for name in metric_names
    ]
    rows = []
    for i, record in enumerate(records, start=1):
        scores = {metric["name"]: metric.get("score") for metric in record.get("metrics", [])}
        row = [
            str(i),
            clip(record.get("document_name", ""), 16),
            "yes" if record.get("passed") else "no",
            clip(record.get("query", ""), 46),
        ]
        for name in metric_names:
            score = scores.get(name)
            row.append("—" if score is None else f"{score:.2f}")
        rows.append(row)

    print_table(headers, rows)


def print_text(title, text):
    print(title)
    for line in (text or "").splitlines() or [""]:
        print(f"  {line}")
    print()


def print_details(records):
    for i, record in enumerate(records, start=1):
        print("=" * 88)
        print(f"[{i}] {record.get('document_name')}   passed: {record.get('passed')}")
        print(f"Query: {record.get('query')}")
        print("-" * 88)
        print_text("Ideal answer", record.get("ideal_answer"))
        print_text("Actual answer", record.get("actual_answer"))

        print("Metrics")
        for metric in record.get("metrics", []):
            score = metric.get("score")
            score_text = "—" if score is None else f"{score:.2f}"
            status = "pass" if metric.get("passed") else "fail"
            print(f"  {metric['name']:<24} {score_text:>6}  {status}")
            for line in textwrap.wrap(metric.get("explanation") or "", width=80):
                print(f"    {line}")
            print()

        print("Retrieved chunks")
        for n, chunk in enumerate(record.get("retrieved_chunks") or [], start=1):
            preview = " ".join(chunk.split())
            wrapped = textwrap.fill(preview, width=80, subsequent_indent="      ")
            print(f"  [{n}] {wrapped}")
            print()


def main():
    records = load_records(PATH)
    print(f"{len(records)} cases from {PATH}\n")
    print_summary(records)
    print_details(records)


if __name__ == "__main__":
    main()