"""Score curated visual concept coverage for MiniMind-O JSONL evaluations."""

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


def _normalise(text):
    return unicodedata.normalize("NFKC", text).casefold()


def _contains_alias(answer, alias):
    answer, alias = _normalise(answer), _normalise(alias).strip()
    if not alias:
        return False
    if re.fullmatch(r"[a-z0-9]+(?:[ '-][a-z0-9]+)*", alias):
        return re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", answer) is not None
    return alias in answer


def score_visual_results(results_path, references_path):
    references = json.loads(Path(references_path).read_text(encoding="utf-8"))
    by_source = defaultdict(list)
    with Path(results_path).open(encoding="utf-8") as results_file:
        for line_number, line in enumerate(results_file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("mode") != "image":
                continue
            source = Path(row["source"]).name
            if source not in references:
                raise ValueError(f"No visual reference for {source!r} (line {line_number})")
            by_source[source].append(row["answer"] or "")

    per_image = []
    for source, concepts in references.items():
        answers = by_source.get(source, [])
        if not answers:
            continue
        recalls, exact_hits = [], []
        for answer in answers:
            hits = sum(any(_contains_alias(answer, alias) for alias in concept["aliases"])
                       for concept in concepts)
            recalls.append(hits / len(concepts))
            exact_hits.append(hits == len(concepts))
        per_image.append({
            "source": source,
            "responses": len(answers),
            "mean_concept_recall": sum(recalls) / len(recalls),
            "all_concepts_hit_rate": sum(exact_hits) / len(exact_hits),
        })

    return {
        "expected_images": len(references),
        "evaluated_images": len(per_image),
        "response_count": sum(item["responses"] for item in per_image),
        "missing_images": [source for source in references if source not in by_source],
        "mean_concept_recall": (
            sum(item["mean_concept_recall"] for item in per_image) / len(per_image)
            if per_image else 0.0
        ),
        "all_concepts_hit_rate": (
            sum(item["all_concepts_hit_rate"] for item in per_image) / len(per_image)
            if per_image else 0.0
        ),
        "per_image": per_image,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_jsonl", help="JSONL written by eval_omni.py --results_jsonl")
    parser.add_argument(
        "--references", default="dataset/eval_omni/visual_references.json",
        help="curated concept reference JSON",
    )
    args = parser.parse_args()
    print(json.dumps(score_visual_results(args.results_jsonl, args.references),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
