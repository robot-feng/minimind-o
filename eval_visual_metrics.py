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


def _load_image_answers(results_path, references):
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
    return by_source


def _concept_recall(answer, concepts):
    hits = sum(any(_contains_alias(answer, alias) for alias in concept["aliases"])
               for concept in concepts)
    return hits / len(concepts), hits == len(concepts)


def score_visual_results(results_path, references_path):
    references = json.loads(Path(references_path).read_text(encoding="utf-8"))
    by_source = _load_image_answers(results_path, references)

    per_image = []
    for source, concepts in references.items():
        answers = by_source.get(source, [])
        if not answers:
            continue
        scored = [_concept_recall(answer, concepts) for answer in answers]
        recalls = [recall for recall, _ in scored]
        exact_hits = [hit for _, hit in scored]
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


def compare_visual_results(before_path, after_path, references_path):
    references = json.loads(Path(references_path).read_text(encoding="utf-8"))
    before = _load_image_answers(before_path, references)
    after = _load_image_answers(after_path, references)
    before_metrics = score_visual_results(before_path, references_path)
    after_metrics = score_visual_results(after_path, references_path)

    per_image = []
    for source, concepts in references.items():
        before_answers = before.get(source, [])
        after_answers = after.get(source, [])
        before_recalls = [_concept_recall(answer, concepts)[0] for answer in before_answers]
        after_recalls = [_concept_recall(answer, concepts)[0] for answer in after_answers]
        before_recall = sum(before_recalls) / len(before_recalls) if before_recalls else None
        after_recall = sum(after_recalls) / len(after_recalls) if after_recalls else None
        per_image.append({
            "source": source,
            "before_answers": before_answers,
            "after_answers": after_answers,
            "before_concept_recall": before_recall,
            "after_concept_recall": after_recall,
            "recall_delta": (
                after_recall - before_recall
                if before_recall is not None and after_recall is not None else None
            ),
        })

    metric_names = ("mean_concept_recall", "all_concepts_hit_rate")
    return {
        "before": {name: before_metrics[name] for name in metric_names},
        "after": {name: after_metrics[name] for name in metric_names},
        "delta": {
            name: after_metrics[name] - before_metrics[name]
            for name in metric_names
        },
        "missing_before": before_metrics["missing_images"],
        "missing_after": after_metrics["missing_images"],
        "per_image": per_image,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_jsonl", help="JSONL written by eval_omni.py --results_jsonl")
    parser.add_argument("--compare", help="second checkpoint JSONL; align and report per-image changes")
    parser.add_argument(
        "--references", default="dataset/eval_omni/visual_references.json",
        help="curated concept reference JSON",
    )
    args = parser.parse_args()
    scorer = compare_visual_results if args.compare else score_visual_results
    result = (
        scorer(args.results_jsonl, args.compare, args.references)
        if args.compare else scorer(args.results_jsonl, args.references)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
