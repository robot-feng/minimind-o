"""Render a PNG summary from eval_visual_metrics.py --compare output."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1600, 900
COLORS = {
    "before": "#8b95a5",
    "after": "#3978b5",
    "grid": "#e3e7ec",
    "text": "#1f2937",
    "muted": "#526174",
}


def _font(size, bold=False):
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu") / filename,
        Path("/usr/share/fonts/dejavu") / filename,
    )
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _text_center(draw, center_x, top_y, text, font, fill):
    box = draw.multiline_textbbox((0, 0), text, font=font, align="center", spacing=2)
    draw.multiline_text(
        (center_x - (box[2] - box[0]) / 2, top_y),
        text,
        font=font,
        fill=fill,
        align="center",
        spacing=2,
    )


def _short_image_label(index):
    return f"{index:02d}"


def _validate_comparison(data):
    for section in ("before", "after"):
        for key in ("mean_concept_recall", "all_concepts_hit_rate"):
            if key not in data.get(section, {}):
                raise ValueError(f"comparison JSON is missing {section}.{key}")
    rows = data.get("per_image")
    if not isinstance(rows, list) or not rows:
        raise ValueError("comparison JSON must contain at least one per-image result")
    if data.get("missing_before") or data.get("missing_after"):
        raise ValueError("comparison JSON has missing image answers")
    for row in rows:
        for key in ("source", "before_concept_recall", "after_concept_recall"):
            if key not in row:
                raise ValueError(f"per-image result is missing {key}")
        if row["before_concept_recall"] is None or row["after_concept_recall"] is None:
            raise ValueError("comparison JSON has an image without both answers")


def render_comparison(comparison_path, output_path):
    data = json.loads(Path(comparison_path).read_text(encoding="utf-8"))
    _validate_comparison(data)
    rows = data["per_image"]

    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(34, bold=True)
    panel_font = _font(25, bold=True)
    label_font = _font(17)
    small_font = _font(15)
    metric_font = _font(16, bold=True)

    _text_center(draw, WIDTH / 2, 28, "MiniMind-O visual evaluation · Mini baseline vs full model", title_font, COLORS["text"])
    _text_center(draw, WIDTH / 2, 78, f"{len(rows)} fixed images · annotated concept coverage · greedy decoding", label_font, COLORS["muted"])

    left = (65, 160, 1035, 735)
    right = (1090, 160, 1535, 735)
    _text_center(draw, (left[0] + left[2]) / 2, 125, "Per-image concept recall", panel_font, COLORS["text"])
    _text_center(draw, (right[0] + right[2]) / 2, 125, "Aggregate metrics", panel_font, COLORS["text"])

    plot_top, plot_bottom = 205, 670
    for x0, x1 in (left[:1] + left[2:3], right[:1] + right[2:3]):
        for tick in (0, 25, 50, 75, 100):
            y = plot_bottom - (plot_bottom - plot_top) * tick / 100
            draw.line((x0 + 42, y, x1 - 12, y), fill=COLORS["grid"], width=1)
            draw.text((x0, y - 9), f"{tick}%", font=small_font, fill=COLORS["muted"])

    before_color, after_color = COLORS["before"], COLORS["after"]
    chart_left, chart_right = left[0] + 52, left[2] - 8
    group_width = (chart_right - chart_left) / len(rows)
    bar_width = min(25, group_width * 0.28)
    for index, row in enumerate(rows):
        center = chart_left + group_width * (index + 0.5)
        for value_key, offset, color in (
            ("before_concept_recall", -bar_width / 2, before_color),
            ("after_concept_recall", bar_width / 2, after_color),
        ):
            value = max(0.0, min(1.0, float(row[value_key]))) * 100
            height = (plot_bottom - plot_top) * value / 100
            x = center + offset
            draw.rectangle((x - bar_width / 2, plot_bottom - height, x + bar_width / 2, plot_bottom), fill=color)
        _text_center(draw, center, plot_bottom + 12, _short_image_label(index + 1), small_font, COLORS["text"])

    legend_y = 180
    draw.rectangle((left[0] + 265, legend_y, left[0] + 283, legend_y + 16), fill=before_color)
    draw.text((left[0] + 290, legend_y - 2), "sft_i2t_mini", font=small_font, fill=COLORS["text"])
    draw.rectangle((left[0] + 435, legend_y, left[0] + 453, legend_y + 16), fill=after_color)
    draw.text((left[0] + 460, legend_y - 2), "sft_omni", font=small_font, fill=COLORS["text"])

    metrics = (
        ("Mean concept recall", "mean_concept_recall"),
        ("All concepts hit", "all_concepts_hit_rate"),
    )
    agg_left, agg_right = right[0] + 48, right[2] - 18
    agg_group_width = (agg_right - agg_left) / len(metrics)
    agg_bar_width = 36
    for index, (label, key) in enumerate(metrics):
        center = agg_left + agg_group_width * (index + 0.5)
        for offset, section, color in ((-34, "before", before_color), (34, "after", after_color)):
            value = max(0.0, min(1.0, float(data[section][key]))) * 100
            height = (plot_bottom - plot_top) * value / 100
            x = center + offset
            draw.rectangle((x - agg_bar_width / 2, plot_bottom - height, x + agg_bar_width / 2, plot_bottom), fill=color)
            text_y = plot_bottom - height - 25 if value > 0 else plot_bottom - 24
            _text_center(draw, x, text_y, f"{value:.1f}%", metric_font, COLORS["text"])
        _text_center(draw, center, plot_bottom + 14, label, small_font, COLORS["text"])

    note = "Concept recall is a keyword-coverage diagnostic, not a measure of hallucinations, relations or fluency."
    _text_center(draw, WIDTH / 2, 805, note, label_font, COLORS["muted"])
    _text_center(draw, WIDTH / 2, 835, "Image IDs follow visual_references.json order. Review paired JSONL answers and audio before judging quality.", small_font, COLORS["muted"])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_json", help="JSON from eval_visual_metrics.py --compare")
    parser.add_argument("--output", required=True, help="destination PNG path")
    args = parser.parse_args()
    path = render_comparison(args.comparison_json, args.output)
    print(path)


if __name__ == "__main__":
    main()
