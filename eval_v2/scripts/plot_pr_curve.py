"""Plot reproducible Precision-Recall curves from scorer CSV output."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    groups = defaultdict(list)
    with args.curves.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            groups[int(row["k"])].append((float(row["recall"]), float(row["precision"])))
    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    selected = metrics.get("selected_joint_operating_point")
    width, height, left, top, plot_w, plot_h = 900, 560, 80, 55, 760, 430
    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]
    def xy(recall: float, precision: float) -> tuple[float, float]:
        return left + recall * plot_w, top + (1 - precision) * plot_h
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="450" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">FinAgent RAG calibration Precision–Recall curves</text>']
    for tick in range(6):
        value = tick / 5
        x, y = xy(value, value)
        svg += [f'<line x1="{x}" y1="{top}" x2="{x}" y2="{top+plot_h}" stroke="#ddd"/>',
                f'<line x1="{left}" y1="{y}" x2="{left+plot_w}" y2="{y}" stroke="#ddd"/>',
                f'<text x="{x}" y="{top+plot_h+22}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.1f}</text>',
                f'<text x="{left-12}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>']
    svg += [f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="black"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="black"/>']
    for index, (k, points) in enumerate(sorted(groups.items())):
        points.sort(); color = colors[index % len(colors)]
        coordinates = " ".join(f"{xy(x,y)[0]:.1f},{xy(x,y)[1]:.1f}" for x, y in points)
        svg.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
        lx, ly = 105 + (index % 3) * 115, 520 + (index // 3) * 18
        svg += [f'<line x1="{lx}" y1="{ly-4}" x2="{lx+22}" y2="{ly-4}" stroke="{color}" stroke-width="3"/>',
                f'<text x="{lx+28}" y="{ly}" font-family="sans-serif" font-size="11">K={k}</text>']
    if selected:
        sx, sy = xy(float(selected["recall"]), float(selected["precision"]))
        svg += [f'<circle cx="{sx}" cy="{sy}" r="5" fill="black"/>',
                f'<text x="{sx-8}" y="{sy+20}" text-anchor="end" font-family="sans-serif" font-size="10">provisional K={selected["k"]}, t={selected["threshold"]:.3f}</text>']
    svg += [f'<text x="{left+plot_w/2}" y="550" text-anchor="middle" font-family="sans-serif" font-size="13">Recall (evidence-group macro)</text>',
            f'<text x="18" y="{top+plot_h/2}" transform="rotate(-90 18 {top+plot_h/2})" text-anchor="middle" font-family="sans-serif" font-size="13">Precision (query macro)</text>', '</svg>']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(svg), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
