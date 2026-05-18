#!/usr/bin/env python3
"""Plot loss curves from an AlphaZero learner log.

Expected log format:
  [timestamp] Step: 12
  [timestamp] Losses: policy: 1.2345, value: 0.1234, l2: 0.0123, sum: 1.3702
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sys


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?inf|nan"
LINE_RE = re.compile(r"^\[(?P<timestamp>[^\]]+)\]\s*(?P<message>.*)$")
STEP_RE = re.compile(r"\bStep:\s*(?P<step>\d+)\b")
LOSSES_RE = re.compile(
    rf"Losses:\s*"
    rf"policy:\s*(?P<policy>{FLOAT_RE}),\s*"
    rf"value:\s*(?P<value>{FLOAT_RE}),\s*"
    rf"l2:\s*(?P<l2>{FLOAT_RE}),\s*"
    rf"sum:\s*(?P<sum>{FLOAT_RE})",
    re.IGNORECASE,
)

METRICS = ("policy", "value", "l2", "sum")
COLORS = {
    "policy": "#d55e00",
    "value": "#009e73",
    "l2": "#7a5195",
    "sum": "#0072b2",
}


@dataclass(frozen=True)
class LossPoint:
    update: int
    step: int | None
    timestamp: datetime | None
    policy: float
    value: float
    l2: float
    sum: float


def parse_timestamp(raw: str) -> datetime | None:
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def parse_loss_log(log_path: Path) -> list[LossPoint]:
    points: list[LossPoint] = []
    current_step: int | None = None

    with log_path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            line_match = LINE_RE.match(line.strip())
            if line_match is None:
                continue

            timestamp = parse_timestamp(line_match.group("timestamp"))
            message = line_match.group("message")

            step_match = STEP_RE.search(message)
            if step_match is not None:
                current_step = int(step_match.group("step"))
                continue

            losses_match = LOSSES_RE.search(message)
            if losses_match is None:
                continue

            points.append(
                LossPoint(
                    update=len(points) + 1,
                    step=current_step,
                    timestamp=timestamp,
                    policy=float(losses_match.group("policy")),
                    value=float(losses_match.group("value")),
                    l2=float(losses_match.group("l2")),
                    sum=float(losses_match.group("sum")),
                )
            )

    return points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot policy/value/l2/sum loss curves from log-learner.txt."
    )
    parser.add_argument(
        "log_path",
        nargs="?",
        default=Path("log-learner.txt"),
        type=Path,
        help="Path to learner log. Default: log-learner.txt",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=Path("loss_curve.png"),
        type=Path,
        help="Output image path. Default: loss_curve.png",
    )
    parser.add_argument(
        "--x-axis",
        choices=("update", "step", "time"),
        default="update",
        help=(
            "X-axis to use. 'update' is monotonic even if training restarts; "
            "'step' uses the logged Step value; 'time' uses log timestamps."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=METRICS,
        default=list(METRICS),
        help="Loss metrics to plot. Default: policy value l2 sum",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the plot window after saving.",
    )
    return parser.parse_args()


def get_x_values(points: list[LossPoint], x_axis: str) -> tuple[list[object], str]:
    if x_axis == "step":
        x_values = [point.step for point in points]
        if any(value is None for value in x_values):
            raise ValueError("Some loss entries do not have a preceding Step line.")
        return x_values, "Step"

    if x_axis == "time":
        x_values = [point.timestamp for point in points]
        if any(value is None for value in x_values):
            raise ValueError("Some loss entries do not have a parseable timestamp.")
        return x_values, "Time"

    return [point.update for point in points], "Loss update"


def plot_loss_curve(
    points: list[LossPoint],
    log_path: Path,
    output: Path,
    x_axis: str,
    metrics: list[str],
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    x_values, x_label = get_x_values(points, x_axis)

    plt.figure(figsize=(11, 6))
    for metric in metrics:
        y_values = [getattr(point, metric) for point in points]
        plt.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=1.8,
            markersize=3,
            label=metric,
            color=COLORS[metric],
        )

    plt.title(f"Loss curve from {log_path.name}")
    plt.xlabel(x_label)
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
    plt.legend()
    plt.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=150)

    if show:
        plt.show()


def main() -> int:
    args = parse_args()
    points = parse_loss_log(args.log_path)
    if not points:
        print(f"No loss entries found in {args.log_path}", file=sys.stderr)
        return 1

    try:
        plot_loss_curve(
            points,
            args.log_path,
            args.output,
            args.x_axis,
            args.metrics,
            args.show,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(f"Saved {args.output} ({len(points)} loss points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
