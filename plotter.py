"""Matplotlib rendering for the ``;curve`` command.

``generate_curve_plot`` returns a PNG :class:`io.BytesIO` that can be handed
straight to :class:`discord.File`. Rendering happens in a worker thread in the
bot so the event loop is never blocked.
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Any, Iterable, Mapping

# Serverless filesystems are read-only except for /tmp; keep matplotlib's font
# cache and config out of the home directory.
os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "ihave-matplotlib")
)

import matplotlib

matplotlib.use("Agg")  # headless rendering; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402  (import order is intentional)
from matplotlib import ticker  # noqa: E402
import numpy as np  # noqa: E402

_BACKGROUND = "#eaeaf2"
_GRID_COLOR = "#ffffff"
_LINE_COLOR = "#4c72b0"
_RAW_COLOR = "#9a9ab0"
_TEXT_COLOR = "#3d3d4e"
_SUBTLE_COLOR = "#6b6b7b"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"]


def _value(row: Any, key: str, default: Any = None) -> Any:
    if isinstance(row, Mapping):
        return row.get(key, default)
    return getattr(row, key, default)


def generate_curve_plot(handle: str, curve_data: Iterable[Any]) -> io.BytesIO:
    """Draw the raw and smoothed solve-time curves and return a PNG buffer."""
    rows: list[Any] = []
    for row in curve_data or ():
        rating = _value(row, "rating")
        smoothed = _value(row, "smoothed_minutes")
        if rating is None or smoothed is None:
            continue
        try:
            float(smoothed)
            int(rating)
        except (TypeError, ValueError):
            continue
        rows.append(row)
    rows.sort(key=lambda row: int(_value(row, "rating")))

    figure, axes = plt.subplots(figsize=(8.6, 4.8), dpi=150)
    figure.patch.set_facecolor(_BACKGROUND)
    axes.set_facecolor(_BACKGROUND)

    try:
        axes.grid(True, color=_GRID_COLOR, linewidth=1.4)
        axes.set_axisbelow(True)
        for side in ("top", "right"):
            axes.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axes.spines[side].set_color("#cfcfdd")
            axes.spines[side].set_linewidth(1.1)
        axes.tick_params(colors=_TEXT_COLOR, labelsize=9, length=0)
        axes.set_xlabel("Rating", color=_TEXT_COLOR, fontsize=11, labelpad=8)
        axes.set_ylabel("Minutes spent", color=_TEXT_COLOR, fontsize=11, labelpad=8)

        if not rows:
            axes.set_xlim(800, 3500)
            axes.set_ylim(0, 10)
            axes.set_xticks(np.arange(800, 3501, 200))
            axes.text(
                0.5,
                0.5,
                "No curve data yet",
                ha="center",
                va="center",
                transform=axes.transAxes,
                color=_SUBTLE_COLOR,
                fontsize=13,
            )
        else:
            ratings = np.array([int(_value(row, "rating")) for row in rows], dtype=float)
            smoothed = np.array([float(_value(row, "smoothed_minutes")) for row in rows], dtype=float)

            raw_x: list[float] = []
            raw_y: list[float] = []
            for row in rows:
                raw = _value(row, "raw_minutes")
                sample_size = _value(row, "sample_size", 0) or 0
                if raw is None or sample_size <= 0:
                    continue
                try:
                    raw_x.append(float(_value(row, "rating")))
                    raw_y.append(float(raw))
                except (TypeError, ValueError):
                    continue

            if raw_x:
                axes.plot(
                    raw_x,
                    raw_y,
                    color=_RAW_COLOR,
                    linestyle=(0, (4, 3)),
                    linewidth=1.3,
                    marker="o",
                    markersize=4,
                    markerfacecolor=_BACKGROUND,
                    markeredgewidth=1.2,
                    alpha=0.95,
                    label="Raw averages",
                    zorder=2,
                )

            axes.fill_between(ratings, 0, smoothed, color=_LINE_COLOR, alpha=0.06, zorder=1)
            axes.plot(
                ratings,
                smoothed,
                color=_LINE_COLOR,
                linewidth=2.6,
                marker="o",
                markersize=4.2,
                markerfacecolor=_LINE_COLOR,
                label=f"@{handle}",
                zorder=3,
            )

            x_min = max(800, int(np.floor(ratings.min() / 200.0) * 200))
            x_max = min(3500, int(np.ceil(ratings.max() / 200.0) * 200))
            if x_max <= x_min:
                x_max = min(3500, x_min + 200)
            axes.set_xticks(np.arange(x_min, x_max + 1, 200))
            axes.set_xlim(x_min - 50, x_max + 50)

            y_max = float(smoothed.max()) if smoothed.size else 10.0
            y_limit = max(10.0, y_max * 1.18)
            step = 10.0
            while y_limit / step > 7:
                step *= 2
            y_limit = step * float(np.ceil(y_limit / step))
            axes.set_ylim(0, y_limit)
            axes.set_yticks(np.arange(0, y_limit + step / 2, step))
            axes.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

            axes.legend(
                frameon=False,
                loc="upper left",
                fontsize=9.5,
                labelcolor=_TEXT_COLOR,
                handlelength=2.4,
            )

        axes.set_title(
            f"Solve-time curve · @{handle}",
            loc="left",
            color=_TEXT_COLOR,
            fontsize=14,
            fontweight="bold",
            pad=16,
        )

        figure.tight_layout(pad=1.5)
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            facecolor=figure.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.25,
        )
        buffer.seek(0)
        return buffer
    finally:
        plt.close(figure)


__all__ = ["generate_curve_plot"]
