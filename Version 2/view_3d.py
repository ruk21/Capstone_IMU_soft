#!/usr/bin/env python3
"""
Interactive 3-D viewer for a processed session (drag to rotate, scroll to zoom).
Usage:  python view_3d.py <session>_position.csv
The GUI launches this automatically; it needs process_position.py to have run
with TRACK_Z = True so the CSV contains z_m.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    if len(sys.argv) < 2 or not os.path.exists(sys.argv[1]):
        sys.exit("Usage: python view_3d.py <session>_position.csv")
    path = sys.argv[1]
    df = pd.read_csv(path)
    stem = os.path.basename(path).replace("_position.csv", "")

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    pts = []
    for dev, d in df.groupby("device"):
        p = d[["x_m", "y_m", "z_m"]].to_numpy()
        pts.append(p)
        line, = ax.plot(p[:, 0], p[:, 1], p[:, 2], linewidth=1.5, label=dev)
        ax.scatter(*p[0], c=line.get_color(), marker="o", s=80)
        ax.scatter(*p[-1], c=line.get_color(), marker="s", s=80)
    allp = np.vstack(pts)
    c = (allp.max(axis=0) + allp.min(axis=0)) / 2
    h = max(float(np.ptp(allp, axis=0).max()) / 2, 0.02)
    ax.set_xlim(c[0] - h, c[0] + h)
    ax.set_ylim(c[1] - h, c[1] + h)
    ax.set_zlim(c[2] - h, c[2] + h)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:  # noqa: BLE001
        pass
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z - up (m)")
    ax.set_title(f"{stem} - 3-D path (circle = start, square = end)")
    ax.legend(loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
