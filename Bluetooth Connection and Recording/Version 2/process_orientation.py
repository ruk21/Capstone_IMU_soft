#!/usr/bin/env python3
"""
Turn the raw session.csv (accel + gyro) into ORIENTATION (roll / pitch / yaw)
for each board, using a 6-axis Mahony complementary filter.

Usage:   python process_orientation.py                (uses SESSION_NAME)
         python process_orientation.py other.csv

Reads from SESSION_DIR and writes to ORIENTATION_DIR (both set below):
  <session>_orientation.csv   time_s, device, qw,qx,qy,qz, roll_deg, pitch_deg, yaw_deg
  <session>_<board>_acc.png / _gyro.png / _orientation.png   one figure per board
                    per signal type (e.g. session1_left_acc.png)

All angles are RELATIVE TO THE POSE AT THE START of the recording, so keep the
board still for the first ~1 s (also used to estimate gyro bias).

Limits (inherent to a 6-axis IMU, no magnetometer):
  * roll & pitch are drift-free (gravity corrects them)
  * yaw is gyro-integrated only, so it slowly drifts
  * absolute position (metres) is NOT recoverable by double-integrating accel
"""
# ======================================================================
# EDIT THESE EACH SESSION
# ======================================================================
SESSION_NAME = "session1"   # reads  Session Data\<SESSION_NAME>.csv
# ======================================================================

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------- settings -----------------------------------
INPUT_FILE = f"{SESSION_NAME}.csv"   # looked up in SESSION_DIR
SESSION_DIR = r"C:\Users\Frank\OneDrive - University of Waterloo\01 University of Waterloo\01 University of Waterloo [BME]\01 Terms\4A\BME 461 [Capstone]\Code\Bluetooth Connection and Recording\Session Data"
ORIENTATION_DIR = r"C:\Users\Frank\OneDrive - University of Waterloo\01 University of Waterloo\01 University of Waterloo [BME]\01 Terms\4A\BME 461 [Capstone]\Code\Bluetooth Connection and Recording\Processing Data\Orientation"
SHOW_PLOTS = True            # open plot windows (PNGs are always saved too)
FS = 200.0            # output / filter rate, Hz (matches the ODR you recorded at)
BIAS_WINDOW_S = 1.0   # board must be still this long (found automatically, see below)
REST_GYRO_STD_DPS = 1.0   # "at rest" = gyro noise (std) below this ...
REST_ACC_STD_G = 0.02     # ... and accel noise (std) below this, over BIAS_WINDOW_S
KP = 2.0              # accel correction strength (higher = trusts gravity more)
ACC_TRUST_BAND = 0.2  # only correct with accel when |a| is within 1 g +/- this
# -----------------------------------------------------------------------------


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def euler_deg(q):
    w, x, y, z = q
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.degrees([roll, pitch, yaw])


def quat_from_up(a):
    """Quaternion that aligns measured 'up' (accel at rest) with world +Z."""
    a = a / np.linalg.norm(a)
    up = np.array([0.0, 0.0, 1.0])
    axis = np.cross(a, up)
    s = np.linalg.norm(axis)
    c = np.dot(a, up)
    if s < 1e-9:
        return np.array([1.0, 0, 0, 0]) if c > 0 else np.array([0.0, 1, 0, 0])
    axis /= s
    ang = np.arctan2(s, c)
    return np.concatenate([[np.cos(ang/2)], axis*np.sin(ang/2)])


def find_still_start(acc_g, gyr_dps, label=""):
    """
    Index where the board first sits still for BIAS_WINDOW_S. Anything before
    that (picking the board up, placing it down, pressing start) is ignored, so
    gyro bias and the gravity reference are measured while genuinely at rest.
    Uses the noise (std) rather than the level, so a constant gyro bias does not
    stop a still board from being recognised.
    """
    n = max(5, int(BIAS_WINDOW_S * FS))
    gs = pd.DataFrame(gyr_dps).rolling(n).std().max(axis=1)
    as_ = pd.DataFrame(acc_g).rolling(n).std().max(axis=1)
    ok = ((gs < REST_GYRO_STD_DPS) & (as_ < REST_ACC_STD_G)).to_numpy()
    hits = np.flatnonzero(ok)
    if len(hits) == 0:
        print(f"[{label}] WARNING: the board never sat still for {BIAS_WINDOW_S:.1f} s - "
              f"gyro bias and gravity are being guessed from the first second, so "
              f"orientation/position will be unreliable. Hold the board still for "
              f"~1 s at the start of each recording.")
        return 0
    s0 = int(hits[0]) - n + 1
    if s0 > 0:
        print(f"[{label}] board first sat still at {s0 / FS:.1f} s into the recording; "
              f"ignoring the {s0 / FS:.1f} s before that")
    return s0


def uniform_series(df):
    """
    The CSV timestamps are host-arrival times, which BLE batches/jitters, so
    they are poor for integration. Samples are spread evenly across the
    recording instead (correct on average even if a few were dropped).
    """
    n = len(df)
    t0, t1 = df.host_time_s.iloc[0], df.host_time_s.iloc[-1]
    t = np.linspace(t0, t1, n)
    return t, df[["x", "y", "z"]].to_numpy()


def mahony(acc, gyro_rad, dt, kp=None, trust=None):
    """kp may be a single gain or one gain per sample (array of length n)."""
    n = len(acc)
    kp = KP if kp is None else kp
    per_sample_kp = np.ndim(kp) > 0
    q = quat_from_up(acc[: int(FS)].mean(axis=0))
    q0 = q.copy()
    out = np.zeros((n, 4))
    for i in range(n):
        a, g = acc[i], gyro_rad[i]
        an = np.linalg.norm(a)
        ok = True if trust is None else bool(trust[i])
        if ok and an > 1e-6 and abs(an - 1.0) < ACC_TRUST_BAND:
            a = a / an
            w, x, y, z = q
            v = np.array([2*(x*z - w*y), 2*(w*x + y*z), w*w - x*x - y*y + z*z])
            g = g + (kp[i] if per_sample_kp else kp) * np.cross(a, v)
        q = q + 0.5 * dt * qmul(q, np.array([0.0, *g]))
        q /= np.linalg.norm(q)
        out[i] = qmul(qconj(q0), q)   # orientation relative to the start pose
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE
    # Relative names are looked up in the Session Data folder
    if not os.path.isabs(path):
        path = os.path.join(SESSION_DIR, path)
    if not os.path.exists(path):
        sys.exit(f"Could not find {path}\nCheck SESSION_DIR / INPUT_FILE at the "
                 f"top of this file.")
    process(path, show_plots=SHOW_PLOTS)


def process(path, show_plots=False):
    """Process one session CSV (full path); results go to ORIENTATION_DIR."""
    stem = os.path.splitext(os.path.basename(path))[0]   # prefix for output files
    df = pd.read_csv(path)
    folder = ORIENTATION_DIR
    os.makedirs(folder, exist_ok=True)

    rows, fig_data = [], {}
    for dev, d in df.groupby("device"):
        n_a, n_g = int((d.sensor == "acc").sum()), int((d.sensor == "gyro").sum())
        if min(n_a, n_g) < 2 * FS or abs(n_a - n_g) > 0.05 * max(n_a, n_g):
            print(f"[{dev}] SKIPPED: file has {n_a} accel and {n_g} gyro samples. "
                  f"Both are needed in about equal numbers (>= 2 s). The recording or "
                  f"download was incomplete - record again.")
            continue
        ta, acc = uniform_series(d[d.sensor == "acc"])
        tg, gyro = uniform_series(d[d.sensor == "gyro"])

        t = np.arange(max(ta[0], tg[0]), min(ta[-1], tg[-1]), 1.0 / FS)
        acc_i = np.column_stack([np.interp(t, ta, acc[:, k]) for k in range(3)])
        gyr_i = np.column_stack([np.interp(t, tg, gyro[:, k]) for k in range(3)])

        s0 = find_still_start(acc_i, gyr_i, dev)
        t, acc_i, gyr_i = t[s0:], acc_i[s0:], gyr_i[s0:]

        still = t < t[0] + BIAS_WINDOW_S
        bias = gyr_i[still].mean(axis=0)
        print(f"[{dev}] gyro bias removed (deg/s): {np.round(bias, 3)}")
        gyr_rad = np.radians(gyr_i - bias)

        q = mahony(acc_i, gyr_rad, 1.0 / FS)
        eul = np.array([euler_deg(qq) for qq in q])
        eul[:, 2] = np.degrees(np.unwrap(np.radians(eul[:, 2])))

        rows.append(pd.DataFrame({
            "time_s": t - t[0], "device": dev,
            "qw": q[:, 0], "qx": q[:, 1], "qy": q[:, 2], "qz": q[:, 3],
            "roll_deg": eul[:, 0], "pitch_deg": eul[:, 1], "yaw_deg": eul[:, 2],
        }))
        fig_data[dev] = {"t": t - t[0], "acc": acc_i, "gyro": gyr_i, "eul": eul}

    if not rows:
        print("No usable board data in this file, so no orientation was computed.")
        return
    out = pd.concat(rows, ignore_index=True)
    out_csv = os.path.join(folder, f"{stem}_orientation.csv")
    out.to_csv(out_csv, index=False, float_format="%.5f")

    if not show_plots:
        plt.switch_backend("Agg")

    def make_plot(dev, kind, t, data, labels, ylabel, title):
        fig, ax = plt.subplots(figsize=(10, 3.5))
        for k, name in enumerate(labels):
            ax.plot(t, data[:, k], label=name, linewidth=1)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{dev} - {title}")
        ax.grid(alpha=.3)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fname = os.path.join(folder, f"{stem}_{dev}_{kind}.png")
        fig.savefig(fname, dpi=120)
        print(f"Wrote {fname}")

    # One figure per board per signal type:
    #   <session>_<board>_acc.png, _gyro.png, _orientation.png
    for dev, d in fig_data.items():
        make_plot(dev, "acc", d["t"], d["acc"], ["x", "y", "z"],
                  "acceleration (g)", "accelerometer")
        make_plot(dev, "gyro", d["t"], d["gyro"], ["x", "y", "z"],
                  "angular velocity (deg/s)", "gyroscope (raw)")
        make_plot(dev, "orientation", d["t"], d["eul"], ["roll", "pitch", "yaw"],
                  "angle (deg)", "orientation (relative to start)")

    if show_plots:
        plt.show()
    else:
        plt.close("all")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
