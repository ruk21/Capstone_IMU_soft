#!/usr/bin/env python3
"""
MetaWear IMU recorder - one window for the whole workflow.

    python metawear_gui.py

  1. Tick the boards to use, click "Connect boards"
  2. Type a session name + duration, click "Record"
        - a short countdown lets you set the boards down and let go
        - "Stop early" ends the recording and keeps what was captured
  3. When recording ends, the data is saved and processed automatically:
        Session Data\\<name>.csv
        Processing Data\\Orientation\\<name>_*.csv / *.png
        Processing Data\\Positioning\\<name>_*.csv / *.png
  4. The boards stay connected, so you can record the next session straight away.

This file only adds the window. Everything else is done by the existing scripts,
which must stay in the same folder:
    record_two_metawear.py   (board addresses, sensor settings, folders)
    process_orientation.py   (orientation + Session Data / Orientation folders)
    process_position.py      (position + Positioning folder)
"""
import matplotlib
matplotlib.use("Agg")           # plots are saved to PNG; the GUI shows the PNGs

import glob
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback

import pandas as pd

try:
    from PIL import Image, ImageTk      # optional: smoother preview scaling
except ImportError:
    Image = ImageTk = None

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)                  # the MetaWear SDK keeps a ".metawear" cache here
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except ImportError:             # lets the logic be imported/tested without a display
    tk = None

import record_two_metawear as rec
import process_orientation as ori
import process_position as pos


# =============================================================================
# Logic (no GUI code): connecting, recording, stopping, processing
# =============================================================================
class _Cancelled(Exception):
    pass


class SessionController:
    """
    Runs the slow work on background threads and reports back through
    `self.events` (a queue the window polls):
        ("state", name)               name in STATES
        ("board", label, status)
        ("progress", phase, a, b)     phase "countdown": a=seconds left
                                      phase "recording": a=elapsed, b=duration
        ("log", text)
        ("done", session_stem)
    """

    def __init__(self):
        self.events = queue.Queue()
        self.boards = []
        self.state = "disconnected"
        self._stop = threading.Event()

    # ---- helpers ----------------------------------------------------------------
    def _set_state(self, s):
        self.state = s
        self.events.put(("state", s))

    def _teardown(self):
        for b in self.boards:
            try:
                b.stop()
            except Exception as e:  # noqa: BLE001
                print(f"[{b.label}] stop warning: {e}")
        for b in self.boards:
            b.disconnect()
            self.events.put(("board", b.label, "disconnected"))
        self.boards = []

    # ---- connect / disconnect -----------------------------------------------------
    def connect(self, labels):
        if self.state != "disconnected":
            return
        self._set_state("connecting")
        threading.Thread(target=self._connect_worker, args=(list(labels),),
                         daemon=True).start()

    def _connect_worker(self, labels):
        try:
            devices = {label: rec.DEVICES[label] for label in labels}
            missing = rec.scan_for_boards(devices.values(), rec.SCAN_TIMEOUT_S)
            if missing:
                print("Scan did not see: " + ", ".join(sorted(missing)))
                print("Check: board awake (press its button), nothing else connected "
                      "to it, within ~1 m, Windows Bluetooth on, MAC correct.")
                if rec.ABORT_IF_NOT_SEEN:
                    raise RuntimeError("board(s) not seen by the scan")
                print("Trying to connect anyway ...")
            for label, mac in devices.items():
                self.events.put(("board", label, "connecting"))
                board = rec.Board(label, mac)
                board.connect()                       # retries + rescans internally
                self.boards.append(board)
                self.events.put(("board", label, "connected"))
            print("All selected boards connected.")
            self._set_state("ready")
        except Exception as e:  # noqa: BLE001
            print(f"Connection failed: {e}")
            for label in labels:
                self.events.put(("board", label, "failed"))
            self._teardown()
            self._set_state("disconnected")

    def disconnect(self, wait=False):
        if not self.boards:
            self._set_state("disconnected")
            return
        self._set_state("disconnecting")
        t = threading.Thread(target=self._disconnect_worker, daemon=True)
        t.start()
        if wait:
            t.join(10)

    def _disconnect_worker(self):
        self._teardown()
        print("Boards disconnected.")
        self._set_state("disconnected")

    # ---- recording ----------------------------------------------------------------
    def start_recording(self, name, duration_s, delay_s, do_process=True, track_z=True):
        if self.state != "ready":
            return
        self._stop.clear()
        threading.Thread(target=self._record_worker,
                         args=(name, duration_s, delay_s, do_process, track_z),
                         daemon=True).start()

    def stop_early(self):
        """During the countdown this cancels; during recording it ends and saves."""
        self._stop.set()

    def _record_worker(self, name, duration_s, delay_s, do_process, track_z=True):
        path = os.path.join(rec.SESSION_DIR, f"{name}.csv")
        recorder = None
        cancelled = False
        started = False
        elapsed = 0.0
        try:
            os.makedirs(rec.SESSION_DIR, exist_ok=True)
            recorder = rec.Recorder(path)
            for b in self.boards:
                b.rec = recorder
                b.configure()

            self._set_state("countdown")
            if delay_s > 0:
                print("Place the boards where they will start and let go. Keep them "
                      "STILL for the first second or two of the recording.")
            for k in range(int(math.ceil(delay_s)), 0, -1):
                self.events.put(("progress", "countdown", k, 0))
                if self._stop.wait(1.0):
                    raise _Cancelled()

            self._set_state("recording")
            recorder.t0 = time.perf_counter()
            for b in self.boards:
                b.start()
            started = True
            t_start = time.perf_counter()
            print(f"Recording '{name}' for {duration_s:g} s ...")
            while True:
                elapsed = time.perf_counter() - t_start
                self.events.put(("progress", "recording", elapsed, duration_s))
                if elapsed >= duration_s:
                    break
                if self._stop.wait(0.1):
                    print(f"Stopped early after {elapsed:.1f} s - saving what was recorded.")
                    break
                dropped = [b.label for b in self.boards
                           if not self._still_connected(b)]
                if dropped:
                    print(f"WARNING: lost connection to {', '.join(dropped)} - "
                          f"ending the recording.")
                    break
        except _Cancelled:
            cancelled = True
            print("Recording cancelled before it started.")
        except Exception as e:  # noqa: BLE001
            print(f"Recording error: {e}")
            traceback.print_exc()
        finally:
            for b in self.boards:
                try:
                    b.stop()
                except Exception as e:  # noqa: BLE001
                    print(f"[{b.label}] stop warning: {e}")
                b.rec = None
            if recorder is not None:
                recorder.close()

        if cancelled or recorder is None:
            if os.path.exists(path):
                try:
                    os.remove(path)          # nothing useful in it
                except OSError:
                    pass
            self._set_state("ready")
            return

        print(f"Saved {path}")
        for (label, sensor), n in sorted(recorder.counts.items()):
            print(f"  {label:>8s} {sensor:<5s} {n} samples")
        silent = [b.label for b in self.boards
                  if not any(k[0] == b.label for k in recorder.counts)]
        if silent:
            print(f"WARNING: no data received from {', '.join(silent)}")
        if started and elapsed < 2.5:
            print("Note: that recording is very short - the processing needs about "
                  "1 s of the board sitting still at the start.")

        if do_process and recorder.counts:
            self._set_state("processing")
            self._process(path, track_z)
        self.events.put(("done", name))
        self._set_state("ready")

    def _still_connected(self, board):
        try:
            return bool(getattr(board.device, "is_connected", True))
        except Exception:  # noqa: BLE001
            return True

    # ---- processing ---------------------------------------------------------------
    def _process(self, path, track_z=True):
        # Set the height option here so it does not depend on what is typed in the
        # script file (also works with older process_position.py versions).
        if hasattr(pos, "TRACK_Z"):
            pos.TRACK_Z = bool(track_z)
        if hasattr(pos, "FLAT_PLANE"):                  # older script versions
            pos.FLAT_PLANE = not track_z
        print(f"Height (z) tracking for this run: {'ON' if track_z else 'OFF'}")
        for label, fn in (("orientation", ori.process), ("position", pos.process)):
            try:
                print(f"Processing {label} ...")
                fn(path, show_plots=False)
            except Exception as e:  # noqa: BLE001
                print(f"{label} processing failed: {e}")
                traceback.print_exc()
        self._report_height(path, track_z)

    def _report_height(self, path, track_z):
        """Read back the result and say plainly what height came out."""
        stem = os.path.splitext(os.path.basename(path))[0]
        csv_path = os.path.join(pos.POSITION_DIR, f"{stem}_position.csv")
        try:
            d = pd.read_csv(csv_path)
        except Exception as e:  # noqa: BLE001
            print(f"Could not read back {csv_path}: {e}")
            return
        for dev, g in d.groupby("device"):
            z = g["z_m"].to_numpy()
            if not track_z:
                continue
            if not z.any():
                print(f"[{dev}] WARNING: height (z) came out exactly 0 for the whole "
                      f"recording although tracking is ON. The position script that "
                      f"ran is: {os.path.abspath(pos.__file__)} - make sure that is "
                      f"the latest file, then close and reopen this window.")
            else:
                print(f"[{dev}] height (z) range {z.min():+.3f} to {z.max():+.3f} m, "
                      f"ends at {z[-1]:+.3f} m")

    def process_existing(self, name, track_z=True):
        """Re-run processing on a saved session (works with the boards disconnected)."""
        if self.state not in ("disconnected", "ready"):
            return
        prev = self.state

        def worker():
            self._set_state("processing")
            path = os.path.join(rec.SESSION_DIR, f"{name}.csv")
            if not os.path.exists(path):
                print(f"No saved session found: {path}")
            else:
                self._process(path, track_z)
                self.events.put(("done", name))
            self._set_state(prev)

        threading.Thread(target=worker, daemon=True).start()


# =============================================================================
# Window
# =============================================================================
STATUS_TEXT = {
    "disconnected": "Not connected",
    "connecting": "Connecting to the boards ... (can take up to a minute)",
    "ready": "Connected - ready to record",
    "countdown": "Get ready ...",
    "recording": "Recording ...",
    "processing": "Processing orientation and position ...",
    "disconnecting": "Disconnecting ...",
}


def clean_name(text):
    """Keep file names simple: letters, digits, '-' and '_'."""
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", text.strip()).strip("_")


def next_name(name):
    """session3 -> session4, run_07 -> run_08, trial -> trial_2"""
    m = re.search(r"(\d+)$", name)
    if m:
        digits = m.group(1)
        return name[: m.start()] + str(int(digits) + 1).zfill(len(digits))
    return name + "_2"


class _Tee:
    """Copies everything printed into the window's log (and the console)."""

    def __init__(self, original, events):
        self.original, self.events = original, events

    def write(self, text):
        if self.original is not None:
            try:
                self.original.write(text)
            except Exception:  # noqa: BLE001
                pass
        if text:
            self.events.put(("log", text))
        return len(text)

    def flush(self):
        if self.original is not None:
            try:
                self.original.flush()
            except Exception:  # noqa: BLE001
                pass


class App:
    PREVIEW_W, PREVIEW_H = 480, 400

    def __init__(self, root, ctl):
        self.root, self.ctl = root, ctl
        self._quit_when_idle = False
        self._preview_files = {}
        root.title("MetaWear IMU Recorder")

        self.var_name = tk.StringVar(value=clean_name(rec.SESSION_NAME) or "session1")
        self.var_duration = tk.StringVar(value=f"{rec.DURATION_S:g}")
        self.var_delay = tk.StringVar(value=f"{rec.START_DELAY_S:g}")
        self.var_process = tk.BooleanVar(value=True)
        self.var_trackz = tk.BooleanVar(value=True)
        self.var_auto3d = tk.BooleanVar(value=True)   # open 3-D viewer after each test
        self._last_stem = None
        self.var_preview = tk.StringVar(value="")
        self.board_vars = {label: tk.BooleanVar(value=True) for label in rec.DEVICES}
        self.board_status = {}

        self._build()
        self._apply_state("disconnected")
        z = getattr(pos, "TRACK_Z", None)
        print(f"Using {os.path.abspath(pos.__file__)}")
        if z is None:
            print("WARNING: this process_position.py is an OLD version (it has no "
                  "TRACK_Z setting), so height (z) will always be 0. Replace it with "
                  "the latest file and restart this window.")
        else:
            print(f"Height (z) tracking is {'ON' if z else 'OFF - set TRACK_Z = True'}.")
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        root.after(100, self._poll)

    # ---- layout -------------------------------------------------------------------
    def _build(self):
        root = self.root
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)
        left = ttk.Frame(root)
        left.grid(row=0, column=0, sticky="nsew", padx=8, pady=6)
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew", padx=8, pady=6)

        # 1. boards
        f = ttk.LabelFrame(left, text="1. Boards")
        f.pack(fill="x", pady=4)
        self.board_checks = []
        for label, mac in rec.DEVICES.items():
            row = ttk.Frame(f)
            row.pack(fill="x", padx=6, pady=2)
            cb = ttk.Checkbutton(row, text=f"{label}  ({mac})",
                                 variable=self.board_vars[label])
            cb.pack(side="left")
            self.board_checks.append(cb)
            st = ttk.Label(row, text="not connected", foreground="gray")
            st.pack(side="right")
            self.board_status[label] = st
        row = ttk.Frame(f)
        row.pack(fill="x", padx=6, pady=4)
        self.btn_connect = ttk.Button(row, text="Connect boards", command=self._on_connect)
        self.btn_connect.pack(side="left")
        self.btn_disconnect = ttk.Button(row, text="Disconnect", command=self._on_disconnect)
        self.btn_disconnect.pack(side="left", padx=6)

        # 2. recording
        f = ttk.LabelFrame(left, text="2. Record")
        f.pack(fill="x", pady=4)
        grid = ttk.Frame(f)
        grid.pack(fill="x", padx=6, pady=4)
        self.entries = []
        for r, (text, var, width) in enumerate([
                ("Session name", self.var_name, 22),
                ("Duration (s)", self.var_duration, 8),
                ("Start delay (s)", self.var_delay, 8)]):
            ttk.Label(grid, text=text).grid(row=r, column=0, sticky="w", pady=2)
            e = ttk.Entry(grid, textvariable=var, width=width)
            e.grid(row=r, column=1, sticky="w", padx=6, pady=2)
            self.entries.append(e)
        self.chk_process = ttk.Checkbutton(
            f, text="Process automatically after recording",
            variable=self.var_process)
        self.chk_process.pack(anchor="w", padx=6)
        self.chk_trackz = ttk.Checkbutton(
            f, text="Track height (z) - untick for flat table-top work",
            variable=self.var_trackz)
        self.chk_trackz.pack(anchor="w", padx=6)
        row = ttk.Frame(f)
        row.pack(fill="x", padx=6, pady=6)
        self.btn_record = ttk.Button(row, text="\u25cf Record", command=self._on_record)
        self.btn_record.pack(side="left")
        self.btn_stop = ttk.Button(row, text="Stop early (keep data)", command=self._on_stop)
        self.btn_stop.pack(side="left", padx=6)
        self.progress = ttk.Progressbar(f, maximum=100)
        self.progress.pack(fill="x", padx=6, pady=2)
        self.lbl_progress = ttk.Label(f, text="")
        self.lbl_progress.pack(anchor="w", padx=6)
        self.lbl_status = ttk.Label(f, text="")
        self.lbl_status.pack(anchor="w", padx=6, pady=(0, 4))

        # 3. results
        f = ttk.LabelFrame(left, text="3. Results")
        f.pack(fill="x", pady=4)
        row = ttk.Frame(f)
        row.pack(fill="x", padx=6, pady=4)
        for text, getter in (("Session Data", lambda: rec.SESSION_DIR),
                             ("Orientation", lambda: ori.ORIENTATION_DIR),
                             ("Positioning", lambda: pos.POSITION_DIR)):
            ttk.Button(row, text=text,
                       command=lambda g=getter: self._open_folder(g())).pack(side="left", padx=2)
        self.btn_reprocess = ttk.Button(f, text="Re-process saved session (name above)",
                                        command=self._on_reprocess)
        self.btn_reprocess.pack(anchor="w", padx=6, pady=(0, 2))
        row2 = ttk.Frame(f)
        row2.pack(fill="x", padx=6, pady=(0, 4))
        ttk.Button(row2, text="Show plots (no re-process)",
                   command=self._on_show_plots).pack(side="left", padx=2)
        ttk.Button(row2, text="Open 3D plot (interactive)",
                   command=self._on_open_3d).pack(side="left", padx=2)
        ttk.Checkbutton(f, text="Open interactive 3D plot automatically after each test",
                        variable=self.var_auto3d).pack(anchor="w", padx=6, pady=(0, 6))

        # preview
        ttk.Label(right, text="Preview").pack(anchor="w")
        self.cmb_preview = ttk.Combobox(right, textvariable=self.var_preview,
                                        state="readonly", width=48)
        self.cmb_preview.pack(fill="x", pady=2)
        self.cmb_preview.bind("<<ComboboxSelected>>", lambda _e: self._show_selected_preview())
        self.lbl_image = ttk.Label(right, text="(plots appear here after processing)",
                                   anchor="center")
        self.lbl_image.pack(fill="both", expand=True)

        # log
        lf = ttk.LabelFrame(root, text="Log")
        lf.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=6)
        self.log = ScrolledText(lf, height=12, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True, padx=4, pady=4)

    # ---- state ---------------------------------------------------------------------
    def _apply_state(self, s):
        idle = s in ("disconnected", "ready")
        for cb in self.board_checks:
            cb.config(state="normal" if s == "disconnected" else "disabled")
        self.btn_connect.config(state="normal" if s == "disconnected" else "disabled")
        self.btn_disconnect.config(state="normal" if s == "ready" else "disabled")
        self.btn_record.config(state="normal" if s == "ready" else "disabled")
        self.btn_stop.config(state="normal" if s in ("countdown", "recording") else "disabled")
        self.btn_reprocess.config(state="normal" if idle else "disabled")
        self.chk_process.config(state="normal" if idle else "disabled")
        self.chk_trackz.config(state="normal" if idle else "disabled")
        for e in self.entries:
            e.config(state="normal" if idle else "disabled")
        self.lbl_status.config(text=STATUS_TEXT.get(s, s))
        if s not in ("countdown", "recording"):
            self.progress["value"] = 0
            self.lbl_progress.config(text="")
        if s == "disconnected":
            for lbl in self.board_status.values():
                if lbl.cget("text") != "failed":
                    lbl.config(text="not connected", foreground="gray")

    # ---- events from the controller ------------------------------------------------
    def _poll(self):
        try:
            while True:
                try:
                    ev = self.ctl.events.get_nowait()
                except queue.Empty:
                    break
                try:
                    self._handle(ev)
                except Exception:  # noqa: BLE001
                    self._append_log("GUI error:\n" + traceback.format_exc())
        finally:
            self.root.after(100, self._poll)

    def _handle(self, ev):
        kind = ev[0]
        if kind == "log":
            self._append_log(ev[1])
        elif kind == "state":
            self._apply_state(ev[1])
            if self._quit_when_idle and ev[1] in ("ready", "disconnected"):
                self._finish_quit()
        elif kind == "board":
            _, label, status = ev
            colors = {"connected": "green", "connecting": "orange",
                      "failed": "red", "disconnected": "gray"}
            lbl = self.board_status.get(label)
            if lbl is not None:
                lbl.config(text=status if status != "disconnected" else "not connected",
                           foreground=colors.get(status, "gray"))
        elif kind == "progress":
            _, phase, a, b = ev
            if phase == "countdown":
                self.progress["value"] = 0
                self.lbl_progress.config(
                    text=f"Starting in {int(a)} s - set the boards down and keep them still")
            else:
                self.progress["value"] = 100.0 * min(a / b, 1.0) if b else 0
                self.lbl_progress.config(text=f"Recording  {a:4.1f} / {b:g} s")
        elif kind == "done":
            self._last_stem = ev[1]
            self._refresh_previews(ev[1])
            if self.var_auto3d.get() and self.var_trackz.get():
                self._open_3d(ev[1])

    def _append_log(self, text):
        self.log.config(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.config(state="disabled")

    # ---- button handlers -------------------------------------------------------------
    def _on_connect(self):
        labels = [l for l, v in self.board_vars.items() if v.get()]
        if not labels:
            messagebox.showerror("No boards selected", "Tick at least one board first.")
            return
        for l in labels:
            self.board_status[l].config(text="waiting", foreground="gray")
        self.ctl.connect(labels)

    def _on_disconnect(self):
        self.ctl.disconnect()

    def _on_record(self):
        raw = self.var_name.get()
        name = clean_name(raw)
        if not name:
            messagebox.showerror("Session name", "Enter a session name (letters, digits, - or _).")
            return
        if name != raw.strip():
            print(f"Session name cleaned to '{name}' (only letters, digits, - and _ are used).")
            self.var_name.set(name)
        try:
            duration = float(self.var_duration.get())
            delay = float(self.var_delay.get())
            if duration <= 0 or delay < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Duration / delay",
                                 "Duration must be a number > 0 and start delay a number >= 0.")
            return
        path = os.path.join(rec.SESSION_DIR, f"{name}.csv")
        if os.path.exists(path) and not messagebox.askyesno(
                "Overwrite?", f"{name}.csv already exists in Session Data.\nOverwrite it?"):
            return
        self.ctl.start_recording(name, duration, delay, self.var_process.get(),
                                 self.var_trackz.get())
        self._pending_next = next_name(name)

    def _on_stop(self):
        self.ctl.stop_early()

    def _on_reprocess(self):
        name = clean_name(self.var_name.get())
        if not name:
            messagebox.showerror("Session name", "Enter the name of a saved session.")
            return
        self.ctl.process_existing(name, self.var_trackz.get())

    def _open_folder(self, folder):
        try:
            os.makedirs(folder, exist_ok=True)
            if sys.platform.startswith("win"):
                os.startfile(folder)                      # noqa: S606
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder])
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Could not open folder", f"{folder}\n\n{e}")

    # ---- preview -----------------------------------------------------------------------
    def _pick_stem(self):
        """Session name box if it has results, otherwise the last session run."""
        name = clean_name(self.var_name.get())
        if name and os.path.exists(os.path.join(pos.POSITION_DIR, f"{name}_position.csv")):
            return name
        return self._last_stem or name

    def _on_show_plots(self):
        stem = self._pick_stem()
        if not stem:
            messagebox.showerror("Session name", "Enter the name of a processed session.")
            return
        self._refresh_previews(stem)

    def _on_open_3d(self):
        stem = self._pick_stem()
        if not stem:
            messagebox.showerror("Session name", "Enter the name of a processed session.")
            return
        self._open_3d(stem)

    def _open_3d(self, stem):
        """Open the interactive 3-D viewer in its own process (this GUI uses Agg)."""
        csv_path = os.path.join(pos.POSITION_DIR, f"{stem}_position.csv")
        script = os.path.join(HERE, "view_3d.py")
        if not os.path.exists(csv_path):
            messagebox.showerror("3D plot", f"No position file yet:\n{csv_path}\n\n"
                                            "Process the session first.")
            return
        if not os.path.exists(script):
            messagebox.showerror("3D plot", f"view_3d.py not found next to the GUI:\n{script}")
            return
        try:
            subprocess.Popen([sys.executable, script, csv_path], cwd=HERE)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("3D plot", f"Could not open the 3D viewer:\n{e}")

    def _refresh_previews(self, stem):
        found = {}
        for tag, folder in (("Positioning", pos.POSITION_DIR),
                            ("Orientation", ori.ORIENTATION_DIR)):
            for f in sorted(glob.glob(os.path.join(folder, f"{stem}_*.png"))):
                found[f"{tag}: {os.path.basename(f)}"] = f

        def rank(item):
            n = item[0]
            return (0 if n.endswith("_all_xy.png") else
                    1 if n.endswith("_xy.png") else
                    2 if n.endswith("_3d.png") else
                    3 if n.endswith("_z.png") else
                    4 if "position_time" in n else 5, n)

        self._preview_files = dict(sorted(found.items(), key=rank))
        names = list(self._preview_files)
        self.cmb_preview.config(values=names)
        if not names:
            self.lbl_image.config(image="", text=f"No plots found for '{stem}' in:\n"
                                  f"{pos.POSITION_DIR}\n{ori.ORIENTATION_DIR}\n\n"
                                  "Check the log for 'processing failed' errors.")
        if names:
            self.var_preview.set(names[0])
            self._show_selected_preview()
        # suggest the next session name once a recording has finished
        nxt = getattr(self, "_pending_next", None)
        if nxt:
            self.var_name.set(nxt)
            self._pending_next = None

    def _show_selected_preview(self):
        path = self._preview_files.get(self.var_preview.get())
        if not path:
            return
        try:
            if Image is not None:
                im = Image.open(path)
                im.thumbnail((self.PREVIEW_W, self.PREVIEW_H))
                img = ImageTk.PhotoImage(im)
                self.lbl_image.config(image=img, text="")
                self.lbl_image.image = img
                return
            img = tk.PhotoImage(file=path)
            f = max(1, math.ceil(img.width() / self.PREVIEW_W),
                    math.ceil(img.height() / self.PREVIEW_H))
            if f > 1:
                img = img.subsample(f, f)
            self.lbl_image.config(image=img, text="")
            self.lbl_image.image = img                    # keep a reference
        except Exception as e:  # noqa: BLE001
            self.lbl_image.config(image="", text=f"Could not show preview: {e}")

    # ---- closing -----------------------------------------------------------------------
    def _on_close(self):
        if self.ctl.state in ("connecting", "countdown", "recording",
                              "processing", "disconnecting"):
            if not messagebox.askyesno(
                    "Quit", "Something is still running.\n"
                            "Stop it (a recording is saved first) and quit?"):
                return
            self._quit_when_idle = True
            self.ctl.stop_early()
            return
        self._finish_quit()

    def _finish_quit(self):
        self._quit_when_idle = False
        self.ctl.disconnect(wait=True)
        self.root.destroy()


def main():
    if tk is None:
        sys.exit("tkinter is not available in this Python install. On Windows, "
                 "re-run the Python installer and tick 'tcl/tk and IDLE'.")
    ctl = SessionController()
    root = tk.Tk()
    sys.stdout = _Tee(sys.__stdout__, ctl.events)
    sys.stderr = _Tee(sys.__stderr__, ctl.events)
    App(root, ctl)
    root.mainloop()
    os._exit(0)     # the Bluetooth library can keep background threads alive


if __name__ == "__main__":
    main()