#!/usr/bin/env python3
"""
Record BMI160 accelerometer + gyroscope data from two MetaWear RG/RPro boards
simultaneously and write everything to a single CSV.

Install:   pip install metawear
Run:       python record_two_metawear.py

Edit DEVICES / DURATION_S below. Command-line overrides are optional:
           python record_two_metawear.py --duration 60 --out other.csv

CSV columns:
  host_time_s     - streaming mode: time.perf_counter() when the packet reached this
                    computer, relative to recording start.
                    LOG_ON_BOARD mode: seconds since the earliest sample, taken
                    from the boards' own timestamps (device_epoch_ms).
  device_epoch_ms - timestamp reported by the SDK for that sample
  device          - label from DEVICES
  sensor          - 'acc' (g) or 'gyro' (deg/s)
  x, y, z
"""
# ======================================================================
# EDIT THESE EACH SESSION
# ======================================================================
SESSION_NAME = "session1"   # saves  Session Data\<SESSION_NAME>.csv
DURATION_S = 30.0           # recording length in seconds
START_DELAY_S = 5           # countdown (s) after connecting, so you can set the boards
                            # down and let go before recording begins
# ======================================================================

import argparse
import csv
import os
import threading
import time
from time import sleep

from ctypes import byref, cast

from mbientlab.warble import BleScanner
from mbientlab.metawear import MetaWear, libmetawear, parse_value
from mbientlab.metawear.cbindings import (
    FnVoid_VoidP_DataP,
    FnVoid_VoidP_UByte_Long_UByteP_UByte,
    FnVoid_VoidP_UInt_UInt,
    FnVoid_VoidP_VoidP,
    GyroBoschOdr,
    GyroBoschRange,
    LogDownloadHandler,
    Module,
)

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
DEVICES = {
    "left":  "F3:FA:18:F7:61:32",   # board 1
    "right": "D4:12:C5:89:FE:ED",   # board 2
}

OUT_FILE = f"{SESSION_NAME}.csv"     # output CSV filename (from SESSION_NAME)
SESSION_DIR = r"C:\Users\Frank\OneDrive - University of Waterloo\01 University of Waterloo\01 University of Waterloo [BME]\01 Terms\4A\BME 461 [Capstone]\Code\Bluetooth Connection and Recording\Session Data"

SCAN_TIMEOUT_S = 30.0               # how long to look for the boards first
ABORT_IF_NOT_SEEN = False           # False = still try to connect even if the scan
                                    #         didn't see a board (often works anyway)
RESCAN_S = 8.0                      # short re-scan between failed connect attempts
CONNECT_RETRIES = 5

ACC_ODR_HZ = 200.0                  # BMI160: 0.78 ... 1600 Hz
ACC_RANGE_G = 16.0                  # 2, 4, 8, or 16
GYRO_ODR = GyroBoschOdr._200Hz      # _25Hz ... _3200Hz
GYRO_RANGE = GyroBoschRange._2000dps # _125dps ... _2000dps

# True : each board records into its own flash memory and the data is downloaded
#        over Bluetooth when recording stops (no dropped packets, exact on-board
#        timestamps). The download takes roughly as long as the recording or less.
# False: live BLE streaming (packets can be dropped at high rates).
LOG_ON_BOARD = True
DOWNLOAD_STALL_S = 20.0             # give up if no log data arrives for this long
RESET_ON_CONNECT = True             # wipe old log data / loggers when connecting
                                    # (adds ~10-20 s per board; set False to skip)
# ----------------------------------------------------------------------------


def scan_for_boards(macs, timeout):
    """
    Windows can only open a BLE device by address after it has recently seen
    it advertising. Scanning first makes that happen. Returns the set of
    addresses that were NOT seen.
    """
    wanted = {m.upper() for m in macs}
    seen = set()

    def handler(result):
        mac = result.mac.upper()
        if mac in wanted:
            seen.add(mac)

    print(f"Scanning up to {timeout:.0f} s for {len(wanted)} board(s) ...")
    BleScanner.set_handler(handler)
    BleScanner.start()
    t_end = time.time() + timeout
    while time.time() < t_end and seen != wanted:
        sleep(0.25)
    BleScanner.set_handler(None)
    BleScanner.stop()

    for m in sorted(seen):
        print(f"  found {m}")
    return wanted - seen


class Recorder:
    """Shared CSV writer; callbacks from both boards are serialized with a lock."""

    def __init__(self, path):
        self.file = open(path, "w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(
            ["host_time_s", "device_epoch_ms", "device", "sensor", "x", "y", "z"]
        )
        self.lock = threading.Lock()
        self.t0 = None
        self.counts = {}
        self.rows = []                    # on-board logging: buffered until close()

    def write(self, label, sensor, epoch_ms, x, y, z):
        if LOG_ON_BOARD:
            # Downloaded entries arrive long after the movement, so host arrival
            # time means nothing. Buffer them; close() writes device-clock times.
            with self.lock:
                if self.file.closed:
                    return
                self.rows.append((label, sensor, epoch_ms, x, y, z))
                key = (label, sensor)
                self.counts[key] = self.counts.get(key, 0) + 1
            return
        if self.t0 is None:
            return
        t = time.perf_counter() - self.t0
        with self.lock:
            if self.file.closed:          # late packet after recording stopped
                return
            self.writer.writerow([f"{t:.6f}", epoch_ms, label, sensor, x, y, z])
            key = (label, sensor)
            self.counts[key] = self.counts.get(key, 0) + 1

    def close(self):
        with self.lock:
            if self.rows and not self.file.closed:
                # host_time_s = seconds since the earliest sample, from the boards'
                # own timestamps (same column the processing scripts already read)
                epoch0 = min(r[2] for r in self.rows)
                self.rows.sort(key=lambda r: (r[0], r[1], r[2]))
                for label, sensor, ep, x, y, z in self.rows:
                    self.writer.writerow(
                        [f"{(ep - epoch0) / 1000.0:.6f}", ep, label, sensor, x, y, z])
                self.rows = []
            self.file.flush()
            self.file.close()


class Board:
    def __init__(self, label, mac, recorder=None):
        self.label = label
        self.mac = mac
        self.rec = recorder
        self.device = MetaWear(mac)
        self.board = self.device.board
        self.connected = False
        self.configured = False
        self.started = False
        # Keep references to the C callbacks so they aren't garbage collected
        self._acc_cb = FnVoid_VoidP_DataP(self._on_acc)
        self._gyro_cb = FnVoid_VoidP_DataP(self._on_gyro)
        self.model_name = "unknown"
        self.gyro_kind = None
        self._acc_logger = None
        self._gyro_logger = None
        self._last_rx = time.time()
        self.n_acc = 0
        self.n_gyro = 0
        self._keep = []     # callback objects that must outlive the C calls

    # --- callbacks -----------------------------------------------------------
    def _on_acc(self, ctx, data):
        self._last_rx = time.time()
        self.n_acc += 1
        if self.rec is None:
            return
        v = parse_value(data)
        self.rec.write(self.label, "acc", data.contents.epoch, v.x, v.y, v.z)

    def _on_gyro(self, ctx, data):
        self._last_rx = time.time()
        self.n_gyro += 1
        if self.rec is None:
            return
        v = parse_value(data)
        self.rec.write(self.label, "gyro", data.contents.epoch, v.x, v.y, v.z)

    # --- gyro chip differs between boards ------------------------------------
    def _detect_gyro(self):
        """RG/RPro use a BMI160 gyro; MetaMotion S uses a BMI270."""
        if self.gyro_kind is not None:
            return
        impl = libmetawear.mbl_mw_metawearboard_lookup_module(self.board, Module.GYRO)
        if impl == 1:
            self.gyro_kind = "bmi270"
        elif impl == 0:
            self.gyro_kind = "bmi160"
        else:   # lookup failed; fall back on the model name
            self.gyro_kind = "bmi270" if "MetaMotion S" in self.model_name else "bmi160"
        print(f"[{self.label}] gyro chip: {self.gyro_kind.upper()}")

    def _gyro(self, name):
        return getattr(libmetawear, f"mbl_mw_gyro_{self.gyro_kind}_{name}")

    # --- on-board logging ----------------------------------------------------
    def _create_logger(self, signal):
        """Ask the board to log `signal` to its flash; returns the logger handle."""
        done, out = threading.Event(), {}

        def handler(ctx, pointer):
            out["p"] = pointer
            done.set()

        cb = FnVoid_VoidP_VoidP(handler)
        self._keep.append(cb)
        libmetawear.mbl_mw_datasignal_log(signal, None, cb)
        if not done.wait(10) or not out.get("p"):
            raise RuntimeError(f"[{self.label}] board did not create a data logger")
        return out["p"]

    def _download(self):
        """Pull the logged entries off the board; they flow into _on_acc/_on_gyro."""
        b = self.board
        done = threading.Event()
        self._last_rx = time.time()

        def progress(ctx, left, total):
            self._last_rx = time.time()
            if total:
                pct = 100 * (total - left) / total
                print(f"[{self.label}] downloading {pct:.0f}% ({total} entries)", end="\r")
            if left == 0:
                done.set()

        unknown = [0]

        def on_unknown(ctx, logger_id, epoch, data, length):
            unknown[0] += 1     # entry the SDK could not match to a logger

        progress_cb = FnVoid_VoidP_UInt_UInt(progress)
        unknown_cb = FnVoid_VoidP_UByte_Long_UByteP_UByte(on_unknown)
        self._keep += [progress_cb, unknown_cb]
        handler = LogDownloadHandler(
            context=None,
            received_progress_update=progress_cb,
            received_unknown_entry=unknown_cb,
            received_unhandled_entry=cast(None, FnVoid_VoidP_DataP),
        )
        self.n_acc = self.n_gyro = 0
        t_dl = time.time()
        libmetawear.mbl_mw_logger_subscribe(self._acc_logger, None, self._acc_cb)
        libmetawear.mbl_mw_logger_subscribe(self._gyro_logger, None, self._gyro_cb)
        print(f"[{self.label}] downloading recording from the board ...")
        libmetawear.mbl_mw_logging_download(b, 100, byref(handler))
        while not done.wait(1.0):
            if time.time() - self._last_rx > DOWNLOAD_STALL_S:
                raise RuntimeError(f"[{self.label}] log download stalled")
        sleep(0.5)      # let the last entries be delivered
        n_a, n_g = self.n_acc, self.n_gyro
        print(f"[{self.label}] download complete in {time.time() - t_dl:.0f} s: "
              f"{n_a} acc, {n_g} gyro samples, {unknown[0]} unrecognised entries" + " " * 10)
        if n_a == 0 or n_g == 0 or abs(n_a - n_g) > 0.02 * max(n_a, n_g):
            print(f"[{self.label}] WARNING: acc and gyro sample counts differ or one is "
                  f"missing - this recording is incomplete. Do not use it for processing.")

    def _remove_loggers(self):
        for name in ("_acc_logger", "_gyro_logger"):
            lg = getattr(self, name)
            if lg:
                try:
                    libmetawear.mbl_mw_logger_remove(lg)
                except Exception as e:  # noqa: BLE001
                    print(f"[{self.label}] logger cleanup warning: {e}")
                setattr(self, name, None)
        try:
            libmetawear.mbl_mw_logging_clear_entries(self.board)
        except Exception as e:  # noqa: BLE001
            print(f"[{self.label}] log clear warning: {e}")

    # --- lifecycle -----------------------------------------------------------
    def connect(self):
        self._open()
        try:
            name = libmetawear.mbl_mw_metawearboard_get_model_name(self.board)
            self.model_name = name.decode() if name else "unknown"
            print(f"[{self.label}] board model: {self.model_name}")
        except Exception as e:  # noqa: BLE001
            print(f"[{self.label}] could not read model name: {e}")
        self._detect_gyro()
        if LOG_ON_BOARD and RESET_ON_CONNECT:
            self._reset_board()
            self._open()
        print(f"[{self.label}] connected")

    def _reset_board(self):
        """Wipe the board's log and leftover loggers/routes, then reboot it."""
        print(f"[{self.label}] clearing old log data and resetting the board ...")
        b = self.board
        try:
            libmetawear.mbl_mw_logging_stop(b)
            libmetawear.mbl_mw_logging_clear_entries(b)
            sleep(1.5)                                  # let the flash erase finish
            libmetawear.mbl_mw_debug_reset_after_gc(b)  # drop old loggers, then reboot
            sleep(1.5)
        except Exception as e:  # noqa: BLE001
            print(f"[{self.label}] reset warning: {e}")
        try:
            self.device.disconnect()
        except Exception:  # noqa: BLE001
            pass
        self.connected = False
        # Forget the SDK's saved copy of this board's state; it no longer matches
        cache = os.path.join(".metawear", self.mac.replace(":", "").upper() + ".json")
        try:
            os.remove(cache)
        except OSError:
            pass
        sleep(2.0)
        scan_for_boards([self.mac], RESCAN_S)           # board is advertising again

    def _open(self):
        last_err = None
        for attempt in range(1, CONNECT_RETRIES + 1):
            try:
                print(f"[{self.label}] connecting to {self.mac} "
                      f"(attempt {attempt}/{CONNECT_RETRIES}) ...")
                self.device.connect()
                self.connected = True
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"[{self.label}] connect failed: {e}")
                scan_for_boards([self.mac], RESCAN_S)   # let Windows see it again
                sleep(1.0)
        if not self.connected:
            raise RuntimeError(
                f"Could not connect to {self.label} ({self.mac}): {last_err}"
            )

        # Short connection interval = more BLE throughput (important for 2 boards)
        libmetawear.mbl_mw_settings_set_connection_parameters(
            self.board, 7.5, 7.5, 0, 6000
        )
        sleep(1.5)

    def configure(self):
        b = self.board
        # Accelerometer
        libmetawear.mbl_mw_acc_set_odr(b, ACC_ODR_HZ)
        libmetawear.mbl_mw_acc_set_range(b, ACC_RANGE_G)
        libmetawear.mbl_mw_acc_write_acceleration_config(b)
        acc_sig = libmetawear.mbl_mw_acc_get_acceleration_data_signal(b)

        # Gyroscope (BMI160 on RG/RPro, BMI270 on MetaMotion S)
        self._detect_gyro()
        self._gyro("set_odr")(b, GYRO_ODR)
        self._gyro("set_range")(b, GYRO_RANGE)
        self._gyro("write_config")(b)
        gyro_sig = self._gyro("get_rotation_data_signal")(b)

        if LOG_ON_BOARD:
            self._acc_logger = self._create_logger(acc_sig)
            self._gyro_logger = self._create_logger(gyro_sig)
            # Erase old entries now (during setup) so it can't delay the start
            libmetawear.mbl_mw_logging_clear_entries(b)
        else:
            libmetawear.mbl_mw_datasignal_subscribe(acc_sig, None, self._acc_cb)
            libmetawear.mbl_mw_datasignal_subscribe(gyro_sig, None, self._gyro_cb)
        self.configured = True

    def start(self):
        b = self.board
        if LOG_ON_BOARD:
            libmetawear.mbl_mw_logging_start(b, 0)      # 0 = don't overwrite when full
        libmetawear.mbl_mw_acc_enable_acceleration_sampling(b)
        self._gyro("enable_rotation_sampling")(b)
        libmetawear.mbl_mw_acc_start(b)
        self._gyro("start")(b)
        self.started = True

    def stop(self):
        b = self.board
        try:
            if self.started:
                libmetawear.mbl_mw_acc_stop(b)
                self._gyro("stop")(b)
                libmetawear.mbl_mw_acc_disable_acceleration_sampling(b)
                self._gyro("disable_rotation_sampling")(b)
                self.started = False
                if LOG_ON_BOARD:
                    libmetawear.mbl_mw_logging_stop(b)
                    if "MetaMotion S" in self.model_name:
                        # MetaMotion S buffers log data in RAM pages; write the
                        # partial last page to flash or it is not downloaded
                        libmetawear.mbl_mw_logging_flush_page(b)
                        sleep(1.0)
                    self._download()
        finally:
            self.started = False
            if self.configured:
                if LOG_ON_BOARD:
                    self._remove_loggers()
                else:
                    acc_sig = libmetawear.mbl_mw_acc_get_acceleration_data_signal(b)
                    gyro_sig = self._gyro("get_rotation_data_signal")(b)
                    libmetawear.mbl_mw_datasignal_unsubscribe(acc_sig)
                    libmetawear.mbl_mw_datasignal_unsubscribe(gyro_sig)
                self.configured = False

    def disconnect(self):
        if not self.connected:
            return
        try:
            self.device.disconnect()
        except Exception as e:  # noqa: BLE001
            print(f"[{self.label}] disconnect warning: {e}")
        self.connected = False
        print(f"[{self.label}] disconnected")


def main():
    # The MetaWear SDK creates a ".metawear" cache folder in the current working
    # directory. Run from the script's own folder so it never lands somewhere
    # protected (e.g. C:\Windows\System32).
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Optional command-line overrides; with no arguments the values from the
    # config block at the top of the file are used.
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=DURATION_S, help="seconds to record")
    ap.add_argument("--out", default=OUT_FILE)
    args = ap.parse_args()

    # Save into the Session Data folder (created if missing) unless an absolute
    # path was given with --out
    if not os.path.isabs(args.out):
        os.makedirs(SESSION_DIR, exist_ok=True)
        args.out = os.path.join(SESSION_DIR, args.out)

    # Make sure Windows has seen both boards advertising before we connect
    missing = scan_for_boards(DEVICES.values(), SCAN_TIMEOUT_S)
    if missing:
        print("\nScan did not see these boards advertising:")
        for m in sorted(missing):
            print(f"  {m}")
        print("Check: board awake (press its button; LED should blink), nothing "
              "else connected to it, within ~1 m, Windows Bluetooth on, MAC correct.")
        if ABORT_IF_NOT_SEEN:
            return
        print("Trying to connect anyway ...\n")

    rec = Recorder(args.out)
    boards = [Board(label, mac, rec) for label, mac in DEVICES.items()]

    try:
        # Connect + configure each board (sequential is more reliable for BLE)
        for b in boards:
            b.connect()
        for b in boards:
            b.configure()

        if START_DELAY_S > 0:
            print("\nBoards are ready. Place them where they will start and let go - "
                  "keep them STILL for the first second or two of recording.")
            for k in range(int(START_DELAY_S), 0, -1):
                print(f"  recording starts in {k} ...")
                sleep(1)

        # Start recording clock, then start both boards back-to-back
        rec.t0 = time.perf_counter()
        for b in boards:
            b.start()
        print(f"Recording for {args.duration:.0f} s ... (Ctrl+C to stop early)")

        try:
            sleep(args.duration)
        except KeyboardInterrupt:
            print("\nStopping early.")

    except Exception as e:  # noqa: BLE001
        print(f"\nERROR: {e}")

    finally:
        for b in boards:
            try:
                b.stop()
            except Exception as e:  # noqa: BLE001
                print(f"[{b.label}] stop warning: {e}")
        sleep(0.5)
        for b in boards:
            b.disconnect()
        rec.close()

    print(f"\nSaved {args.out}")
    for (label, sensor), n in sorted(rec.counts.items()):
        print(f"  {label:>8s} {sensor:<5s} {n} samples")


if __name__ == "__main__":
    main()
