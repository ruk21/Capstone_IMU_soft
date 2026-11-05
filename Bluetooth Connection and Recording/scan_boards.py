#!/usr/bin/env python3
"""
List every Bluetooth LE device this PC can see, so you can confirm your
MetaWear boards are visible and copy their MAC addresses.

Run:  python scan_boards.py
Boards named "MetaWear" are marked with <<<. Signal strength (RSSI) is shown if
available: closer to 0 is stronger (about -50 is very close, -90 is barely there).
"""
import time

from mbientlab.warble import BleScanner

SCAN_SECONDS = 20

seen = {}   # mac -> (name, best rssi)


def handler(result):
    mac = result.mac.upper()
    name = result.name or ""
    rssi = getattr(result, "rssi", None)
    old = seen.get(mac)
    if old is None:
        tag = "  <<< MetaWear" if "metawear" in name.lower() else ""
        print(f"  found {mac}  {name or '(no name)'}  rssi={rssi}{tag}")
    best = rssi if old is None or old[1] is None or (rssi is not None and rssi > old[1]) else old[1]
    seen[mac] = (name or (old[0] if old else ""), best)


def main():
    print(f"Scanning for {SCAN_SECONDS} s ... (press each board's button now)\n")
    BleScanner.set_handler(handler)
    BleScanner.start()
    time.sleep(SCAN_SECONDS)
    BleScanner.set_handler(None)
    BleScanner.stop()

    boards = {m: v for m, v in seen.items() if "metawear" in v[0].lower()}
    print(f"\n{len(seen)} BLE device(s) seen, {len(boards)} MetaWear board(s):")
    for m, (name, rssi) in sorted(boards.items()):
        print(f"  {m}   rssi={rssi}")
    if not seen:
        print("\nNothing at all was seen: the PC's Bluetooth is likely off or stuck.\n"
              "Turn Bluetooth off and on in Windows settings (or unplug/replug a USB "
              "Bluetooth dongle) and run again.")
    elif not boards:
        print("\nOther devices are visible but no MetaWear boards: wake the boards "
              "(press the button), move them closer, and make sure no phone app is "
              "connected to them.")


if __name__ == "__main__":
    main()
