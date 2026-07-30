#!/usr/bin/env python3
"""Simple network traffic monitor for Linux.

Reads /proc/net/dev and prints bytes/packets per second for each interface.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, Tuple


def read_net_dev() -> Dict[str, Tuple[int, int, int, int]]:
    values: Dict[str, Tuple[int, int, int, int]] = {}
    with open('/proc/net/dev', 'r', encoding='utf-8') as f:
        lines = f.readlines()[2:]

    for line in lines:
        if ':' not in line:
            continue
        iface, data = line.split(':', 1)
        iface = iface.strip()
        fields = data.split()
        if len(fields) < 16:
            continue
        rx_bytes = int(fields[0])
        rx_packets = int(fields[1])
        tx_bytes = int(fields[8])
        tx_packets = int(fields[9])
        values[iface] = (rx_bytes, rx_packets, tx_bytes, tx_packets)
    return values


def format_rate(value: float) -> str:
    if value < 1024:
        return f"{value:.1f} B/s"
    for unit in ['KiB/s', 'MiB/s', 'GiB/s', 'TiB/s']:
        value /= 1024.0
        if value < 1024:
            return f"{value:.2f} {unit}"
    return f"{value:.2f} PiB/s"


def format_bar(value: float, maximum: float, width: int = 24) -> str:
    """Return a fixed-width activity bar scaled to the highest observed rate."""
    fraction = min(1.0, value / maximum) if maximum > 0 else 0.0
    filled = round(fraction * width)
    return "█" * filled + "░" * (width - filled)


def monitor(interval: float, interfaces: tuple[str, ...]) -> None:
    previous = read_net_dev()
    previous_time = time.monotonic()
    peak_rate = 1.0
    time.sleep(interval)

    while True:
        current = read_net_dev()
        current_time = time.monotonic()
        delta_seconds = current_time - previous_time
        rows = []

        for iface, stats in sorted(current.items()):
            if interfaces and iface not in interfaces:
                continue
            prev_stats = previous.get(iface)
            if prev_stats is None:
                continue

            rx_bytes, rx_pkts, tx_bytes, tx_pkts = stats
            prev_rx_bytes, prev_rx_pkts, prev_tx_bytes, prev_tx_pkts = prev_stats
            rx_rate = max(0, rx_bytes - prev_rx_bytes) / delta_seconds
            rx_pkts_rate = (rx_pkts - prev_rx_pkts) / delta_seconds
            tx_rate = max(0, tx_bytes - prev_tx_bytes) / delta_seconds
            tx_pkts_rate = (tx_pkts - prev_tx_pkts) / delta_seconds
            peak_rate = max(peak_rate, rx_rate, tx_rate)
            rows.append((iface, rx_rate, rx_pkts_rate, tx_rate, tx_pkts_rate))

        # Move to the top-left and clear the terminal instead of appending output.
        sys.stdout.write("\033[H\033[2J")
        print(f"Network traffic monitor  •  refresh {delta_seconds:.1f}s  •  Ctrl+C to stop")
        print(f"Bar scale: {format_rate(peak_rate)}")
        print()
        for iface, rx_rate, rx_pkts_rate, tx_rate, tx_pkts_rate in rows:
            print(f"{iface}")
            print(
                f"  RX [{format_bar(rx_rate, peak_rate)}] "
                f"{format_rate(rx_rate):>12}  {rx_pkts_rate:7.1f} pkt/s"
            )
            print(
                f"  TX [{format_bar(tx_rate, peak_rate)}] "
                f"{format_rate(tx_rate):>12}  {tx_pkts_rate:7.1f} pkt/s"
            )
        sys.stdout.flush()

        previous = current
        previous_time = current_time
        time.sleep(interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Linux network traffic monitor')
    parser.add_argument(
        '--interval', '-i', type=float, default=1.0,
        help='Seconds between samples (default: 1.0)'
    )
    parser.add_argument(
        '--interfaces', '-n', nargs='*', default=[],
        help='Optional list of interface names to monitor. If omitted, all interfaces are shown.'
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    interfaces = tuple(args.interfaces)
    try:
        monitor(args.interval, interfaces)
    except KeyboardInterrupt:
        print('\nMonitor stopped.')


if __name__ == '__main__':
    main()
