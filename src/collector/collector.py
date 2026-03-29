#!/usr/bin/env python3
# coding: utf-8
import os
import logging
import subprocess
from typing import List, Optional
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PerfCollector")


class PerfCollector:
    """Linux perf performance data collector"""

    def __init__(
            self,
            sampling_interval_ms: int,
            monitor_events: List[str],
            pid: int,
            output_dir: str,
            duration_sec: int,
            output_filename: Optional[str] = None
    ):
        """
        Args:
            sampling_interval_ms: Sampling interval in milliseconds
            monitor_events: List of performance events to monitor
            pid: Target process ID
            output_dir: Output directory for performance data
            duration_sec: Sampling duration in seconds
            output_filename: Output filename
        """
        self._validate_params(
            sampling_interval_ms, monitor_events, pid, output_dir, duration_sec
        )

        self.sampling_interval_ms = sampling_interval_ms
        self.monitor_events = monitor_events
        self.pid = pid
        self.output_dir = output_dir
        self.duration_sec = duration_sec

        self.output_filename = output_filename or self._generate_default_filename()
        self.output_file_path = os.path.join(output_dir, self.output_filename)

        self._create_output_directory()

        logger.info(f"PerfCollector initialized | PID: {pid} | Output: {self.output_file_path}")

    @staticmethod
    def _validate_params(
            sampling_interval_ms: int,
            monitor_events: List[str],
            pid: int,
            output_dir: str,
            duration_sec: int
    ) -> None:
        if not isinstance(sampling_interval_ms, int) or sampling_interval_ms <= 0:
            raise ValueError("sampling_interval_ms must be positive integer")

        if not isinstance(monitor_events, list) or len(monitor_events) == 0:
            raise ValueError("monitor_events cannot be empty")

        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("pid must be positive integer")

        if not isinstance(duration_sec, int) or duration_sec <= 0:
            raise ValueError("duration_sec must be positive integer")

        if not isinstance(output_dir, str) or len(output_dir.strip()) == 0:
            raise ValueError("output_dir cannot be empty")

    def _generate_default_filename(self) -> str:
        """Generate default output filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"perf_pid{self.pid}_{timestamp}.log"

    def _create_output_directory(self) -> None:
        """Create output directory if not exists"""
        os.makedirs(self.output_dir, exist_ok=True)
        logger.debug(f"Output directory ready: {self.output_dir}")

    def _build_perf_command(self) -> str:
        """Build complete perf command string"""
        events_str = ",".join(self.monitor_events)

        command = (
            f"sudo perf stat -I {self.sampling_interval_ms} "
            f"-e {events_str} "
            f"-p {self.pid} "
            f"-o {self.output_file_path} "
            f"sleep {self.duration_sec}"
        )

        return command

    def start_collect(self) -> subprocess.Popen:
        """Start performance data collection (non-blocking)"""
        perf_cmd = self._build_perf_command()
        logger.info(f"Starting perf collection: {perf_cmd}")

        collect_process = subprocess.Popen(
            perf_cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        logger.info(f"Collection process started, PID: {collect_process.pid}")
        return collect_process

    def start_collect_block(self) -> None:
        """Start performance data collection (blocking)"""
        process = self.start_collect()
        stdout, stderr = process.communicate()

        if process.returncode == 0:
            logger.info(f"Collection completed! Data saved to: {self.output_file_path}")
        else:
            if "sudo:" in stderr or "permission denied" in stderr.lower():
                error_msg = "Perf collection requires sudo privileges"
                logger.error(f"Collection failed! Error: {stderr}")
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            else:
                logger.error(f"Collection failed! Error: {stderr}")
                raise RuntimeError(f"Perf collection failed, return code: {process.returncode}")
