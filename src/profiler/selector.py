#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import logging
import numpy as np
from typing import List, Dict, Tuple
from scipy.stats import pearsonr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.utils import parse_perf_output, extract_data_log, EVENT_GROUP_FILES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PearsonFeatureSelector")


class PearsonFeatureSelector:
    """Pearson correlation-based feature selector with stability analysis"""
    
    DATA_TRUNCATE_LENGTH = 180
    STABILITY_TOP_K = 30
    STABILITY_INTERVAL = (0, 30, 60, 90)

    def __init__(self, folder_path: str):
        self._validate_folder_path(folder_path)
        self.folder_path = folder_path

        self.event_counts: dict = {}
        self.lats: list = []
        self.sorted_pearson: list = []
        self.sorted_last: list = []

        logger.info(f"PearsonFeatureSelector initialized | Data dir: {folder_path}")

    @staticmethod
    def _validate_folder_path(folder_path: str) -> None:
        if not isinstance(folder_path, str) or len(folder_path.strip()) == 0:
            raise ValueError("Data folder path cannot be empty")
        if not os.path.isdir(folder_path):
            raise NotADirectoryError(f"Invalid data directory: {folder_path}")

    def _load_and_clean_data(self) -> None:
        """Load latency and perf data, clean and truncate"""
        self.event_counts = {}
        self.lats = []

        for event_filename in EVENT_GROUP_FILES:
            parsed_results = []
            current_lats = []

            for filename in os.listdir(self.folder_path):
                file_path = os.path.join(self.folder_path, filename)
                lat_filename = f"{event_filename.split('.')[0]}_lat.log"

                if filename == lat_filename:
                    current_lats = extract_data_log(file_path)
                    if len(current_lats) == self.DATA_TRUNCATE_LENGTH:
                        current_lats = current_lats[:self.DATA_TRUNCATE_LENGTH]
                    elif len(current_lats) > self.DATA_TRUNCATE_LENGTH:
                        current_lats = current_lats[1:self.DATA_TRUNCATE_LENGTH+1]
                    else:
                        raise ValueError(f"Invalid latency data length: {len(current_lats)} (expected 180)")

                if filename == event_filename:
                    file_event_counts = parse_perf_output(file_path)
                    for event_name in file_event_counts.keys():
                        file_event_counts[event_name] = file_event_counts[event_name][:self.DATA_TRUNCATE_LENGTH]
                    parsed_results.append(file_event_counts)

            self._check_duplicate_events(parsed_results)
            for data in parsed_results:
                self.event_counts.update(data)

            self.lats = current_lats
            pearson_res = self._compute_pearson_correlation(file_event_counts, current_lats)
            self.sorted_pearson.extend(pearson_res)

        self.sorted_pearson = sorted(self.sorted_pearson, key=lambda x: x[1], reverse=True)
        logger.info(f"Correlation computed | Valid events: {len(self.sorted_pearson)}")

    def _check_duplicate_events(self, parsed_results: list) -> None:
        """Detect duplicate performance events"""
        events_list = []
        repeat_events = []
        for data in parsed_results:
            for key in data.keys():
                if key in events_list:
                    repeat_events.append(key)
                else:
                    events_list.append(key)
        if repeat_events:
            logger.warning(f"Found duplicate events: {repeat_events}")

    def _compute_pearson_correlation(self, event_counts: dict, lats: list) -> List[Tuple[str, float]]:
        """Compute Pearson correlation between events and latency"""
        correlation_results = {}
        for event_name, counts in event_counts.items():
            if len(counts) != len(lats):
                logger.warning(f"Length mismatch: {event_name} ({len(counts)} vs {len(lats)})")
                continue
            if np.std(counts) == 0 or np.std(lats) == 0:
                logger.warning(f"Constant data, skipping event: {event_name}")
                continue
            corr, _ = pearsonr(counts, lats)
            correlation_results[event_name] = abs(corr)

        correlation_results = {
            event: corr for event, corr in correlation_results.items()
            if corr is not None and not np.isnan(corr)
        }
        return sorted(correlation_results.items(), key=lambda x: x[1], reverse=True)

    def _compute_stability(self) -> None:
        """Compute event stability (standard deviation) for top-k correlated events"""
        if not self.sorted_pearson:
            raise RuntimeError("Please compute correlation first")

        std_list = []
        start1, end1, start2, end2 = self.STABILITY_INTERVAL
        for event, _ in self.sorted_pearson[:self.STABILITY_TOP_K]:
            std_val = np.std(
                self.event_counts[event][start1:end1] + self.event_counts[event][start2:end2]
            )
            std_list.append((event, std_val))

        self.sorted_last = sorted(std_list, key=lambda x: x[1], reverse=False)
        logger.info(f"Stability computed | Final events: {len(self.sorted_last)}")

    def select(self) -> None:
        """Run complete selection pipeline"""
        self._load_and_clean_data()
        self._compute_stability()
        logger.info("Performance counter selection completed")

    def get_sorted_last(self) -> List[Tuple[str, float]]:
        """Get final selection results sorted by stability"""
        if not self.sorted_last:
            raise RuntimeError("Please call select() first")
        return self.sorted_last

    def get_top_k_stable_events(self, k: int = 10) -> List[Tuple[str, float]]:
        """Get top-k most stable events"""
        sorted_res = self.get_sorted_last()
        return sorted_res[:k]
