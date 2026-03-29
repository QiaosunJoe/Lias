#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import pickle
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.utils import parse_perf_output, EVENT_GROUP_FILES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RFTrainer")


class RFTrainer:
    """Random Forest model trainer for performance counter feature selection"""
    
    FIXED_LABEL = [0] * 30 + [1] * 30 + [0] * 30 + [2] * 30 + [0] * 30 + [3] * 30
    DATA_TRUNCATE_LENGTH = 180

    def __init__(
            self,
            file_paths: List[str],
            selected_events: Optional[List[str]] = None,
            save_model: bool = False,
            model_save_path: Optional[str] = None
    ):
        self._validate_params(file_paths, save_model, model_save_path)

        self.file_paths = file_paths
        self.selected_events = selected_events
        self.save_model = save_model
        self.model_save_path = model_save_path

        self.parsed_results: List[Dict] = []
        self.event_counts: Dict = {}
        self.sorted_feature_importances: List[Tuple[str, float]] = []
        self.model: Optional[RandomForestClassifier] = None

        logger.info("RFTrainer initialized")

    @staticmethod
    def _validate_params(
            file_paths: List[str],
            save_model: bool,
            model_save_path: Optional[str]
    ) -> None:
        if not isinstance(file_paths, list) or len(file_paths) == 0:
            raise ValueError("perf file paths cannot be empty")

        if save_model and (model_save_path is None or len(model_save_path.strip()) == 0):
            raise ValueError("model_save_path required when save_model=True")

    def _parse_perf_files(self) -> None:
        """Parse all perf output files"""
        self.parsed_results = []
        for file_path in self.file_paths:
            logger.info(f"Parsing file: {file_path}")
            event_data = parse_perf_output(file_path)
            self.parsed_results.append(event_data)

    def _check_duplicate_events(self) -> List[str]:
        """Detect duplicate performance events"""
        events_list = []
        repeat_events = []
        for data in self.parsed_results:
            for key in data.keys():
                if key in events_list:
                    repeat_events.append(key)
                else:
                    events_list.append(key)
        logger.info(f"Found {len(repeat_events)} duplicate events")
        return repeat_events

    def _merge_and_clean_events(self, repeat_events: List[str]) -> None:
        """Merge event data and clean invalid data"""
        event_counts = {}
        for data in self.parsed_results:
            event_counts.update(data)

        for event_name in event_counts.keys():
            event_counts[event_name] = event_counts[event_name][:self.DATA_TRUNCATE_LENGTH]

        wrong_events = []
        for event in event_counts.keys():
            if len(event_counts[event]) != self.DATA_TRUNCATE_LENGTH:
                wrong_events.append(event)

        for key in wrong_events:
            del event_counts[key]
            logger.warning(f"Removed invalid event: {key} (length mismatch)")

        self.event_counts = event_counts

    def _prepare_train_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training features X and labels y"""
        if self.selected_events:
            event_counts_train = {k: self.event_counts[k] for k in self.selected_events}
            logger.info(f"Training with {len(self.selected_events)} selected features")
        else:
            event_counts_train = self.event_counts
            logger.info(f"Training with all {len(self.event_counts)} valid features")

        X = np.array([event_counts_train[event] for event in event_counts_train]).T
        y = np.array(self.FIXED_LABEL)
        return X, y

    def train(self) -> None:
        """Train Random Forest model"""
        self._parse_perf_files()
        repeat_events = self._check_duplicate_events()
        self._merge_and_clean_events(repeat_events)
        X, y = self._prepare_train_data()

        logger.info("Training Random Forest model...")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X, y)
        logger.info("Model training completed")

        feature_importances = self.model.feature_importances_
        features = list(self.event_counts.keys() if self.selected_events is None else self.selected_events)
        self.sorted_feature_importances = sorted(
            zip(features, feature_importances),
            key=lambda x: x[1],
            reverse=True
        )

        if self.save_model:
            self._save_model()

    def _save_model(self) -> None:
        """Save model to file"""
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(self.model, f)
        logger.info(f"Model saved to: {self.model_save_path}")

    def get_sorted_feature_importances(self) -> List[Tuple[str, float]]:
        """Get sorted feature importances (descending)"""
        if not self.sorted_feature_importances:
            raise RuntimeError("Please call train() first")
        return self.sorted_feature_importances

    def get_top_k_events(self, k: int = 10) -> List[str]:
        """Get top-k important features"""
        sorted_imp = self.get_sorted_feature_importances()
        return [item[0] for item in sorted_imp[:k]]
