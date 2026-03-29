#!/usr/bin/env python3
# coding: utf-8
import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.utils import EVENT_GROUP_FILES
from src.profiler.rf_trainer import RFTrainer
from src.profiler.selector import PearsonFeatureSelector


def save_counters_to_json(app_name: str, counter_list: list, save_dir: str, 
                         model_path: str = None, selector_type: str = "rf") -> str:
    """Save selected counters to JSON config file"""
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{app_name}_selected_counters_{selector_type}.json")
    
    config_data = {
        "app_name": app_name,
        "perf_counters": counter_list,
        "top_k": len(counter_list),
        "selector_type": selector_type,
        "model_path": model_path
    }
    
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4, ensure_ascii=False)
    return save_path


def run_rf_trainer_demo(app_name: str, folder_path: str, top_k: int = 10) -> tuple:
    """Run Random Forest feature selection demo"""
    print("=" * 50)
    print("Demo 1: Random Forest Feature Selection")
    print("=" * 50)
    
    file_paths = [folder_path + file_name for file_name in EVENT_GROUP_FILES]
    
    # Train with all features to get importance ranking
    print("\n1.1 Training with all features")
    trainer1 = RFTrainer(
        file_paths=file_paths,
        selected_events=None,
        save_model=False
    )
    trainer1.train()
    
    sorted_imp = trainer1.get_sorted_feature_importances()
    print("\nTop 20 Important Performance Counters:")
    for feature, imp in sorted_imp[:20]:
        print(f"{feature}: {imp:.4f}")

    # Train with top-k features and save model
    print(f"\n1.2 Training with top-{top_k} features")
    top_k_events = trainer1.get_top_k_events(k=top_k)
    
    model_filename = f'rf_model_{folder_path.split("/")[-2]}.pkl'
    model_save_path = os.path.join(MODEL_SAVE_DIR, model_filename)
    
    trainer2 = RFTrainer(
        file_paths=file_paths,
        selected_events=top_k_events,
        save_model=True,
        model_save_path=model_save_path
    )
    trainer2.train()

    final_imp = trainer2.get_sorted_feature_importances()
    print(f"\nTop-{top_k} Feature Importance:")
    for feature, imp in final_imp:
        print(f"{feature}: {imp:.4f}")

    return top_k_events, model_save_path


def run_pearson_selector_demo(app_name: str, folder_path: str, top_k: int = 10) -> list:
    """Run Pearson correlation feature selection demo"""
    print("\n" + "=" * 50)
    print("Demo 2: Pearson Correlation + Stability Selection")
    print("=" * 50)
    
    selector = PearsonFeatureSelector(folder_path=folder_path)
    selector.select()

    top_k_events_with_std = selector.get_top_k_stable_events(k=top_k)
    top_k_event_names = [event[0] for event in top_k_events_with_std]

    print("\n" + "="*60)
    print(f"App [{app_name}] Pearson Selection Top-{top_k} Counters")
    print("="*60)
    for idx, event_name in enumerate(top_k_event_names, 1):
        print(f"Top{idx}: {event_name}")

    return top_k_event_names


if __name__ == '__main__':
    # Configuration
    APP_NAME = "xapian"
    
    FOLDER_PATH = '' # fill your folder path here
    # FOLDER_PATH should contain {group_name}.txt in EVENT_GROUP_FILES, as well as corresponding {group_name}_lat.log files
    # Example folder structure for our '/data/yfqiao/exper/data/xapian/':
    # - cache_specific_events_lat.log, cache_specific_events.txt
    # - hardware_events_lat.log, hardware_events.txt
    # ...
    #
    # Latency log format (latency values in ns):
    # === Fri Dec 27 11:44:23 2024 ===
    # 20241228-01:44:23/
    # RPS:1781
    # 9221047
    #
    # === Fri Dec 27 11:44:24 2024 ===
    # 20241228-01:44:24/
    # RPS:1807
    # 9106102
    
    MODEL_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
    CONFIG_SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "configs", "selected_counters")
    TOP_K = 10

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(CONFIG_SAVE_DIR, exist_ok=True)

    # Demo 1: RFTrainer
    top10_events_rf, model_save_path = run_rf_trainer_demo(APP_NAME, FOLDER_PATH, TOP_K)

    json_save_path_rf = save_counters_to_json(
        app_name=APP_NAME,
        counter_list=top10_events_rf,
        save_dir=CONFIG_SAVE_DIR,
        model_path=model_save_path,
        selector_type="rf"
    )

    print("\n" + "="*60)
    print(f"App [{APP_NAME}] RF Selection Top-{TOP_K} Counters Saved!")
    print(f"Config: {json_save_path_rf}")
    print(f"Model: {model_save_path}")

    # Demo 2: PearsonFeatureSelector
    top_k_event_names_pearson = run_pearson_selector_demo(APP_NAME, FOLDER_PATH, TOP_K)

    json_save_path_pearson = save_counters_to_json(
        app_name=APP_NAME,
        counter_list=top_k_event_names_pearson,
        save_dir=CONFIG_SAVE_DIR,
        selector_type="pearson"
    )

    print("\nConfig saved! Scheduler can read directly:")
    print(f"Config: {json_save_path_pearson}")
