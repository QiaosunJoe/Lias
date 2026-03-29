# Lias

ICPP25: Lias: Leveraging Performance Counters for Interference Quantification and Mitigation in Multi-processor Systems

## Introduction

Lias is a system for interference quantification and mitigation in multi-processor systems. It quantifies, detects, and mitigates interference between applications by analyzing hardware performance counters.

## Notes

This project is a temporary version with unpolished features or potential bugs. All configurations are based on our experimental setup, so extensibility is limited. We apologize for any inconvenience.

We have primarily open-sourced the details of counter selection (including counters based on correlation coefficients for calculating interference entropy, and counters for resource bottleneck localization using random forest models).

Due to environmental issues and experimental setup limitations, we have not yet completed the implementation of a general scheduler, but the scheduler's heuristic logic remains consistent with the paper and can be reproduced according to individual experimental setups.

## Project Structure

```
Lias/
├── configs/                    # Configuration files directory
│   └── selected_counters/      # Selected performance counter configurations
├── data/                      # Data directory
│   └── perf_output/          # Perf output data
├── events/                    # Perf event group files
├── models/                   # Model storage directory
├── src/                      # Source code directory
│   ├── collector/            # Performance data collection module
│   ├── profiler/             # Performance Counter selection module
│   └── utils/                # Utility functions
└── README.md
```

## Dependency Installation

```bash
pip install numpy scipy scikit-learn pandas
```

System dependencies:
```bash
# perf tool
sudo apt-get install linux-tools-common linux-tools-generic

# pqos tool (for COS configuration)
sudo apt-get install intel-cmt-cat

```

## Usage

### 1. Performance Counter Selection

```bash
python src/profiler/profiler.py
```

Configuration instructions:

You need to modify the configuration in profiler.py according to your experimental setup:
APP_NAME, FOLDER_PATH, and TOP_K

### 2. Performance Data Collection

```python
from src.collector.collector import PerfCollector

collector = PerfCollector(
    sampling_interval_ms=1000,
    monitor_events=["cycles", "instructions"],
    pid=12345,
    output_dir="./data/perf_output",
    duration_sec=60
)

collector.start_collect_block()
```

## Core Modules

- **RFTrainer**: Random Forest model training, outputs feature importance
- **PearsonFeatureSelector**: Pearson correlation coefficient selection
- **PerfCollector**: perf data collection

# Citation

If you find this work useful in your research, please cite our paper:
```
@inproceedings{qiao2025lias,
  title={Lias: Leveraging Performance Counters for Interference Quantification and Mitigation in Multi-processor Systems},
  author={Qiao, Yangfan and Li, Zhuozhao},
  booktitle={Proceedings of the 54th International Conference on Parallel Processing},
  pages={521--530},
  year={2025}
}
```