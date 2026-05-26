# Federated Learning-Based Adaptive Intrusion Detection for 6G Network Slices

## Requirements

- Python 3.12
- See `requirements.txt` for package versions

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU acceleration, install the CUDA build of PyTorch from https://pytorch.org/get-started/locally/ (this project was developed with torch 2.5.1+cu121 on an NVIDIA RTX 3070 Ti).

## Datasets

The datasets are **not included** in this repository due to their size and redistribution terms. Download them from their original sources:

- **NSL-KDD** — https://www.unb.ca/cic/datasets/nsl.html
- **CICIDS2017** — https://www.unb.ca/cic/datasets/ids-2017.html
- **CICIoT2023** — https://www.unb.ca/cic/datasets/iotdataset-2023.html

Place them in the following structure:

```
data/
├── NSL-KDD/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
├── CICIDS2017/      # the .csv files
└── CICIoT2023/      # the .csv files
```

## Core Files

This repository centres on four files that form the complete reproducible pipeline:

| File | Purpose |
|------|---------|
| `src/models/fedbn_model.py` | The FedBN model architecture |
| `src/preprocessing/preprocess_multi_datasets.py` | Unifies all three datasets into a common 15-feature, 7-class format |
| `src/experiments/federated_multidataset_cl.py` | The main federated continual learning experiment |
| `src/analysis/generate_final_visualisations.py` | Generates the result figures |

Additional scripts present in the repository are development artefacts from earlier stages of the project and are not required to reproduce the final results.

## Usage

All commands should be run from the **project root**.

### Step 1 — Preprocess the datasets

Aligns all three datasets into a unified 15-feature space with a common 7-class taxonomy, saving the output to `data/processed_unified/`:

```bash
python src/preprocessing/preprocess_multi_datasets.py
```

### Step 2 — Run the federated continual learning experiment

Runs all four strategies (naive, EWC, replay, EWC+replay) and saves results to `results/multidataset_fcl_final_results.json`:

```bash
python src/experiments/federated_multidataset_cl.py
```

### Step 3 — Generate the figures

```bash
python src/analysis/generate_final_visualisations.py
```

## Method Summary

- **Architecture:** FedBN — Batch Normalisation running statistics are kept local to each client while remaining weights are federated, allowing the model to accommodate the differing feature distributions of the three datasets.
- **Federation:** Manual FedAvg aggregation, weighted by client data size.
- **Continual learning:** Task 1 (Benign, DoS/DDoS, Probe/Recon) is common to all clients; Task 2 comprises client-specific emerging attacks. EWC (λ=1000) and a class-balanced replay buffer (2000 samples) are evaluated.
- **Evaluation:** Catastrophic forgetting, measured as the drop in Task 1 accuracy after Task 2 training, is the primary metric.

## Acknowledgements

This work uses the NSL-KDD, CICIDS2017, and CICIoT2023 datasets provided by the Canadian Institute for Cybersecurity (CIC), University of New Brunswick.