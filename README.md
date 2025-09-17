
# Non-Myopic Active Perception for LTL-Specified Multi-Target Tracking

This repository contains the **official implementation** of our paper:

> **Non-Myopic Active Perception for LTL-Specified Multi-Target Tracking**  
> Zhen Tian, Zhou Zhou, Lei Xie, Xia Hua, Chenyang Wang*, F. Richard Yu  
> *ICASSP 2025 (to appear), arXiv preprint available soon*

---

## 🔍 Overview
We propose **LTL-MCTS**, a non-myopic active perception framework that integrates:
- **Linear Temporal Logic (LTL)** for high-level task specification  
- **Poisson Multi-Bernoulli Mixture (PMBM)** filter for multi-target tracking  
- **Monte Carlo Tree Search (MCTS)** for non-myopic planning  
- **Semantic–Probabilistic Bridge** to evaluate LTL constraints on uncertain beliefs  

This repository provides reproducible experiments for two scenarios:
1. **Persistent Surveillance** (`ltl_mcts_scenario1_persistent_surveillance`)  
2. **Patrol and Respond** (`ltl_mcts_scenario2_patrol_respond`)  

---

## 📂 Repository Structure
```

gm-embodied-ltl-mcts-2025/
├── ltl\_mcts\_scenario1\_persistent\_surveillance/   # Scenario 1: Persistent Surveillance
├── ltl\_mcts\_scenario2\_patrol\_respond/            # Scenario 2: Patrol and Respond
├── LICENSE
├── requirements.txt
└── README.md   # (this file)

````

Each scenario folder includes:
- Source code (`src/`)
- Configurations (`configs/`)
- Experiment runners (`experiments/`)
- Results and logs (`results/`)
- A dedicated README with detailed instructions

---

## ⚙️ Installation
1. Clone the repository:
```bash
git clone https://github.com/gm-embodied/ltl-mcts-2025.git
cd ltl-mcts-2025
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Scenario 1: Persistent Surveillance

```bash
cd ltl_mcts_scenario1_persistent_surveillance
python experiments/run_experiment.py
```

### Scenario 2: Patrol and Respond

```bash
cd ltl_mcts_scenario2_patrol_respond
python experiments/run_experiment.py
```

See each scenario’s README for detailed configuration and expected results.

---

## 📊 Expected Results

* **Scenario 1**: Non-myopic planning yields higher success rates and balanced surveillance coverage.
* **Scenario 2**: LTL-MCTS efficiently completes patrol duties while resisting distraction targets.

Plots and metrics (success rate, OSPA error, planning time, etc.) are automatically generated in each `results/` folder.

---

## 📑 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{tian2025ltlmcts,
  title={Non-Myopic Active Perception for LTL-Specified Multi-Target Tracking},
  author={Tian, Zhen and Zhou, Zhou and Xie, Lei and Hua, Xia and Wang, Chenyang and Yu, F. Richard},
  booktitle={ICASSP},
  year={2025}
}
```

The arXiv preprint will be available soon.

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📬 Contact

For questions, please open a GitHub issue or contact:
**Chenyang Wang** (Corresponding author) – [chenyangwang@ieee.org](mailto:chenyangwang@ieee.org)

