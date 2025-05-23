# Neural Operators for Stochastic Modeling of System Response to Natural Hazards

**[Somdatta Goswami](https://scholar.google.com/citations?user=GaKrpSkAAAAJ&hl=en), [Dimitris Giovanis](https://scholar.google.com/citations?user=dnFLyp4AAAAJ&hl=en), [Bowei Li](https://scholar.google.com/citations?user=MDVtPqwAAAAJ&hl=en), [Seymour Spence](https://scholar.google.com/citations?user=gDH80t0AAAAJ), [Michael D. Shields](https://scholar.google.com/citations?user=hc85Ll0AAAAJ)**

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-2502.11279-b31b1b.svg" />
</p>
<p align="center">
  <a href="https://arxiv.org/abs/2502.11279"><strong>Paper</strong></a> •
  <a href="https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/sgoswam4_jh_edu/ElqEfANCWC5BrvojtY_vCHoBF5T_3ZtnVxyQUs3UMDuGVQ?e=OBqf1s"><strong>Data</strong></a>
</p>

---

## Overview

We propose and evaluate **neural operator frameworks** (self-adaptive FNO and DeepFNOnet)to model stochastic response of nonlinear structural systems under seismic and wind excitations. The approach is benchmarked against traditional simulators and surrogate models, achieving high accuracy and generalizability under uncertainty.

---

## 📁 Repository Structure
├── data/
│ ├── raw_data/ # OpenSees simulation outputs
│ ├── preprocessed_data/ # Inputs to neural operator models
│ └── additional_results/ # Extended results (e.g., DeepFNO for wind problem)
├── models/
│ ├── DeepONet/
│ ├── DeepFNO/
│ └── BaselineModels/
├── scripts/
│ ├── preprocess/
│ ├── train/
│ ├── evaluate/
│ └── visualize/
├── notebooks/
│ └── exploratory_analysis.ipynb
├── requirements.txt
└── README.md


---

## 🔗 Data Access

You can download the dataset used in this study from the link below:

📂 **[Download Data](https://livejohnshopkins-my.sharepoint.com/:f:/g/personal/sgoswam4_jh_edu/ElqEfANCWC5BrvojtY_vCHoBF5T_3ZtnVxyQUs3UMDuGVQ?e=OBqf1s)**

Contents:
- `raw_data/`: OpenSees simulation outputs (earthquake and wind).
- `preprocessed_data/`: Ready-to-use data for model training and testing.
- `additional_results/`: Results related to DeepFNO on wind datasets.

Preprocessing scripts are provided in `scripts/preprocess/`.

---

## 🛠️ Installation

Clone the repo and set up the environment:

```bash
git clone https://github.com/YOUR-USERNAME/neural-operators-hazard-response.git
cd neural-operators-hazard-response
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

🚀 Running the Code

Preprocessing
python scripts/preprocess/generate_training_data.py --input-dir data/raw_data/ --output-dir data/preprocessed_data/
Training
Train DeepONet or DeepFNO:

python scripts/train/train_deeponet.py --config configs/deeponet.yaml
python scripts/train/train_deepfno.py --config configs/deepfno.yaml
Evaluation
python scripts/evaluate/evaluate_model.py --model deeponet --data-dir data/preprocessed_data/
📊 Results

Our proposed models outperform traditional surrogate models, especially in extrapolative regimes and under varying uncertainty structures.

For detailed results and visualizations, refer to:

notebooks/exploratory_analysis.ipynb
data/additional_results/ for DeepFNO wind case study.
📌 Dependencies

Key libraries:

PyTorch
DeepXDE / NeuralOperators
NumPy, SciPy, Matplotlib
OpenSees (for generating raw data)
See requirements.txt for a full list.

📖 Citation

If you use this code or data in your work, please cite:

@article{goswami2024neural,
  title={Neural Operators for Stochastic Modeling of Nonlinear Structural System Response to Natural Hazards},
  author={Goswami, Somdatta and Giovanis, Dimitris and Li, Bowei and Spence, Seymour and Shields, Michael D},
  journal={arXiv preprint arXiv:2502.11279},
  year={2024}
}
📬 Contact

For questions or collaboration inquiries, contact:

Somdatta Goswami
📧 sgoswam4 [at] jh.edu


Let me know if you’d like help customizing `requirements.txt`, example configs, or setting up Colab/Binder integration.

