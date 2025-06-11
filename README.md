# LGraphDTA: Integrating LLM and GNN for Accurate Drug-Target Affinity Prediction
LGraphDTA integrates large language models and graph neural networks for accurate prediction of drug–target affinity. By fusing global sequence features with structural graphs through the NAM module, it captures both local and long-range interaction patterns, achieving state-of-the-art performance on benchmark datasets.

## Installation
### Prerequisites
- Python 3.11.7

### Install dependencies
- git clone https://github.com/R12942159/LGraphDTA.git
- pip install -r requirements.txt

### Download datasets
- bash download_dataset.sh

## Usage & Ablation Study
### Train the model
- **To train the LGraphDTA model on the Davis dataset, run:** <br>
    python train.py --dataset davis
- **To train the LGraphDTA model on the Kiba dataset, run:** <br>
    python train.py --dataset kiba

### Reproduce Ablation Studies
To reproduce each ablation setting, edit the following flags in `params.py` accordingly.

| Ablation Experiment                     | Flag in `params.py`             | Set To  |
|-----------------------------------------|---------------------------------|---------|
| LGraphDTA w/o Fingerprint               | `LGRAPHDTA_WITHOUT_FP`          | `True`  |
| LGraphDTA w/o ESM2                      | `LGRAPHDTA_WITHOUT_ESM2`        | `True`  |
| LGraphDTA w/o Domain                    | `LGRAPHDTA_WITHOUT_FEATURE`     | `True`  |
| Replace ESM2 with random embeddings     | `LGRAPHDTA_RANDOM_EMBEDDING`    | `True`  |
| Replace ESM2 with LLM embeddings        | `LGRAPHDTA_LLAMA_EMBEDDING`     | `True`  |