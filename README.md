# LGraphDTA: Integrating LLM and GNN for Accurate Drug-Target Affinity Prediction
LGraphDTA integrates large language models and graph neural networks for accurate prediction of drug–target affinity. By fusing global sequence features with structural graphs through the NAM module, it captures both local and long-range interaction patterns, achieving state-of-the-art performance on benchmark datasets.

## Installation
### Prerequisites
- Python 3.8.20
- NVIDIA Driver with CUDA 12.2 support (e.g., Driver Version 535.183.01)

### Install dependencies
- git clone https://github.com/R12942159/LGraphDTA.git
- cd LGraphDTA
- pip install -r requirements.txt

### For CUDA 12.2 environment, install PyTorch with CUDA 11.3 support as PyTorch currently does not support CUDA 12 directly:
pip uninstall -y torch torchvision
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

### Download datasets
- bash download_dataset.sh

## Usage & Ablation Study
### Train & Test the model with 5-fold cross validation
- **To train the LGraphDTA model on the Davis dataset, run:** <br>
    python test.py --dataset davis
- **To train the LGraphDTA model on the Kiba dataset, run:** <br>
    python test.py --dataset kiba

### Train & Test the model with 1-fold cross validation
Before running the script, make sure to open params.py and set the COMBINED_TRAINING_SET parameter to True.
- **To train the LGraphDTA model on the Davis dataset, run:** <br>
    python test.py --dataset davis
- **To train the LGraphDTA model on the Kiba dataset, run:** <br>
    python test.py --dataset kiba

### Reproduce Ablation Studies
To reproduce each ablation setting, edit the following flags in `params.py` accordingly.

| Ablation Experiment                | Flag in `params.py`                            | Set To         |
|------------------------------------|------------------------------------------------|----------------|
| LGraphDTA w/o ESM2                 |`LGRAPHDTA_WITHOUT_ESM2` & `N_PROT_NODE_FEAT`   | `True` & `41`  |
| LGraphDTA w/o Domain               |`LGRAPHDTA_WITHOUT_FEATURE` & `N_PROT_NODE_FEAT`| `True` & `1280`|
| LGraphDTA w/o Fingerprint          |`LGRAPHDTA_WITHOUT_FP`                          | `True`         |
| Replace ESM2 with random embeddings|`LGRAPHDTA_RANDOM_EMBEDDING`                    | `True`         |
| Replace ESM2 with LLM embeddings   |`LGRAPHDTA_LLAMA_EMBEDDING` & `N_PROT_NODE_FEAT`| `True` & `1577`|