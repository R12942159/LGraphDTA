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
pip uninstall -y torch torchvision \
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

### Download datasets and checkpoints
- **To download both the datasets and pretrained checkpoints, run the following command:** <br>
    bash download_dataset.sh

This script will:
- Download the LGraphDTA datasets and unzip them.
- Download the pretrained checkpoint files (ckpt-LGraphDTA.zip) and unzip them
- Remove the downloaded zip files after extraction

## Usage & Ablation Study
### Test the model using pretrained checkpoints with 5-fold cross validation
If you have downloaded the pretrained checkpoints in `ckpt-LGraphDTA/5-fold/`, you can test each fold as follows:

- **To test on the Davis dataset:**  
  - Fold 0: <br>
    python3 test.py --datasets davis --folds 4 --checkpoint_path ckpt-LGraphDTA/5-fold/davis_fold0.ckpt <br>
    → results saved to: results/davis_fold0.pkl  
  - Fold 1: <br>
    python3 test.py --datasets davis --folds 4 --checkpoint_path ckpt-LGraphDTA/5-fold/davis_fold1.ckpt <br>
    → results saved to: results/davis_fold1.pkl  
  - Fold 2: <br>
    python3 test.py --datasets davis --folds 4 --checkpoint_path ckpt-LGraphDTA/5-fold/davis_fold2.ckpt <br>
    → results saved to: results/davis_fold2.pkl  
  - Fold 3: <br>
    python3 test.py --datasets davis --folds 4 --checkpoint_path ckpt-LGraphDTA/5-fold/davis_fold3.ckpt <br>
    → results saved to: results/davis_fold3.pkl  
  - Fold 4: <br>
    python3 test.py --datasets davis --folds 4 --checkpoint_path ckpt-LGraphDTA/5-fold/davis_fold4.ckpt <br>
    → results saved to: results/davis_fold4.pkl  

- **To test on the KIBA dataset:**  
  - Fold 0: <br>
    python3 test.py --datasets kiba --folds 4 --checkpoint_path ckpt-LGraphDTA/5-fold/kiba_fold0.ckpt <br>
    → results saved to: results/kiba_fold0.pkl  
  - Fold 1: <br>
    python3 test.py --datasets kiba --folds 4 --checkpoint_path ckpt-LGraphDTA/5-fold/kiba_fold1.ckpt <br>
    → results saved to: results/kiba_fold1.pkl  
  - Fold 2: <br>
    python3 test.py --datasets kiba --folds 4 --checkpoint_path ckpt-LGraphDTA/5-fold/kiba_fold2.ckpt <br>
    → results saved to: results/kiba_fold2.pkl  
  - Fold 3: <br>
    python3 test.py --datasets kiba --folds 4 --checkpoint_path ckpt-LGraphDTA/5-fold/kiba_fold3.ckpt <br>
    → results saved to: results/kiba_fold3.pkl  
  - Fold 4: <br>
    python3 test.py --datasets kiba --folds 4 --checkpoint_path ckpt-LGraphDTA/5-fold/kiba_fold4.ckpt <br>
    → results saved to: results/kiba_fold4.pkl

### Test the model using pretrained checkpoints with train-test split
- **To test on the Davis dataset:**  
  - python3 test.py --datasets davis --folds 0 --checkpoint_path ckpt-LGraphDTA/1-fold/davis_fold0_0.143.ckpt <br>
    → results saved to: results/davis_fold0.pkl  
- **To test on the KIBA dataset:**  
  - python3 test.py --datasets kiba --folds 0 --checkpoint_path ckpt-LGraphDTA/1-fold/kiba_fold0_0.116.ckpt <br>
    → results saved to: results/kiba_fold0.pkl  

### Train & Test the model with 5-fold cross validation
- **To train the LGraphDTA model on the Davis dataset, run:** <br>
    python test.py --dataset davis
- **To train the LGraphDTA model on the Kiba dataset, run:** <br>
    python test.py --dataset kiba

### Train & Test the model with train-test split
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