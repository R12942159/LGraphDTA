import copy
import os
import torch
import numpy as np
from utils import logger, train_val, val, get_metrics_reg, save_pkl
from params import SEED, DEVICE, HP, BATCH_SIZE, LR, EPOCH, COMBINED_TRAINING_SET
from helpers import CustomTrial, CustomDataLoader, load_data
from models import LGraph

import warnings
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    logger.info(f"GPU will be used for training ({torch.cuda.get_device_name()})")
    logger.info(f"Seed: {SEED}. Epoch: {EPOCH}, Batch size: {BATCH_SIZE}. Learning rate: {LR}")
else:
    logger.info("CPUs will be used for training")


def run_dataset_test(dataset, model, folds, ckpt_path=None):
    epochs = EPOCH
    df_train_val, df_test, val_folds, test_fold, protein_to_graph, ligand_to_graph, ligand_to_ecfp = load_data(dataset)

    for fold, idx_val in enumerate(val_folds):
        if fold not in folds:
            continue
        logger.info(f"Testing fold {fold} on {dataset} dataset")

        if COMBINED_TRAINING_SET:
            df_train = df_train_val
        else:
            df_train = df_train_val[~ df_train_val.index.isin(idx_val)]

        val_dl = CustomDataLoader(df=df_test, batch_size=BATCH_SIZE, device=DEVICE, # original batch_size: 32
                                   e1_key_to_graph=ligand_to_graph,
                                   e2_key_to_graph=protein_to_graph,
                                   e1_key_to_fp=ligand_to_ecfp,
                                   shuffle=False)
        train_dl = CustomDataLoader(df=df_train, batch_size=BATCH_SIZE, device=DEVICE, # df=df_train, original batch_size: 32
                                    e1_key_to_graph=ligand_to_graph,
                                    e2_key_to_graph=protein_to_graph,
                                    e1_key_to_fp=ligand_to_ecfp,
                                    shuffle=True)

        model_copy = copy.deepcopy(model)
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=LR)
        criterion = torch.nn.MSELoss()
        epoch_to_metrics = train_val(model=model_copy, optimizer=optimizer, criterion=criterion,
                                     train_dl=train_dl, val_dl=val_dl, epochs=epochs,
                                     score_fn=get_metrics_reg, fold=fold, verbose=True, 
                                     with_rm2=True, with_ci=True, val_nth_epoch=1,
                                     save_ckpt_path=f"ckpt/{dataset}_fold{fold}.ckpt")

        save_pkl(epoch_to_metrics, f"results/{dataset}-fold_{fold}.pkl")

def run_ckpt_test(dataset, model, folds, checkpoint_path):
    df_train_val, df_test, val_folds, test_fold, protein_to_graph, ligand_to_graph, ligand_to_ecfp = load_data(dataset)
    
    for fold in folds:
        logger.info(f"Predicting fold {fold} on {dataset}")

        val_dl = CustomDataLoader(df=df_test, batch_size=BATCH_SIZE, device=DEVICE,
                                   e1_key_to_graph=ligand_to_graph,
                                   e2_key_to_graph=protein_to_graph,
                                   e1_key_to_fp=ligand_to_ecfp,
                                   shuffle=False)
        ckpt_path = checkpoint_path.format(dataset=dataset, fold=fold)
        logger.info(f"Loading checkpoint from {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        criterion = torch.nn.MSELoss()

        y_true, y_pred, epoch_loss = val(model, val_dl, criterion)
        y_true = y_true.numpy()
        y_pred = y_pred.numpy()

        save_pkl({"pred": y_pred, "label": y_true, "loss": epoch_loss}, f"bootstrap/{dataset}_fold{fold}.pkl")
        logger.info(f"Saved results to bootstrap/{dataset}_fold{fold}.pkl | MSE: {epoch_loss:.4f}")

def main(datasets, folds, checkpoint_path=None):
    torch.cuda.empty_cache()
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    model = LGraph(trial=CustomTrial(hp=HP)).to(DEVICE)
    logger.info(f"Model Architecture: {model}")

    os.makedirs("results/", exist_ok=True)
    os.makedirs("bootstrap/", exist_ok=True)
    if checkpoint_path:
        logger.info(f"Running prediction only using checkpoint: {checkpoint_path}")
        if "davis" in datasets:
            run_ckpt_test("davis", model, folds, checkpoint_path=checkpoint_path)
        if "kiba" in datasets:
            run_ckpt_test("kiba", model, folds, checkpoint_path=checkpoint_path)
    else:
        if "davis" in datasets:
            run_dataset_test("davis", model, folds)
        if "kiba" in datasets:
            run_dataset_test("kiba", model, folds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', type=str, required=False, nargs='+',
                        default=["davis", "kiba"], choices=["davis", "kiba"])
    parser.add_argument('--folds', type=int, required=False, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--checkpoint_path', type=str, required=False,
                        help="Path to checkpoint. Use {fold} as placeholder for fold number if needed.")
    args = parser.parse_args()
    main(args.datasets, args.folds, args.checkpoint_path)