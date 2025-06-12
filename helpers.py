import math
import json

import torch
import pandas as pd
import numpy as np

from utils import logger, GraphPairsBatchConstructor, load_pkl
from params import N_CHEM_NODE_FEAT, N_CHEM_EDGE_FEAT, N_PROT_EDGE_FEAT, N_PROT_NODE_FEAT
from params import LGRAPHDTA, LGRAPHDTA_WITHOUT_ESM2, LGRAPHDTA_WITHOUT_DOMAIN, COLD_SETTING, LGRAPHDTA_RANDOM_EMBEDDING, LGRAPHDTA_LLAMA_EMBEDDING


class CustomTrial:
    def __init__(self, hp):
        self.hp = hp
        self.suggest_int = self.get_from_dict
        self.suggest_float = self.get_from_dict
        self.suggest_categorical = self.get_from_dict
        self.suggest_discrete_uniform = self.get_from_dict
        self.suggest_loguniform = self.get_from_dict
        self.suggest_uniform = self.get_from_dict

    def get_from_dict(self, key, *args, **kwargs):
        try:
            param = self.hp[key]
        except KeyError:
            param = args[0]
            try:
                iter(param)
                param = param[0]
            except TypeError:
                pass
            logger.debug(f"No key {key} in hyper parameters, {param} will be used instead")

        return param


class CustomDataLoader:
    key_kwargs = {"e1_key": "ligand", "e2_key": "protein", "label_key": "label"}

    def __init__(self, df, batch_size, device, e1_key_to_graph, e2_key_to_graph, e1_key_to_fp, shuffle=False):
        self.df = df
        self.batch_size = batch_size
        self.device = device
        self.e1_key_to_fp = e1_key_to_fp
        self.shuffle = shuffle

        self.n_batches = math.ceil(self.df.shape[0] / self.batch_size)
        self.random_state = 0
        self.batch_creator = GraphPairsBatchConstructor(e1_key_to_graph, e2_key_to_graph, self.device,
                                                        e1_node_features_len=N_CHEM_NODE_FEAT,
                                                        e2_node_features_len=N_PROT_NODE_FEAT, # 1280+41
                                                        e1_edge_features_len=N_CHEM_EDGE_FEAT,
                                                        e2_edge_features_len=N_PROT_EDGE_FEAT)
        self.df_batches = None

    def split(self):
        if self.shuffle:
            self.df_batches = np.array_split(self.df.sample(frac=1, random_state=self.random_state), self.n_batches)
            self.random_state += 1
        else:
            self.df_batches = np.array_split(self.df, self.n_batches)

    def __iter__(self):
        self.curr_batch = 0
        self.split()
        return self

    def __next__(self):
        if self.curr_batch < self.n_batches:
            features = self.get_features(self.df_batches[self.curr_batch])
            self.curr_batch += 1
            return features
        else:
            raise StopIteration

    def get_features(self, df_batch):
        e1_graph, e2_graph = self.batch_creator.get_batch(df_batch, **self.key_kwargs)
        e1_fp = torch.from_numpy(np.vstack(df_batch[self.key_kwargs["e1_key"]].map(self.e1_key_to_fp))).type(
            torch.float32).to(self.device)
        y = torch.from_numpy(np.array(df_batch[self.key_kwargs["label_key"]])).type(torch.float32).view(-1, 1).to(
            self.device)
        return y, {"e1_graph": e1_graph, "e2_graph": e2_graph, "e1_fp": e1_fp}


def load_data(dataset):
    df = pd.read_csv(f"data/{dataset}/full.csv")

    # Cold target from SSR-DTA
    if COLD_SETTING:
        val_folds = json.load(open(f"data/{dataset}/cold_target/folds/train_fold_setting1.txt"))
        test_fold = json.load(open(f"data/{dataset}/cold_target/folds/.ipynb_checkpoints/test_fold_setting1-checkpoint.txt"))
    else:
        test_fold = json.load(open(f"data/{dataset}/folds/test_fold_setting1.txt"))
        val_folds = json.load(open(f"data/{dataset}/folds/train_fold_setting1.txt"))
    df_train = df[~ df.index.isin(test_fold)]
    df_test = df[df.index.isin(test_fold)]

    ### protein part ###
    if LGRAPHDTA:
        protein_to_graph = load_pkl(f"data/{dataset}/protein_to_graph_41_1280.pkl")
    if LGRAPHDTA_WITHOUT_ESM2:
        protein_to_graph = load_pkl(f"data/{dataset}/protein_to_graph.pkl")  
    if LGRAPHDTA_WITHOUT_DOMAIN:
        protein_to_graph = load_pkl(f"data/{dataset}/protein_to_graph_0_1280.pkl")
    if LGRAPHDTA_RANDOM_EMBEDDING:
        protein_to_graph = load_pkl(f"data/{dataset}/protein_to_graph_41_rd1280.pkl")
    if LGRAPHDTA_LLAMA_EMBEDDING:
        # protein_to_graph = load_pkl(f"data/{dataset}/protein_to_graph_41_llama3072.pkl")
        # protein_to_graph = load_pkl(f"data/{dataset}/protein_to_graph_41_llama2048.pkl")
        protein_to_graph = load_pkl(f"data/{dataset}/protein_to_graph_41_DeepSeek1536.pkl")

    ### ligand part ###
    ligand_to_graph = load_pkl(f"data/{dataset}/ligand_to_graph.pkl")
    ligand_to_ecfp = load_pkl(f"data/{dataset}/ligand_to_ecfp.pkl")

    return df_train, df_test, val_folds, test_fold, protein_to_graph, ligand_to_graph, ligand_to_ecfp