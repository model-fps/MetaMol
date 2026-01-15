import json
import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset, sampler, DataLoader
from tqdm import tqdm
from multiprocessing import Pool
from dataset.data_util import get_chembl_targets, get_bdb_targets
from dataset.data_util import preprocess_targets, BaseMetaDataset
import time
import gc


def batch_process(pool, func, data, batch_size=50):
    results = []
    for i in tqdm(range(0, len(data), batch_size), desc="Processing"):
        batch = data[i:i + batch_size]
        res = pool.map(func, batch)
        results.extend(res)
        del batch
        time.sleep(0.1)
    return results


class CHEM_BDBMetaDataset(BaseMetaDataset):
    def __init__(self, args, exp_string):
        super(CHEM_BDBMetaDataset, self).__init__(args, exp_string)

    def load_dataset(self):
        # assert self.args.datasource == "chembl_bdb"
        print("Loading bdb data---")
        experiment_bdb = get_bdb_targets()
        print("Loading chembl data---")
        experiment_chembl = get_chembl_targets()
        print("Finish loading.")
        # Merge the datasets
        experiment_train = {
            "ligand_sets": {**experiment_chembl["ligand_sets"], **experiment_bdb["ligand_sets"]},
            "targets": experiment_chembl["targets"] + experiment_bdb["targets"]
        }
        self.target_ids = experiment_train["targets"]
        self.chembl_target_ids = experiment_chembl["targets"]
        self.bdb_target_ids = experiment_bdb["targets"]
        print(f'merged experiment_train:{len(experiment_train)}' )

        self.split_name_train_val_test = {}
        chembl_valid_test_path = 'D:/Author/code/model/data/chembl/chembl_valid_test_split.json'
        chembl_valid_test_data = json.load(open(chembl_valid_test_path, "r"))
        chembl_train_path = 'D:/Author/code/model/data/chembl/chembl_train_split.json'
        chembl_train_data = json.load(open(chembl_train_path, "r"))

        bdb_valid_test_path = 'D:/Author/code/model/data/bdb/bdb_valid_test_split.json'
        bdb_valid_test_data = json.load(open(bdb_valid_test_path, "r"))
        bdb_train_path = 'D:/Author/code/model/data/bdb/bdb_train_split.json'
        bdb_train_data = json.load(open(bdb_train_path, "r"))
        self.split_name_train_val_test = {
            'train': chembl_train_data['train'] + bdb_train_data['train'],
            'valid': chembl_valid_test_data['valid'] + bdb_valid_test_data['valid'],
            'test': chembl_valid_test_data['test'] + bdb_valid_test_data['test']
        }
        print(f"number of chembl_bdb training set:", len(self.split_name_train_val_test['train']))

        ligand_set = experiment_train["ligand_sets"]
        self.n_targets = len(ligand_set)
        chembl_ligand_set = experiment_chembl["ligand_sets"]
        bdb_ligand_set = experiment_bdb["ligand_sets"]

        self.Xs = []
        self.smiles_all = []
        self.ys = []
        self.targets = []
        self.train_indices = []
        self.val_indices = []
        self.test_indices = []

        target_list = []

        # test set
        # if self.args.expert_test != "":
        #     if self.args.expert_test == "act_cliff":
        #         experiment_test = get_activity_cliff_targets()
        #     else:
        #         raise ValueError(f"no expert_test {self.args.expert_test}")
        #     ligand_set = {**ligand_set, **experiment_test['ligand_sets']}
        #     self.split_name_train_val_test['test'] = experiment_test['targets']
        target_list += self.split_name_train_val_test['test']
        # valid set
        target_list += self.split_name_train_val_test['valid']
        # train set
        if self.args.train == 1:
            target_list += self.split_name_train_val_test['train']
        print('target_list', len(target_list))

        data_cnt = 0
        with Pool(1) as p:
            res_all = batch_process(p, preprocess_targets, [ligand_set.get(x, None) for x in target_list])

        for res, target_id in zip(res_all, target_list):

                if res is None:
                    continue
                x_tmp, y_tmp, smiles_list = res

                self.Xs.append(x_tmp)
                self.ys.append(y_tmp)
                self.smiles_all.append(smiles_list)
                self.targets.append(target_id)
                if target_id in self.split_name_train_val_test['train']:
                    self.train_indices.append(data_cnt)
                    data_cnt += 1
                elif target_id in self.split_name_train_val_test['valid']:
                    self.val_indices.append(data_cnt)
                    data_cnt += 1
                elif target_id in self.split_name_train_val_test['test']:
                    self.test_indices.append(data_cnt)
                    data_cnt += 1
                else:
                    print(target_id)
                    data_cnt += 1

        train_cnt = len(self.train_indices)
        val_cnt = len(self.val_indices)
        test_cnt = len(self.test_indices)

        self.data_length = {}
        self.data_length['train'] = train_cnt
        self.data_length['val'] = val_cnt
        self.data_length['test'] = test_cnt
        self.data_length['train_weight'] = train_cnt

