import json
import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset, sampler, DataLoader
import tqdm
from multiprocessing import Pool
from dataset.data_util import get_chembl_targets, get_bdb_targets
from dataset.data_util import preprocess_targets, BaseMetaDataset


class MetaDataset(BaseMetaDataset):
    def __init__(self, args, exp_string):
        super(MetaDataset, self).__init__(args, exp_string)

    def load_dataset(self):
        datasource = self.args.datasource

        if datasource == "chembl":
            experiment_train = get_chembl_targets()
        elif datasource == "bdb":
            experiment_train = get_bdb_targets()
        else:
            print("dataset not exist")
            exit()

        self.target_ids = experiment_train["targets"]
        ligand_set = experiment_train["ligand_sets"]
        print(f'len(ligand_set) of {datasource}:', len(ligand_set))

        if datasource == "chembl":
            self.split_name_train_val_test = {}
            valid_test_path = f'D:/Author/code/model/data/chembl/chembl_valid_test_split.json'
            valid_test_data = json.load(open(valid_test_path, "r"))
            self.split_name_train_val_test.update(valid_test_data)
            train_path = f'D:/Author/code/model/data/chembl/chembl_train_split.json'
            train_data = json.load(open(train_path, "r"))
            self.split_name_train_val_test.update(train_data)
            # print(f"number of {datasource} training set:", len(self.split_name_train_val_test['train']))

        elif datasource == "bdb":
            self.split_name_train_val_test = {}
            valid_test_path = f'D:/Author/code/model/data/bdb/bdb_valid_test_split.json'
            valid_test_data = json.load(open(valid_test_path, "r"))
            self.split_name_train_val_test.update(valid_test_data)
            train_path = f'D:/Author/code/model/data/bdb/bdb_train_split.json'
            train_data = json.load(open(train_path, "r"))
            self.split_name_train_val_test.update(train_data)
            # print(f"number of {datasource} training set:", len(self.split_name_train_val_test['train']))

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

        data_cnt = 0
        with Pool(1) as p:
            res_all = p.map(preprocess_targets, tqdm.tqdm([ligand_set.get(x, None) for x in target_list]))
            # print(f"len(res_all):{len(res_all)}")
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
        # print(f'{datasource}_train/valid/test:',train_cnt, val_cnt, test_cnt)
        # print(f'{datasource}:',np.max([len(x) for x in self.Xs]), np.mean([len(x) for x in self.Xs]))

