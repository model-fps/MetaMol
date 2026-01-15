import math
import json
import os
import numpy as np
from torch.utils.data import Dataset, sampler, DataLoader
from typing import Dict, List, Set, Tuple, Union
import torch
import random
from collections import defaultdict
import numpy
import math, os
from tqdm import tqdm
import random
import csv
from collections import OrderedDict
import json, pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import SaltRemover
import h5py

def check_smiles_validity(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 'smiles_unvaild'
        HA_num = mol.GetNumHeavyAtoms()
        if HA_num <= 2:
            return 'smiles_unvaild'
        return smiles
    except:
        return 'smiles_unvaild'

def gen_standardize_smiles(smiles, kekule=False, random=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 'smiles_unvaild'
        desalt = SaltRemover.SaltRemover() ## defnData="[Cl,Br,I,Fe,Na,K,Ca,Mg,Ni,Zn]"
        mol = desalt.StripMol(mol)
        if mol is None:
            return 'smiles_unvaild'
        HA_num = mol.GetNumHeavyAtoms()
        if HA_num <= 2:
            return 'smiles_unvaild'
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekule, doRandom=random, isomericSmiles=True)
        return smiles
    except:
        smiles = 'smiles_unvaild'


def get_chembl_targets():

    save_path = "D:/Author/code/model/data/chembl/chembl_fps.pkl"
    smiles_e3fps = pickle.load(open(save_path, "rb"))

    datas = csv.reader(open("D:/Author/code/model/data/chembl/processed_chembl_target_assay.csv", "r"),
                       delimiter=',')
    header = next(datas)
    target_id_dicts = {}

    for line in tqdm(datas, desc="Processing chembl"):
        unit = line[7]   # nM
        if unit=="%":
            continue
        assay_id = line[11]
        target_chembl_id = line[15]
        cancer_targets = ['CHEMBL1808', 'CHEMBL284', 'CHEMBL203' ]
        if target_chembl_id in cancer_targets:
            continue
        target_id = "{}_{}".format(line[15], line[11]).replace("/", "_")
        if target_id not in target_id_dicts:
            target_id_dicts[target_id] = []

        smiles = line[13]
        mean_e3fp = smiles_e3fps.get(smiles, None)
        # if the smiles is not found then skip
        if mean_e3fp is None:
            continue

        assay_type = line[9]
        std_type = line[8]
        unit = line[7]
        std_rel = line[5]

        if std_rel != "=" and std_rel != "'='":
            continue
        pic50_exp = -math.log10(float(line[6]))
        ligand_info = {
            "std_type": std_type,
            "smiles": smiles,
            "mean_e3fp": mean_e3fp,
            "pic50_exp": pic50_exp,
            "chembl_assay_type": assay_type,
            "unit": unit,
            "domain": "chembl"
        }
        target_id_dicts[target_id].append(ligand_info)

    split_path = f'D:/Author/code/model/data/chembl/chembl_valid_test_split.json'
    split_chembl_val_test = json.load(open(split_path, "r"))
    train_targets = split_chembl_val_test['train']
    # print('train_targets', len(train_targets))

    target_id_dicts_new = {}
    for target_id, ligands in target_id_dicts.items():
        if len(ligands) < 20 or len(ligands) > 30000:
            continue

        if target_id in train_targets:
            if len(ligands) > 64:
                target_set = len(ligands) // 64
                target_ligand = len(ligands) % 64
                for i in range(target_set):
                    first_idx = i * 64
                    end_idx = first_idx + 64
                    ligands_set = ligands[first_idx:end_idx]
                    target_id_dicts_new[f"{target_id}_{i + 1}"] = ligands_set
                if target_ligand > 0:
                    if target_ligand < 20:
                        last_target_id = f"{target_id}_{target_set}"
                        target_id_dicts_new[last_target_id].extend(ligands[-target_ligand:])
                    else:
                        new_target_id = f"{target_id}_{target_set + 1}"
                        target_id_dicts_new[new_target_id] = ligands[-target_ligand:]
            else:
                target_id_dicts_new[target_id] = ligands
        else:
            target_id_dicts_new[target_id] = ligands

    return {"ligand_sets": target_id_dicts_new, "targets": list(target_id_dicts_new.keys())}


def get_bdb_targets():
    data_dir = f"D:/Author/code/model/data/bdb/BDB_target"
    ligand_sets = {}

    save_path = "D:/Author/code/model/data/bdb/bdb_fps.pkl"
    smiles_e3fps = pickle.load(open(save_path, "rb"))

    for target_name in tqdm(list(os.listdir(data_dir))):
        cancer_targets = ['Angiotensin-converting enzyme', 'Dipeptidyl peptidase IV', 'Epidermal growth factor receptor']
        if target_name in cancer_targets:
            continue
        for assay_file in os.listdir(os.path.join(data_dir, target_name)):
            target_assay_name = target_name + "/" + assay_file
            # target_assay_name = target_name + "/" + assay_file[-5]
            entry_assay = "_".join(assay_file.split("_")[:2])
            affi_idx = int(assay_file[-5])
            ligands = []
            affix = []
            file_lines = list(open(os.path.join(data_dir, target_name, assay_file), "r", encoding="utf-8").readlines())
            for i, line in enumerate(file_lines):
                line = line.strip().split("\t")
                pic50_exp = line[8+affi_idx].strip()
                if pic50_exp.startswith(">") or pic50_exp.startswith("<"):
                    continue
                try:
                    pic50_exp = -math.log10(float(pic50_exp))
                except:
                    print("error ic50:", pic50_exp)
                    continue
                smiles = line[1]
                mean_e3fp = smiles_e3fps.get(smiles, None)
                if mean_e3fp is None:
                    continue
                affix.append(pic50_exp)
                ligand_info = {
                    "smiles": smiles,
                    "mean_e3fp": mean_e3fp,
                    "pic50_exp": pic50_exp,
                    "domain": "bdb"
                }
                ligands.append(ligand_info)
            if len(ligands) <= 20:
                continue
            # means.append(np.mean([x["pic50_exp"] for x in ligands]))
            ligand_sets[target_assay_name] = ligands

    split_path = f'D:/Author/code/model/data/bdb/bdb_valid_test_split.json'
    split_bdb_val_test = json.load(open(split_path, "r"))
    train_targets = split_bdb_val_test['train']
    target_id_dicts_new = {}
    for target_id, ligands in ligand_sets.items():
        if len(ligands) < 20 or len(ligands) > 30000:
            continue
        if target_id in train_targets:
            if len(ligands) > 64:
                target_set = len(ligands) // 64
                target_ligand = len(ligands) % 64
                for i in range(target_set):
                    first_idx = i * 64
                    end_idx = first_idx + 64
                    ligands_set = ligands[first_idx:end_idx]
                    target_id_dicts_new[f"{target_id}_{i + 1}"] = ligands_set
                if target_ligand > 0:
                    if target_ligand < 20:
                        last_target_id = f"{target_id}_{target_set}"
                        target_id_dicts_new[last_target_id].extend(ligands[-target_ligand:])
                    else:
                        new_target_id = f"{target_id}_{target_set + 1}"
                        target_id_dicts_new[new_target_id] = ligands[-target_ligand:]
            else:
                target_id_dicts_new[target_id] = ligands
        else:
            target_id_dicts_new[target_id] = ligands
    return {"ligand_sets": target_id_dicts_new,
            "targets": list(target_id_dicts_new.keys())}

def preprocess_targets(in_data):
    lines = in_data
    x_tmp = []
    smiles_list = []
    activity_list = []

    if lines is None:
        return None

    if len(lines) > 30000:
        return None

    for line in lines:
        smiles = line["smiles"]
        fp_numpy = np.array(line["mean_e3fp"])

        pic50_exp = line["pic50_exp"]
        activity_list.append(pic50_exp)
        x_tmp.append(fp_numpy)
        smiles_list.append(smiles)

    x_tmp = np.array(x_tmp).astype(np.float32)
    affis = np.array(activity_list).astype(np.float32)
    if len(x_tmp) < 20 and lines[0].get("domain", "none") in ['chembl', 'bdb']:
        return None
    return x_tmp, affis, smiles_list


class BaseMetaDataset(Dataset):
    def __init__(self, args, exp_string):
        self.args = args
        self.current_set_name = "train"
        self.exp_string = exp_string

        self.init_seed = {"train": args.train_seed, "val": args.val_seed, 'test': args.test_seed,
                          'train_weight': args.train_seed}
        self.batch_size = args.meta_batch_size

        self.train_index = 0
        self.val_index = 0
        self.test_index = 0
        self.split_all = []

        self.current_epoch = 0
        self.load_dataset()

    def load_dataset(self):
        raise NotImplementedError

    def get_split(self, X_in, y_in, is_test=False, ref_num=None,  rand_seed=None, smiles=None):
        def data_split(data_len, ref_num_, rng_):
            if not is_test:
                min_num = math.log10(max(10, int(0.3 * data_len)))
                max_num = math.log10(int(0.85 * data_len))
                # todo:for few-shot setting
                ref_num_ = random.uniform(min_num, max_num)
                ref_num_ = math.floor(10 ** ref_num_)
            split = [1] * ref_num_ + [0] * (data_len - ref_num_)
            rng_.shuffle(split)
            return np.array(split)

        def data_split_bysim(Xs, ref_num_, rng_, sim_cut_):
            def get_sim_matrix(a, b):
                a_bool = (a > 0.).float()
                b_bool = (b > 0.).float()
                and_res = torch.mm(a_bool, b_bool.transpose(0, 1))
                or_res = a.shape[-1] - torch.mm((1. - a_bool), (1. - b_bool).transpose(0, 1))
                sim = and_res / or_res
                return sim

            Xs_torch = torch.tensor(Xs).cuda()
            sim_matrix = get_sim_matrix(Xs_torch, Xs_torch).cpu().numpy() - np.eye(len(Xs))
            split = [1] * ref_num_ + [0] * (len(Xs) - ref_num_)
            rng_.shuffle(split)
            ref_index = [i for i, t in enumerate(split) if t == 1]

            split = []
            for i in range(len(y)):
                if i in ref_index:
                    split.append(1)
                else:
                    max_sim = np.max(sim_matrix[i][ref_index])
                    if max_sim >= sim_cut_:
                        split.append(-1)
                    else:
                        split.append(0)
            return np.array(split)

        rng = np.random.RandomState(seed=rand_seed)
        # 64    arg.datasource
        if len(X_in) > 64 and not is_test:
            subset_num = 64
            raw_data_len = len(X_in)
            select_idx = [1] * subset_num + [0] * (raw_data_len - subset_num)
            rng.shuffle(select_idx)
            select_idx = np.nonzero(np.array(select_idx))
            X, y = X_in[select_idx], y_in[select_idx]
        else:
            X, y = X_in, y_in

        ref_num = self.args.test_ref_num
        if ref_num <= 1:
            ref_num = ref_num * len(X)
        ref_num = int(ref_num)
        if self.args.similarity_cut < 1.:
            assert is_test
            split = data_split_bysim(X, ref_num, rng, self.args.similarity_cut)
            X = np.array([t for i, t in enumerate(X) if split[i] != -1])
            y = [t for i, t in enumerate(y) if split[i] != -1]
            split = np.array([t for i, t in enumerate(split) if t != -1], dtype=np.int)
        else:
            ''' split random'''
            split = data_split(len(X), ref_num, rng)

        return [X, y, split]

    def get_set(self, current_set_name, idx):
        datas = []
        targets = []
        si_list = []
        ligand_nums = []
        ligands_all = []

        if current_set_name == 'train':
            si_list = self.train_indices[idx * self.batch_size: (idx + 1) * self.batch_size]
            ret_weight = [1. for _ in si_list]
        elif current_set_name == 'val':
            si_list = [self.val_indices[idx]]
            ret_weight = [1.]
        elif current_set_name == 'test':
            si_list = [self.test_indices[idx]]
            ret_weight = [1.]
        elif current_set_name == 'train_weight':
            if self.idxes is not None:
                si_list = self.idxes[idx * self.weighted_batch: (idx + 1) * self.weighted_batch]
                ret_weight = self.train_weight[idx * self.weighted_batch: (idx + 1) * self.weighted_batch]
            else:
                si_list = [self.train_indices[idx]]
                ret_weight = [1.]

        for si in si_list:
            ligand_nums.append(len(self.Xs[si]))
            target_name = self.targets[si]
            datas.append(self.get_split(self.Xs[si], self.ys[si],
                                        is_test=current_set_name in ['test', 'val', 'train_weight'],
                                        rand_seed=self.init_seed[current_set_name] + si + self.current_epoch,
                                        smiles=self.smiles_all[si]
                                        ))
            targets.append(target_name)
            ligands_all.append(self.smiles_all[si])

        return tuple([[torch.tensor(x[i]) for x in datas] for i in range(0, 3)] +
                     [targets, ret_weight, ligands_all])

    def __len__(self):
        if self.current_set_name == "train":
            total_samples = self.data_length[self.current_set_name] // self.args.meta_batch_size
        elif self.current_set_name == "train_weight":
            if self.idxes is not None:
                total_samples = len(self.idxes) // self.weighted_batch
            else:
                total_samples = self.data_length["train_weight"] // self.weighted_batch
        else:
            total_samples = self.data_length[self.current_set_name]
        return total_samples

    def length(self, set_name):
        self.switch_set(set_name=set_name)
        return len(self)

    def set_train_weight(self, train_weight=None, idxes=None, weighted_batch=1):
        self.train_weight = train_weight
        self.idxes = idxes
        self.weighted_batch = weighted_batch

    def switch_set(self, set_name, current_epoch=0):
        self.current_set_name = set_name
        self.current_epoch = current_epoch
        if set_name == "train":
            rng = np.random.RandomState(seed=self.init_seed["train"] + current_epoch)
            rng.shuffle(self.train_indices)

    def __getitem__(self, idx):
        return self.get_set(self.current_set_name, idx=idx)


def my_collate_fn(batch):
    batch = batch[0]
    return batch


class SystemDataLoader(object):
    def __init__(self, args, MetaDataset, current_epoch=0, exp_string=None):
        """
        Initializes a meta learning system dataloader. The data loader uses the Pytorch DataLoader class to parallelize
        batch sampling and preprocessing.
        :param args: An arguments NamedTuple containing all the required arguments.
        :param current_epoch: Current iter of experiment. Is used to make sure the data loader continues where it left
        of previously.
        """
        self.args = args
        self.batch_size = args.meta_batch_size
        self.total_train_epochs = 0
        self.dataset = MetaDataset(args, exp_string=exp_string)
        self.full_data_length = self.dataset.data_length
        self.continue_from_epoch(current_epoch=current_epoch)

    def get_train_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        return DataLoader(self.dataset, batch_size=1, num_workers=2, shuffle=False, drop_last=True,
                          collate_fn=my_collate_fn)

    def get_dataloader(self):
        """
        Returns a data loader with the correct set (train, val or test), continuing from the current iter.
        :return:
        """
        return DataLoader(self.dataset, batch_size=1, shuffle=False, drop_last=True, collate_fn=my_collate_fn)

    def continue_from_epoch(self, current_epoch):
        """
        Makes sure the data provider is aware of where we are in terms of training iterations in the experiment.
        :param current_epoch:
        """
        self.total_train_epochs += current_epoch

    def get_train_batches_weighted(self, weights=None, idxes=None, weighted_batch=1):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        self.dataset.switch_set(set_name="train_weight", current_epoch=self.total_train_epochs)
        self.dataset.set_train_weight(weights, idxes, weighted_batch=weighted_batch)
        self.total_train_epochs += 1
        return self.get_dataloader()

    def get_train_batches(self, total_batches=-1):
        """
        Returns a training batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param augment_images: Whether we want the images to be augmented.
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length["train"] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name="train", current_epoch=self.total_train_epochs)
        self.total_train_epochs += self.batch_size
        return self.get_train_dataloader()

    def get_val_batches(self, total_batches=-1, repeat_cnt=0):
        """
        Returns a validation batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param repeat_cnt:
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['val'] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name="val", current_epoch=repeat_cnt)
        return self.get_dataloader()

    def get_test_batches(self, total_batches=-1, repeat_cnt=0):
        """
        Returns a testing batches data_loader
        :param total_batches: The number of batches we want the data loader to sample
        :param repeat_cnt:
        """
        if total_batches == -1:
            self.dataset.data_length = self.full_data_length
        else:
            self.dataset.data_length['test'] = total_batches  # * self.dataset.batch_size
        self.dataset.switch_set(set_name='test', current_epoch=repeat_cnt)
        return self.get_dataloader()
