import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import smart_open

# Generating Fingerprints from sdf
from e3fp.pipeline import fprints_from_sdf, fprints_from_mol
from e3fp.conformer.util import mol_from_sdf, MolItemName, mol_to_standardised_mol, add_conformer_energies_to_mol
from e3fp.fingerprint.metrics.fprint_metrics import tanimoto, dice, cosine, pearson
from e3fp.fingerprint.fprint import Fingerprint

import os
import logging
import smart_open
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from itertools import combinations
from multiprocessing import Pool, cpu_count

CONF_ENERGY_PROPNAME = "Energy"

def mols_from_sdf_molecules(sdf_file, conf_num=None, mode="rb"):
    """Read SDF file into an RDKit `Mol` object.

    Parameters
    ----------
    sdf_file : str
        Path to a SDF file
    # index_file:str
    #     Path to a CSV file
    conf_num : int or None, optional
        Maximum number of conformers to read from file. Defaults to all.
    mode : str (default 'rb')
        Mode with which to open file

    Returns
    -------
    list of RDKit Mol : List of `Mol` objects with each molecule in SDF file containing multiple conformations

    """
    mol_index_dict = {}
    skipped_molecules = []

    with smart_open.open(sdf_file, mode) as f:
        supplier = rdkit.Chem.ForwardSDMolSupplier(f)

        for i, mol in enumerate(supplier):

            if conf_num is not None and i == conf_num:
                break
            if mol is None:
                skipped_molecules.append(i)
                continue
            else:
                try:
                    mol_index = mol.GetProp("Name")
                    mol_number = mol.GetProp("MolNumber")
                    print(f"processing mol {i}, MolNumber：{mol_number}")
                except Exception as e:
                    mol_index = rdkit.Chem.MolToSmiles(mol)
                    print(f"Error reading {mol_index}")
                    print(f"Error processing molecule at index {i}:{e}")
                new_mol = mol

                if mol_index not in mol_index_dict:
                    mol_index_dict[mol_index] = rdkit.Chem.Mol(new_mol)
                    mol_index_dict[mol_index].RemoveAllConformers()

                conf = new_mol.GetConformers()[0]
                mol_index_dict[mol_index].AddConformer(conf, assignId=True)

    mols = list(mol_index_dict.values())
    mol_name = list(mol_index_dict.keys())
    print(f"mols:{len(mols)},mol_name:{len(mol_name)}")

    if skipped_molecules:
        print(f"Skipping {len(skipped_molecules)} molecules")
    # for mol, name in zip(mols, mol_index_dict.keys()):
    #     mol.SetProp("Smiles", name)
    #     print("name",mol.GetProp("Smiles"))

    return mols, mol_name

# def mols_from_sdf_molecules(sdf_file, index_file, conf_num=None, mode="rb"):
#     """Read SDF file into an RDKit `Mol` object.
#
#     Parameters
#     ----------
#     sdf_file : str
#         Path to a SDF file
#     index_file:str
#         Path to a CSV file
#     conf_num : int or None, optional
#         Maximum number of conformers to read from file. Defaults to all.
#     mode : str (default 'rb')
#         Mode with which to open file
#
#     Returns
#     -------
#     list of RDKit Mol : List of `Mol` objects with each molecule in SDF file containing multiple conformations
#
#     """
#     mol_index_dict = {}
#
#     # df_smiles = pd.read_csv(csv_file)
#     # smiles_data = df_smiles['Smiles'].tolist()
#     # name_list = smiles_data
#
#     df_index = pd.read_csv(index_file)
#     index_list = df_index['Name'].tolist()
#
#     with smart_open.open(sdf_file, mode) as f:
#         supplier = rdkit.Chem.ForwardSDMolSupplier(f)
#
#         for i, (mol, index) in enumerate(zip(supplier, index_list)):
#
#             if conf_num is not None and i == conf_num:
#                 break
#
#             if mol is not None:
#                 mol_index = index
#                 new_mol = mol
#
#                 if mol_index not in mol_index_dict:
#                     mol_index_dict[mol_index] = rdkit.Chem.Mol(new_mol)
#                     mol_index_dict[mol_index].RemoveAllConformers()
#
#                 conf = new_mol.GetConformers()[0]
#                 mol_index_dict[mol_index].AddConformer(conf, assignId=True)
#
#     mols = list(mol_index_dict.values())
#
#     # for mol, name in zip(mols, name_list):
#     #     mol.SetProp("task_name", name)
#     #     #print(mol.GetProp("task_name"))
#
#     return mols

# def mols_from_sdf_molecules(sdf_file, csv_file, index_file, conf_num=None, mode="rb"):
#     """Read SDF file into an RDKit `Mol` object.
#
#     Parameters
#     ----------
#     sdf_file : str
#         Path to a SDF file
#     index_file:str
#         Path to a CSV file
#     conf_num : int or None, optional
#         Maximum number of conformers to read from file. Defaults to all.
#     mode : str (default 'rb')
#         Mode with which to open file
#
#     Returns
#     -------
#     list of RDKit Mol : List of `Mol` objects with each molecule in SDF file containing multiple conformations
#
#     """
#     mol_index_dict = {}
#
#     df_smiles = pd.read_csv(csv_file)
#     smiles_data = df_smiles['Smiles'].tolist()
#     name_list = smiles_data
#
#     df_index = pd.read_csv(index_file)
#     index_list = df_index['Name'].tolist()
#
#     with smart_open.open(sdf_file, mode) as f:
#         supplier = rdkit.Chem.ForwardSDMolSupplier(f)
#
#         for i, (mol, index) in enumerate(zip(supplier, index_list)):
#
#             if conf_num is not None and i == conf_num:
#                 break
#
#             if mol is not None:
#                 mol_index = index
#                 new_mol = mol
#
#                 if mol_index not in mol_index_dict:
#                     mol_index_dict[mol_index] = rdkit.Chem.Mol(new_mol)
#                     mol_index_dict[mol_index].RemoveAllConformers()
#
#                 conf = new_mol.GetConformers()[0]
#                 mol_index_dict[mol_index].AddConformer(conf, assignId=True)
#
#     mols = list(mol_index_dict.values())
#
#     for mol, name in zip(mols, name_list):
#         mol.SetProp("task_name", name)
#         #print(mol.GetProp("task_name"))
#
#     return mols

def mols_from_sdf_molecule(sdf_file, conf_num=None, standardise=False, mode="rb"):
    """Read SDF file into an RDKit `Mol` object.

    Parameters
    ----------
    sdf_file : str
        Path to an SDF file
    conf_num : int or None, optional
        Maximum number of conformers to read from file. Defaults to all.
    standardise : bool (default False)
        Clean mol through standardisation
    mode : str (default 'rb')
        Mode with which to open file

    Returns
    -------
    list of RDKit Mol : List of `Mol` objects with each molecule in SDF file
                        as individual conformers when SMILES are the same.
    """
    mol_dict = {}
    conf_energies = []
    skipped_molecules = []       # store skipped molecule information
    with smart_open.open(sdf_file, mode) as f:
        supplier = rdkit.Chem.ForwardSDMolSupplier(f)
        i = 0
        while True:
            if conf_num is not None and i == conf_num:
                break
            try:
                new_mol = next(supplier)
                if new_mol is None:               # Skip invalid molecules
                    skipped_molecules.append(i)   #   Record skipped molecule index
                    continue
            except StopIteration:
                logging.debug(
                    "Read {:d} conformers from {}.".format(i, sdf_file)
                )
                break

            try:
                smiles = Chem.MolToSmiles(new_mol)
            except Exception as e:
                skipped_molecules.append(i)         # Record skipped molecule index
                print(f"Error processing molecule at index {i}: {e}")
                continue

            if smiles not in mol_dict:
                mol_dict[smiles] = rdkit.Chem.Mol(new_mol)
                mol_dict[smiles].RemoveAllConformers()

            if new_mol.HasProp(CONF_ENERGY_PROPNAME):
                conf_energies.append(
                    float(new_mol.GetProp(CONF_ENERGY_PROPNAME))
                )

            conf = new_mol.GetConformers()[0]
            mol_dict[smiles].AddConformer(conf, assignId=True)
            i += 1

    mols = list(mol_dict.values())

    for mol in mols:
        if standardise:
            mol = mol_to_standardised_mol(mol)

        try:
            mol.GetProp("_Name")
        except KeyError:
            name = os.path.basename(sdf_file).split(".sdf")[0]
            mol.SetProp("_Name", name)

        if len(conf_energies) > 0:
            add_conformer_energies_to_mol(mol, conf_energies)
            mol.ClearProp(CONF_ENERGY_PROPNAME)

    # Print skipped error molecule information
    if skipped_molecules:
        print("Skipped molecules at indices:", skipped_molecules)

    return mols


def mols_from_sdf_csv(sdf_file, csv_file, conf_num=None, mode="rb"):
    """Read SDF file into an RDKit `Mol` object.

    Parameters
    ----------
    sdf_file : str
        Path to a SDF file
    csv_file : str
        Path to a CSV file
    conf_num : int or None, optional
        Maximum number of conformers to read from file. Defaults to all.
    mode : str (default 'rb')
        Mode with which to open file

    Returns
    -------
    list of RDKit Mol : List of `Mol` objects with each molecule in SDF file containing multiple conformations

    """
    mol_dict = {}

    df = pd.read_csv(csv_file)
    smiles_data = df['Smiles'].tolist()
    name_list = smiles_data

    with smart_open.open(sdf_file, mode) as f:
        supplier = rdkit.Chem.ForwardSDMolSupplier(f)
        i = 0
        while True:
            if conf_num is not None and i == conf_num:
                break

            try:
                new_mol = next(supplier)
            except StopIteration:
                logging.debug(
                    "Read {:d} conformers from {}.".format(i, sdf_file)
                )
                break

            smiles = Chem.MolToSmiles(new_mol, isomericSmiles=True)  # 使用立体化学

            if smiles not in mol_dict:
                mol_dict[smiles] = rdkit.Chem.Mol(new_mol)
                mol_dict[smiles].RemoveAllConformers()

            conf = new_mol.GetConformers()[0]
            mol_dict[smiles].AddConformer(conf, assignId=True)
            i += 1

    mols = list(mol_dict.values())

    for mol, name in zip(mols, name_list):
        mol.SetProp("task_name", name)
        # print(mol.GetProp("task_name"))

    return mols

