from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from e3fp_documention.e3fp_mols_from_sdf import mols_from_sdf_molecules
from e3fp_documention.e3fp_generate import e3fps_from_mol
from e3fp.pipeline import fprints_from_mol
from e3fp.fingerprint.fprint import add, mean
from glob import glob
from python_utilities.parallel import Parallelizer
import numpy as np
import pickle
from multiprocessing import Pool, cpu_count
import tqdm
import smart_open


def process_molecule(mol, mol_name, fprint_params):
    # smiles = Chem.MolToSmiles(mol)
    mol.SetProp("Index_name", mol_name)
    smiles = mol.GetProp("Index_name")
    # print(f"Get_mol_name:{mol.GetProp("Index_name")}")
    n_conf = mol.GetNumConformers()
    conf_dict = {}
    conf_dict[smiles] = n_conf
    fprints = fprints_from_mol(mol, fprint_params=fprint_params)
    mean_fps = mean(fprints)
    length = fprint_params.get("bits", 2048)
    float_array = np.zeros(length)
    for index, value in mean_fps.counts.items():
        if 0 <= index < length:
            float_array[index] = value

    return conf_dict, float_array

if __name__ == "__main__":

    sdf_file = "./data/different_smiles_confs.sd"

    mols_list, mols_name = mols_from_sdf_molecules(sdf_file, mode="rb")
    fprint_params = {'bits': 2048, 'first': -1, 'radius_multiplier': 1.5, 'rdkit_invariants': True}

    n_confs = []
    limited_mols = {}
    limited_confs = {}
    float_fps_dict = {}
    # processes = cpu_count()
    with Pool(processes = cpu_count()) as pool:
        results = pool.starmap(process_molecule, tqdm.tqdm([(mol, mol_name, fprint_params) for mol, mol_name in zip(mols_list, mols_name)]))

    for conf_dict, float_array in results:
        smiles = list(conf_dict.keys())[0]
        n_conf = list(conf_dict.values())[0]
        n_confs.append(n_conf)
        # Get Conf_num < 10
        if n_conf < 10:
            limited_mols[smiles] = float_array
            limited_confs[smiles] = n_conf
        else:
            float_fps_dict[smiles] = float_array


    float_fps_file = "./data/float_fps.pkl"
    with open(float_fps_file, 'wb') as file:
        pickle.dump(float_fps_dict, file)

