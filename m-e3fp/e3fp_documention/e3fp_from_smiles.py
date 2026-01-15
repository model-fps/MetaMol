from e3fp.fingerprint.db import FingerprintDatabase
from e3fp.fingerprint.fprint import Fingerprint
from e3fp.pipeline import fprints_from_smiles
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from glob import glob
import pandas as pd
from python_utilities.parallel import Parallelizer
from e3fp.conformer.util import smiles_to_dict

from e3fp.pipeline import fprints_from_mol, confs_from_smiles
from e3fp.conformer.util import mol_from_smiles
from e3fp.conformer.generate import generate_conformers
from e3fp.fingerprint.fprint import add, mean
import pickle
import json

# # Test_items
# from itertools import islice
# test_items = dict(islice(smiles_dict.items(), 1000))  # 100k,1000
# # "CHEMBL3422092"  O=C(N[C@@H](c1ccccn1)C1CC1)c1ccc2[nH]nc(-c3ccc(N4[C@H]5CC[C@H]4CC(O)C5)cc3)c2c1
# del test_items["CHEMBL3422092"]
# print(len(test_items))        # 999
# no_conf = [
# "CHEMBL2314388",
# "CHEMBL2367481",
# "CHEMBL4798313",
# "CHEMBL4792910",
# "CHEMBL4779093",
# "CHEMBL4790665",
# "CHEMBL4793871",
# "CHEMBL4776867",
# "CHEMBL4791125",
# "CHEMBL4782789",
# "CHEMBL4795488",
# "CHEMBL4781544",
# "CHEMBL4786167",
# "CHEMBL4789325",
# "CHEMBL4793095",
# "CHEMBL4779198",
# "CHEMBL4526745",
# "CHEMBL3608573",
# "CHEMBL3608572",
# "CHEMBL3667139",
# "CHEMBL4792854",
# "CHEMBL3667136",
# "CHEMBL3667138",
# "CHEMBL3667137",
# "CHEMBL4567120",
# "CHEMBL4789297",
# "CHEMBL4748920",
# "CHEMBL4763463",
# "CHEMBL3608571",
# ]  # RuntimeError:


def slice_dict(input_dict, slice_length):

    items = list(input_dict.items())

    return [dict(items[i:i + slice_length]) for i in range(0, len(items), slice_length)]

if __name__ == '__main__':
    data_path = "./smiles_chemblID.csv"
    # read CSV file
    datas = pd.read_csv(data_path, low_memory=False)  # (420626, 2)
    # print(f"datas.shape:{datas.shape},datas:{datas.head(5)}")

    smiles = datas.drop_duplicates(subset=['Smiles'])
    # print(f"smiles.shape:{smiles.shape},smiles:{smiles.head(5)}")      # (233071, 2)

    smiles_dict = smiles.set_index('Molecule ChEMBL ID')['Smiles'].to_dict()
    None_conf = [
        "CHEMBL3422092", "CHEMBL2314388", "CHEMBL2367481", "CHEMBL4798313", "CHEMBL4792910",
        "CHEMBL4779093", "CHEMBL4790665", "CHEMBL4793871", "CHEMBL4776867", "CHEMBL4791125",
        "CHEMBL4782789", "CHEMBL4795488", "CHEMBL4781544", "CHEMBL4786167", "CHEMBL4789325",
        "CHEMBL4793095", "CHEMBL4779198", "CHEMBL4526745", "CHEMBL3608573", "CHEMBL3608572",
        "CHEMBL3667139", "CHEMBL4792854", "CHEMBL3667136", "CHEMBL3667138", "CHEMBL3667137",
        "CHEMBL4567120", "CHEMBL4789297", "CHEMBL4748920", "CHEMBL4763463", "CHEMBL3608571",
        "CHEMBL4073994", "CHEMBL4782991", "CHEMBL4789031", "CHEMBL4790195"
        ]
    # RuntimeError:   CHEMBL3040216
    print(f"len(None_conf) = {len(None_conf)}")  # 34
    for chemblid in None_conf:
        del smiles_dict[chemblid]
    print(len(smiles_dict))  # 233037
    # print(smiles_dict)

    length = 1000
    smiles_items = slice_dict(smiles_dict, length)
    print(f"len(smiles_items):{len(smiles_items)}")  # 234

    # UFFTYPER
    Error_smiles = {}
    Error_smiles['6'] = smiles_items[6]
    Error_smiles['7'] = smiles_items[7]
    Error_smiles['11'] = smiles_items[11]

    # Test1
    exp_items = smiles_items[6]
    # Change

    print(f"len(exp_items) = {len(exp_items)}")   # 1000
    smiles_iter = ((smiles, name) for name, smiles in exp_items.items())
    fprint_params = {'bits': 2048, 'radius_multiplier': 2, 'first': -1, 'rdkit_invariants': True}
    confgen_params = {'max_energy_diff': 20.0, 'first': -1}
    kwargs = {"confgen_params": confgen_params, "fprint_params": fprint_params}
    parallelizer = Parallelizer(parallel_mode="processes")
    fprints_list = parallelizer.run(fprints_from_smiles, smiles_iter, kwargs=kwargs)
    print(f"len(fprints_list):{len(fprints_list)}")    # 元组形式 ([Fingerprins], (smiles,CHEMBLID))
    float_fps_dict = {}
    for fps in fprints_list:
        # print(len(fps[0]))
        float_fps = mean(fps[0])
        # print(len(float_fps))   # 2048
        length = fprint_params.get("bits", 2048)
        float_array = np.zeros(length)
        for index, value in float_fps.counts.items():
            if 0 <= index < length:
                float_array[index] = value
        # print(len(float_array))
        # print(sum(float_array))
        # print(fps[1][0])
        float_fps_dict[fps[1][0]] = float_array
    print(f"len(float_fps_dict):{len(float_fps_dict)}")
    # print(float_fps_dict)

    # Change
    exp_string = f"float_fps_data{6},pkl"
    # print(f"exp_string:{exp_string}")
    # print(f"exp_items:{len(exp_items)}")
    with open(exp_string, 'wb') as file:
        pickle.dump(float_fps_dict, file)


    # Circulation 0-9
    # for i in range(10):
    #     i = i + 6     # 0,1,2,3,4,5done;
    # #    smiles_iter = ((smiles, name) for name, smiles in exp_items.items())
    #     smiles_iter = ((smiles, name) for name, smiles in smiles_items[i].items())
    #     fprint_params = {'bits': 2048, 'radius_multiplier': 2, 'first': -1, 'rdkit_invariants': True}
    #     confgen_params = {'max_energy_diff': 20.0, 'first': -1}
    #     kwargs = {"confgen_params": confgen_params, "fprint_params": fprint_params}
    #     parallelizer = Parallelizer(parallel_mode="processes")
    #
    #     fprints_list = parallelizer.run(fprints_from_smiles, smiles_iter, kwargs=kwargs)
    #     print(len(fprints_list))    # 元组形式 ([Fingerprins], (smiles,CHEMBLID))
    #
    #     float_fps_dict = {}
    #     for fps in fprints_list:
    #
    #         # print(len(fps[0]))
    #         float_fps = mean(fps[0])
    #         # print(len(float_fps))   # 2048
    #         length = fprint_params.get("bits", 2048)
    #         float_array = np.zeros(length)
    #         for index, value in float_fps.counts.items():
    #             if 0 <= index < length:
    #                 float_array[index] = value
    #         # print(len(float_array))
    #         # print(sum(float_array))
    #         # print(fps[1][0])
    #         float_fps_dict[fps[1][0]] = float_array
    #
    #     print(len(float_fps_dict))
    #
    #     exp_string = f"float_fps_data{i},pkl"
    #     # print(f"exp_string:{exp_string}")
    #     # print(f"exp_items:{len(exp_items)}")
    #
    #     with open(exp_string, 'wb') as file:
    #         pickle.dump(float_fps_dict, file)
