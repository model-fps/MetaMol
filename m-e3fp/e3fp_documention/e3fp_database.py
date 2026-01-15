from e3fp.fingerprint.db import FingerprintDatabase
from e3fp.fingerprint.fprint import Fingerprint
# from e3fp.pipeline import fprints_from_smiles
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# db = FingerprintDatabase(fp_type=Fingerprint, name="TestDB")
# print(f"db:{db}")
# on_inds = [np.random.uniform(0, 2**32, size=30) for i in range(5)]
# fps = [Fingerprint(x, bits=2**32) for x in on_inds]
# db.add_fingerprints(fps)
# print(f"db:{db}")
# print(f"db.get_density():{db.get_density()}")   # db.get_density():6.984919309616089e-09
# fold_db = db.fold(1024)
# print(f"fold_db:{fold_db}")
# print(f"fold_db.get_density():{fold_db.get_density()}")  # fold_db.get_density():0.029296875
#
# #数据库可以转换为不同的指纹类型
# from e3fp.fingerprint.fprint import CountFingerprint
# count_db = db.as_type(CountFingerprint)
# print(f"count_db:{count_db}")
# print(f"count_db[0]:{count_db[0]}")
#
# #e3fp.fingerprint.db.concat方法允许高效连接多个数据库
# from e3fp.fingerprint.db import concat
# dbs = []
# for i in range(10):
#     db = FingerprintDatabase(fp_type=Fingerprint)
#     on_inds = [np.random.uniform(0, 1024, size=30) for j in range(5)]
#     fps = [Fingerprint(x, bits=2**32, name="Mol{}".format(i)) for x in on_inds]
#     db.add_fingerprints(fps)
#     dbs.append(db)
# print(f"dbs[0][0]:{dbs[0][0]}")
# print(f"dbs[0]:{print(dbs[0])}")
# merge_db = concat(dbs)
# print(f"merge_db:{merge_db}")

from glob import glob
import pandas as pd
from python_utilities.parallel import Parallelizer
from e3fp.conformer.util import smiles_to_dict

'''
def smiles_generator(*filenames):
    """Parse SMILES file(s) and yield (name, smile).

    Parameters
    ----------
    files : iterable object
        List of files containing smiles. File must contain one smile per
        line, followed by a space and then the molecule name.

    Yields
    ------
    tuple:
        `tuple` of the format (smile, name).
    """
    for filename in filenames:
        with smart_open.open(filename, "r") as f:
            for i, line in enumerate(f):
                values = line.rstrip("\r\n").split()
                if len(values) >= 2:
                    yield tuple(values[:2])
                else:
                    logging.warning(
                        (
                            "Line {:d} of {} has {:d} entries. Expected at least"
                            " 2.".format(i + 1, filename, len(values))
                        ),
                        exc_info=True,
                    )

def smiles_to_dict(smiles_file, unique=False, has_header=False):
    """Read SMILES file to dict."""
    smiles_gen = smiles_generator(smiles_file)
    if has_header:
        header = next(smiles_gen)
        logging.info("Skipping first (header) values: {!r}".format(header))
    if unique:
        used_smiles = set()
        smiles_dict = {}
        for smiles, name in smiles_gen:
            if name not in smiles_dict and smiles not in used_smiles:
                smiles_dict[name] = smiles
                used_smiles.add(smiles)
    else:
        smiles_dict = {name: smiles for smiles, name in smiles_gen}
    return smiles_dict
    
def csv_to_dict(file_path, key_column, value_column):
    """将 CSV 文件中的某一列作为字典的键，另一列作为字典的值。

    参数
    ----------
    file_path : str
        CSV 文件的路径。
    key_column : str
        键所在的列名。
    value_column : str
        值所在的列名。

    返回
    -------
    dict
        以指定列为键，另一列为值构造的字典。
    """
    # 读取 CSV 文件
    df = pd.read_csv(file_path)
    
    # 将指定列转换为字典
    result_dict = df.set_index(key_column)[value_column].to_dict()
    
    return result_dict

# 示例用法
file_path = 'example.csv'  # 替换为你的 CSV 文件路径
key_column = 'Column1'  # 假设第一列的列名为 'Column1'
value_column = 'Column2'  # 假设第二列的列名为 'Column2'
result = csv_to_dict(file_path, key_column, value_column)
print(result)
'''
# data_path = "./processed_chembl_data1.csv"
# # read CSV file
# datas = pd.read_csv(data_path, low_memory=False)     # (420626, 12)
# print(f"datas.shape:{datas.shape},datas:{datas.head(5)}")
# # save smiles chemblID column
# selected_columns = ['Molecule ChEMBL ID', 'Smiles']  # 'Molecule ChEMBL ID', 'Smiles'
# new_data = datas[selected_columns]
# new_data.to_csv('smiles_chemblID.csv', index=False)

data_path = "./smiles_chemblID.csv"
# read CSV file
datas = pd.read_csv(data_path, low_memory=False)     # (420626, 2)
# print(f"datas.shape:{datas.shape},datas:{datas.head(5)}")

smiles = datas.drop_duplicates(subset=['Smiles'])
# print(f"smiles.shape:{smiles.shape},smiles:{smiles.head(5)}")      # (233071, 2)

smiles_dict = smiles.set_index('Molecule ChEMBL ID')['Smiles'].to_dict()
print(len(smiles_dict))     #  233071
# print(smiles_dict)

from itertools import islice
test_items = dict(islice(smiles_dict.items(), 5))
# test_items['CHEMBL3422092'] = 'O=C(N[C@@H](c1ccccn1)C1CC1)c1ccc2[nH]nc(-c3ccc(N4[C@H]5CC[C@H]4CC(O)C5)cc3)c2c1'
print(len(test_items))

smiles_iter = ((smiles, name) for name, smiles in test_items.items())
# smiles_iter = ((smiles, name) for name, smiles in smiles_dict.items())
fprint_params = {'bits': 2048, 'radius_multiplier': 2, 'first': -1, 'rdkit_invariants': True}
confgen_params = {'max_energy_diff': 20.0, 'first': -1}
kwargs = {"confgen_params": confgen_params, "fprint_params": fprint_params}
parallelizer = Parallelizer(parallel_mode="processes")

from e3fp.pipeline import fprints_from_mol, confs_from_smiles
from e3fp.conformer.util import mol_from_smiles
from e3fp.conformer.generate import generate_conformers
from e3fp.fingerprint.fprint import add, mean

# def confs_from_smiles(smiles, name, confgen_params={}, save=False):
#     """Generate conformations of molecule from SMILES string."""
#     mol = mol_from_smiles(smiles, name)
#     confgen_result = generate_conformers(
#         mol, name, save=save, **confgen_params
#     )
#     if confgen_result != False:
#         mol = confgen_result[0]
#     else:
#         mol = smiles
#         print(smiles)
#     num_conformers = mol.GetNumConformers()
#     print(f"num_conformers: {num_conformers}")
#     return mol

def fprints_from_smiles(
    smiles, name, confgen_params={}, fprint_params={}, save=False
):
    """Generate conformers and fingerprints from a SMILES string."""
    if save is False and "first" not in confgen_params:
        confgen_params["first"] = fprint_params.get("first", -1)
    mol = confs_from_smiles(
        smiles, name, confgen_params=confgen_params, save=save
    )
    if isinstance(mol, str):
        fprints_list = []
        smiles = mol
        mol = Chem.MolFromSmiles(smiles)
        mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fprints_list.append(mfp)
    else:
        fprints_list = fprints_from_mol(
            mol, fprint_params=fprint_params, save=save
        )
    return fprints_list

# def e3fps_from_smiles(smiles, confgen_params={}, fprint_params={}):
#     fprints = fprints_from_smiles(smiles, str(smiles), confgen_params=confgen_params, fprint_params=fprint_params)
#     float_fps = mean(fprints)
#     float_array = get_float_array(2048, float_fps.counts)
#     # print(f"len(float_array):{len(float_array)}")  # 2048
#     # print(f"float_array:{float_array}")
#     non_zero_bit = {index: value for index, value in enumerate(float_array) if value != 0}
#     # print(f"non_zero_bit:{non_zero_bit}")
#     # print(f"len(non_zero_bit):{len(non_zero_bit)}")
#     return float_array
# def get_float_array(length, dict):
#     float_array = np.zeros(length)
#     for index, value in dict.items():
#         if 0 <= index < length:  # 确保索引在数组范围内
#             float_array[index] = value
#     return float_array

def fps_from_smiles(smiles, name, confgen_params={}, fprint_params={}, save=False):
    """Generate conformers and fingerprints from a SMILES string."""
    if save is False and "first" not in confgen_params:
        confgen_params["first"] = fprint_params.get("first", -1)
    mol = confs_from_smiles(
        smiles, name, confgen_params=confgen_params, save=save
    )
    fprints_list = fprints_from_mol(
        mol, fprint_params=fprint_params, save=save
    )

    # float_fps = mean(fprints_list)
    # # float_array = get_float_array(2048, float_fps.counts)
    # length = fprint_params.get("bits", 2048)
    # float_array = np.zeros(length)
    # for index, value in float_fps.counts.items():
    #     if 0 <= index < length:
    #         float_array[index] = value

    return fprints_list

fprints_list = parallelizer.run(fprints_from_smiles, smiles_iter, kwargs=kwargs)
print(len(fprints_list))    # 元组形式 ([Fingerprins], (smiles,CHEMBLID))
# CHEMBL3422092  O=C(N[C@@H](c1ccccn1)C1CC1)c1ccc2[nH]nc(-c3ccc(N4[C@H]5CC[C@H]4CC(O)C5)cc3)c2c1

# for fps in fprints_list:
#
#     print(len(fps[0]))
#     float_fps = mean(fps[0])
#     # print(len(float_fps))   # 2048
#     length = fprint_params.get("bits", 2048)
#     float_array = np.zeros(length)
#     for index, value in float_fps.counts.items():
#         if 0 <= index < length:
#             float_array[index] = value
#     print(len(float_array))
#     print(sum(float_array))
#     print(fps[1])

smiles = 'O=C(N[C@@H](c1ccccn1)C1CC1)c1ccc2[nH]nc(-c3ccc(N4[C@H]5CC[C@H]4CC(O)C5)cc3)c2c1'
name = 'CHEMBL3422092'
res = confs_from_smiles(smiles, name, confgen_params=confgen_params, save=False)
print(res)
