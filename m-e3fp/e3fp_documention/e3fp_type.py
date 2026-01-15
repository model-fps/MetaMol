from e3fp.fingerprint.fprint import Fingerprint, CountFingerprint
import numpy as np

# 用随机的 “on ”索引创建一个位指纹
bits = 2**32
indices = np.sort(np.random.randint(0, bits, 30, dtype=np.int64))
print(f"indice:{indices}")

fp1 = Fingerprint(indices, bits=bits, level=0)
print(f"fp1:{fp1}")
# 指纹非常稀疏
print(f"fp1.bit_count:{fp1.bit_count}")   # fp1.bit_count:30
print(f"fp1.density:{fp1.density}")       # fp1.density:6.984919309616089e-09

#因此，我们可以通过对稀疏向量的一半进行一系列按位“OR”运算来“折叠”指纹，直到它达到指定的长度，并且位冲突最小。
fp_folded = fp1.fold(1024)
print(f"fp_folded:{fp_folded}")
print(f"fp_folded.bit_count:{fp_folded.bit_count}")  # fp_folded.bit_count:30
print(f"fp_folded.density:{fp_folded.density}")      # fp_folded.density:0.029296875

# 还可以通过提供将具有非零计数的索引与计数相匹配的字典来创建CountFingerprint。
indices2 = np.sort(np.random.randint(0, bits, 60, dtype=np.int64))
counts = dict(zip(indices2, np.random.randint(1, 10, indices2.size)))
print(f"counts:{counts}")
cfp1 = CountFingerprint(counts=counts, bits=bits, level=0)
print(f"cfp1:{cfp1}")

#与折叠位指纹不同，默认情况下，折叠计数指纹对冲突计数执行“SUM”运算。
print(f"cfp1.bit_count:{cfp1.bit_count}")    # cfp1.bit_count:60
cfp_folded = cfp1.fold(1024)
print(f"cfp_folded:{cfp_folded}")
print(f"cfp_folded.bit_count:{cfp_folded.bit_count}")  #cfp_folded.bit_count:58

#相互转换指纹很简单。
cfp_folded2 = CountFingerprint.from_fingerprint(fp_folded)
print(f"cfp_folded2:{cfp_folded2}")
print(f"cfp_folded2.indices[:5]:{cfp_folded2.indices[:5]}")
print(f"fp_folded.indices[:5]:{fp_folded.indices[:5]}")

# RDKit Morgan 指纹（类似于 ECFP）可以轻松转换为Fingerprint
from rdkit import Chem
from rdkit.Chem import AllChem
mol = Chem.MolFromSmiles('Cc1ccccc1')
mfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
print(f"Morgan Fingerprint (2048 bits): {mfp}")
print(f"Non-zero indices: {list(mfp.GetOnBits())}")
print(f"Fingerprint.from_rdkit(mfp):{Fingerprint.from_rdkit(mfp)}")

#同样，Fingerprint 可以轻松转换为 NumPy数组 或 SciPy 稀疏矩阵。
print(f"fp_folded.to_vector():{fp_folded.to_vector()}")
print(f"fp_folded.to_vector(sparse=False):{fp_folded.to_vector(sparse=False)}")
print(f"np.where(fp_folded.to_vector(sparse=False))[0]:{np.where(fp_folded.to_vector(sparse=False))[0]}")
print(f"cfp_folded.to_vector(sparse=False):{cfp_folded.to_vector(sparse=False)}")
print(f"cfp_folded.to_vector(sparse=False).sum():{cfp_folded.to_vector(sparse=False).sum()}")

#可以对指纹执行基本代数函数。如果任一指纹是位指纹，则所有代数函数都是按位的。支持以下按位运算：Equality、Union/OR、Intersection/AND、Difference/AND NOT、XOR

#对于计数或浮点指纹，按位运算仍然是可能的，但代数运算应用于计数。

#最后，指纹可以批量添加和平均，在合理时产生计数或浮动指纹。
from e3fp.fingerprint.fprint import add, mean
fps = [Fingerprint(np.random.randint(0, 32, 8), bits=32) for i in range(100)]
print(f"add(fps):{add(fps)}")
print(f"mean(fps):{mean(fps)}")


"""
fprint_params = {'bits': 4096, 'radius_multiplier': 1.5, 'rdkit_invariants': True}
confgen_params = {'max_energy_diff': 20.0, 'first': 3}
smiles = "COC(=O)C(C1CCCCN1)C2=CC=CC=C2"

# Generating Conformers from SMILES
from e3fp.pipeline import confs_from_smiles
mol = confs_from_smiles(smiles, "ritalin", confgen_params=confgen_params)
print(mol.GetNumConformers())

# Generating Fingerprints from Conformers
from e3fp.pipeline import fprints_from_mol
fprints = fprints_from_mol(mol, fprint_params=fprint_params)
print(len(fprints))
print(fprints[0])
print(fprints[1])
print(fprints[2])

# Generating Fingerprints from SMILES
from e3fp.pipeline import fprints_from_smiles
fprints = fprints_from_smiles(smiles, "ritalin", confgen_params=confgen_params, fprint_params=fprint_params)
print(fprints[0])
"""