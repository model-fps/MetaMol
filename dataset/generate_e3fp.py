import os
import logging
import numpy as np
from python_utilities.io_tools import touch_dir
from e3fp.conformer.util import MolItemName
from e3fp.config.params import read_params, get_default_value, get_value
from e3fp.pipeline import fprints_from_sdf, fprints_from_mol, fprints_from_smiles
from e3fp.fingerprint.fprinter import Fingerprinter, BITS
import e3fp.fingerprint.fprint as fp
from e3fp.fingerprint.fprint import add, mean

LEVEL_DEF = get_default_value("fingerprinting", "level", int)
RADIUS_MULTIPLIER_DEF = get_default_value(
    "fingerprinting", "radius_multiplier", float
)
FIRST_DEF = get_default_value("fingerprinting", "first", int)
COUNTS_DEF = get_default_value("fingerprinting", "counts", bool)
STEREO_DEF = get_default_value("fingerprinting", "stereo", bool)
INCLUDE_DISCONNECTED_DEF = get_default_value(
    "fingerprinting", "include_disconnected", bool
)
RDKIT_INVARIANTS_DEF = get_default_value(
    "fingerprinting", "rdkit_invariants", bool
)
EXCLUDE_FLOATING_DEF = get_default_value(
    "fingerprinting", "exclude_floating", bool
)
REMOVE_DUPLICATE_SUBSTRUCTS_DEF = get_default_value(
    "fingerprinting", "remove_duplicate_substructs", bool
)
OUT_EXT_DEF = ".fp.bz2"
def e3fps_dict_from_mol(
    mol,
    bits=BITS,
    level=LEVEL_DEF,
    radius_multiplier=RADIUS_MULTIPLIER_DEF,
    first=FIRST_DEF,
    counts=COUNTS_DEF,
    stereo=STEREO_DEF,
    include_disconnected=INCLUDE_DISCONNECTED_DEF,
    rdkit_invariants=RDKIT_INVARIANTS_DEF,
    exclude_floating=EXCLUDE_FLOATING_DEF,
    remove_duplicate_substructs=REMOVE_DUPLICATE_SUBSTRUCTS_DEF,
    out_dir_base=None,
    out_ext=OUT_EXT_DEF,
    save=False,
    all_iters=False,
    overwrite=False,
):
    """Build a E3FP fingerprint from a mol with at least one conformer.

    Parameters
    ----------
    mol : RDKit Mol
        Input molecule with one or more conformers to be fingerprinted.
    bits : int
        Set number of bits for final folded fingerprint.
    level : int, optional
        Level/maximum number of iterations of E3FP. If -1 is provided, it runs
        until termination, and `all_iters` is set to False.
    radius_multiplier : float, optional
        Radius multiplier for spherical shells.
    first : int, optional
        First `N` number of conformers from file to fingerprint. If -1, all
        are fingerprinted.
    counts : bool, optional
        Instead of bit-based fingerprints. Otherwise, generate count-based
        fingerprints.
    stereo : bool, optional
        Incorporate stereochemistry in fingerprint.
    remove_duplicate_substructs : bool, optional
        If a substructure arises that corresponds to an identifier already in
        the fingerprint, then the identifier for the duplicate substructure is
        not added to fingerprint.
    include_disconnected : bool, optional
        Include disconnected atoms when hashing and for stereo calculations.
        Turn off purely for testing purposes, to make E3FP more like ECFP.
    rdkit_invariants : bool, optional
        Use the atom invariants used by RDKit for its Morgan fingerprint.
    exclude_floating : bool, optional:
        Mask atoms with no bonds (usually floating ions) from the fingerprint.
        These are often placed arbitrarily and can confound the fingerprint.
    out_dir_base : str, optional
        Basename of out directory to save fingerprints. Iteration number is
        appended.
    out_ext : str, optional
        Extension on fingerprint pickles, used to determine compression level.
    save : bool, optional
        Save fingerprints to directory.
    all_iters : bool, optional
        Save fingerprints from all iterations to file(s).
    overwrite : bool, optional
        Overwrite pre-existing file.

    Deleted Parameters
    ------------------
    sdf_file : str
        SDF file path.
    """

    if level is None:
        level = -1

    if bits in (-1, None):
        bits = BITS

    if save:
        filenames = []
        all_files_exist = True
        if level == -1 or not all_iters:
            if level == -1:
                dir_name = "{!s}_complete".format(out_dir_base)
            else:
                dir_name = "{!s}{:d}".format(out_dir_base, level)
            touch_dir(dir_name)
            filenames.append(
                os.path.join(dir_name, "{!s}".format(out_ext))
            )
            if not os.path.isfile(filenames[0]):
                all_files_exist = False
        else:
            for i in range(level + 1):
                dir_name = "{:s}{:d}".format(out_dir_base, i)
                touch_dir(dir_name)
                filename = os.path.join(dir_name, "{!s}".format(out_ext))
                filenames.append(filename)
                if not os.path.isfile(filename):
                    all_files_exist = False

        if all_files_exist and not overwrite:
            logging.warning(
                "All fingerprint files for {!s} already exist. "
            )
            return {}

    fingerprinter = Fingerprinter(
        bits=bits,
        level=level,
        radius_multiplier=radius_multiplier,
        counts=counts,
        stereo=stereo,
        include_disconnected=include_disconnected,
        rdkit_invariants=rdkit_invariants,
        exclude_floating=exclude_floating,
        remove_duplicate_substructs=remove_duplicate_substructs,
    )

    try:
        fprints_dict = {}
        for j, conf in enumerate(mol.GetConformers()):
            if j == first:
                j -= 1
                break
            fingerprinter.run(conf, mol)
            level_range = range(level + 1)
            if level == -1 or not all_iters:
                level_range = (level,)
            else:
                level_range = range(level + 1)
            for i in level_range:
                fprint = fingerprinter.get_fingerprint_at_level(i)
                fprints_dict.setdefault(i, []).append(fprint)
    except Exception:
        logging.error(
            "Error generating fingerprints.",
            exc_info=True,
        )
        return {}

    if save:
        if level == -1 or not all_iters:
            fprints = fprints_dict[max(fprints_dict.keys())]
            try:
                fp.savez(filenames[0], *fprints)
            except Exception:
                logging.error(
                    "Error saving fingerprints for {:s}".format(filenames[0]),
                    exc_info=True,
                )
                return {}
        else:
            try:
                for i, fprints in sorted(fprints_dict.items()):
                    fp.savez(filenames[i], *fprints)
            except Exception:
                logging.error(
                    "Error saving fingerprints for {:s}".format(filenames[i]),
                    exc_info=True,
                )
                return {}

    return fprints_dict


def e3fps_from_fprints_dict(fprints_dict, level=-1):
    """Get fingerprint at `level` from dict of level to fingerprint."""
    fprints_list = fprints_dict.get(
        level, fprints_dict[max(fprints_dict.keys())]
    )
    return fprints_list

def e3fps_from_mol(mol, fprint_params={}, save=False):
    """Generate fingerprints for all `first` conformers in mol."""
    fprints_dict = e3fps_dict_from_mol(mol, save=save, **fprint_params)
    level = fprint_params.get("level", -1)
    fprints = e3fps_from_fprints_dict(fprints_dict, level=level)
    e3fp_fprints = []
    for fprint in fprints:
        e3fp_fprint = fprint.to_vector(sparse=False).astype(int)
        e3fp_fprints.append(e3fp_fprint)
    return e3fp_fprints

def get_float_array(length, dict):
    float_array = np.zeros(length)
    for index, value in dict.items():
        if 0 <= index < length:  # 确保索引在数组范围内
            float_array[index] = value
    return float_array

def e3fps_from_smiles(smiles, confgen_params, fprint_params):
    fprints = fprints_from_smiles(smiles, str(smiles), confgen_params=confgen_params, fprint_params=fprint_params)
    float_fps = mean(fprints)
    float_array = get_float_array(2048, float_fps.counts)
    # print(f"len(float_array):{len(float_array)}")  # 2048
    # print(f"float_array:{float_array}")
    non_zero_bit = {index: value for index, value in enumerate(float_array) if value != 0}
    # print(f"non_zero_bit:{non_zero_bit}")
    # print(f"len(non_zero_bit):{len(non_zero_bit)}")
    return float_array

if __name__ == "__main__":
    smiles = 'CN1CCN(c2cccc3[nH]c(-c4n[nH]c5cc(-c6ccc(N)c(F)c6)ccc45)nc23)CC1'
    fprint_params = {'bits': 2048, 'radius_multiplier': 2, 'first': -1, 'rdkit_invariants': True}
    confgen_params = {'max_energy_diff': 20.0, 'first': -1}
    fp_numpy = e3fps_from_smiles(smiles, confgen_params, fprint_params)
    fp_numpy = np.array(fp_numpy)
    print(sum(fp_numpy))