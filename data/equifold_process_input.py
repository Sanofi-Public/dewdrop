import configparser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import pickle
import subprocess
import sys
from collections import defaultdict
import string
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import default_convert
from torch_geometric.data import Data
from tqdm import tqdm
# sys.path.append("..")
# try:
from data.pdb_to_data import process_pdb 
# except ModuleNotFoundError:
#     sys.path.append("./data")
#     import pdb_to_data
# pylint: disable=wrong-import-position
from utils.cg import (
    N_CG_MAX,
    cg_atom_ambiguous_np,
    cg_atom_rename_np,
    cg_dict,
    cg_to_idx,
    cg_to_np,
    cgidx_to_atomidx,
    resname_to_idx,
)
from openfold_light.data_transforms import atom37_to_torsion_angles
from openfold_light.residue_constants import (
    between_res_bond_length_c_n,
    between_res_bond_length_stddev_c_n,
    between_res_cos_angles_c_n_ca,
    between_res_cos_angles_ca_c_n,
    ca_ca,
    load_stereo_chemical_props,
    residue_atoms,
    restype_1to3,
    van_der_waals_radius,
)
from utils.utils import get_euclidean, get_euclidean_kabsch
from utils.utils_data import MAX_DIST, cg_X0, sequence_to_feats

import pdb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_DIST = 32  # max residue distance
NUM_EDGE_TYPE = MAX_DIST * 2 + 2
# ---- template coords
template_coords = "./utils/cg_X0.npz"
if os.path.exists(template_coords):
    cg_X0 = torch.from_numpy(np.load(template_coords)["x"].astype(np.float32))

    # ADDED: for UNK, add an extra coord that's all 0
    # print("cg_X0.shape: ", cg_X0.shape)
    # cg_X0 = torch.cat([cg_X0, torch.zeros(1,cg_X0.shape[1],cg_X0.shape[2],dtype=torch.float32)], dim=0)
    # print("cg_X0.shape (after): ", cg_X0.shape)
else:
    cg_X0 = None
# ---- precompute arrs to be used for struct violation loss
residue_bonds, residue_virtual_bonds, residue_bond_angles = load_stereo_chemical_props()
tol_factor = 3
ambiguous_atoms = defaultdict(set)
for resname, cgs in cg_dict.items():
    for atom in residue_atoms[resname]:
        # atom appears in more than 1 cg
        if sum([atom in cg for cg in cgs]) > 1:
            ambiguous_atoms[resname].add(atom)


# -- bond length
# (idx1, idx2, length, tol_factor * stddev)
def cross_cg_bond(resname, bond):
    cgs = cg_dict[resname]
    for cg in cgs:
        if bond.atom1_name in cg and bond.atom2_name in cg:
            return False
    return True


bonds_np = dict()
for resname, bonds in residue_bonds.items():
    if resname == "UNK":
        continue
    bonds_virtual = residue_virtual_bonds[resname]
    bonds = bonds + bonds_virtual
    # nprev = len(bonds)

    # need to keep all bonds to inform clash loss
    bonds_ = []
    for bond in bonds:
        length_mask = (
            cross_cg_bond(resname, bond)
            or bond.atom1_name in ambiguous_atoms[resname]
            or bond.atom2_name in ambiguous_atoms[resname]
        )
        bonds_.append(
            (
                residue_atoms[resname].index(bond.atom1_name),
                residue_atoms[resname].index(bond.atom2_name),
                bond.length,
                tol_factor * bond.stddev,
                length_mask,  # whether to mask bond loss since unambiguous
            )
        )
    bonds_np[resname] = tuple([np.asarray(x) for x in zip(*bonds_)])


# -- bond angle
# (idx1, idx2, idx3, mid, tol)
# idx2 is the middle atom
def cross_cg_angle(resname, bond):
    cgs = cg_dict[resname]
    for cg in cgs:
        if bond.atom1_name in cg and bond.atom2_name in cg and bond.atom3name in cg:
            return False
    return True


bond_angles_np = dict()
for resname, bond_angles in residue_bond_angles.items():
    if resname == "UNK":
        continue
    bond_angles_ = []
    # nprev = len(bond_angles)
    for bond in bond_angles:
        # could skip angles
        if not (
            cross_cg_angle(resname, bond)
            or bond.atom1_name in ambiguous_atoms[resname]
            or bond.atom2_name in ambiguous_atoms[resname]
            or bond.atom3name in ambiguous_atoms[resname]
        ):
            continue

        cosa = np.cos(bond.angle_rad)
        cosap = np.cos(bond.angle_rad + tol_factor * bond.stddev)
        cosan = np.cos(bond.angle_rad - tol_factor * bond.stddev)
        mid = (cosan + cosap) / 2.0
        tol = np.abs(cosan - mid)
        bond_angles_.append(
            (
                residue_atoms[resname].index(bond.atom1_name),
                residue_atoms[resname].index(bond.atom2_name),
                residue_atoms[resname].index(bond.atom3name),
                mid,
                tol,
            )
        )
    bond_angles_np[resname] = tuple([np.asarray(x) for x in zip(*bond_angles_)])
# clash
atom_width_np = {
    resname: np.asarray([van_der_waals_radius[atom[0]] for atom in atoms])
    for resname, atoms in residue_atoms.items()
}


def get_peptide_bond_lengths(resname):
    c_n = between_res_bond_length_c_n[0] if resname != "PRO" else between_res_bond_length_c_n[1]
    c_n_stddev = (
        between_res_bond_length_stddev_c_n[0]
        if resname != "PRO"
        else between_res_bond_length_stddev_c_n[1]
    )

    return c_n, c_n_stddev


def get_cg_RT(cg_cgidx, cg_X, cg_mask, cg_atom_mask, use_kabsch):
    # get transformation
    if not use_kabsch:
        cg_T, cg_R = get_euclidean(torch.from_numpy(cg_X[:, :3]))
    else:
        # DEBUG
        # print("ref_coord: ", cg_X0)

        cg_T, cg_R = get_euclidean_kabsch(
            torch.from_numpy(cg_X), cg_X0[cg_cgidx], torch.from_numpy(cg_atom_mask)
        )
    cg_T, cg_R = cg_T.numpy(), cg_R.numpy()

    return cg_X, cg_T, cg_R


class data_process:
    def __init__(self, CONFIG):

        self.CONFIG = CONFIG

    def process_input(self, csv_path, pdb_path, model_type="ab", ncpu=1):
        df = pd.read_csv(csv_path)
        uids = df["uid"].tolist()
        if model_type == "ab":
            seqs1 = df["heavy"].tolist()
            seqs2 = df["light"].tolist()
        else:
            seqs1 = df["seq"].tolist()
            seqs2 = [None] * len(seqs1)
        dataset = []
        chain_id = df["chain_id"].tolist()
       
        # ADDED: unique chain_ids for chain_ids mappings 
        unique_chain_id = df["chain_id"].str.split("_").str[0].unique().tolist()
        # unique_chain_id = None
        pbar = tqdm(range(len(uids)))
        for i in pbar:
            pbar.set_description(f"uid = {uids[i]}", refresh=True)
            dataset.append(
                self.process_one([uids[i], seqs1[i], seqs2[i], chain_id[i]], pdb_path, model_type, unique_chain_id=unique_chain_id)
            )
        return dataset

    def process_one(self, job, pdb_path=None, model_type=None, unique_chain_id=None):
        """Combines fasta with process pdb"""
        uid, seq1, seq2, chain_id = job
        (
            cg_cgidx,
            cg_resnum,
            scatter_index,
            scatter_w,
            dst_resnum,
            dst_atom,
            dst_resname,
            offset,
        ) = sequence_to_feats(seq1, dst_idx_offset=0)
        if seq2 is not None:
            (
                cg_cgidx2,
                cg_resnum2,
                scatter_index2,
                scatter_w2,
                dst_resnum2,
                dst_atom2,
                dst_resname2,
                _,
            ) = sequence_to_feats(seq2, dst_idx_offset=offset)
            seq2_offset = len(seq1) + MAX_DIST
            cg_cgidx = np.concatenate([cg_cgidx, cg_cgidx2])
            cg_resnum = np.concatenate([cg_resnum, cg_resnum2 + seq2_offset])
            scatter_index = np.concatenate([scatter_index, scatter_index2])
            scatter_w = np.concatenate([scatter_w, scatter_w2])
            dst_resnum = np.concatenate([dst_resnum, dst_resnum2 + seq2_offset])
            dst_atom = np.concatenate([dst_atom, dst_atom2])
            dst_resname = np.concatenate([dst_resname, dst_resname2])

        # Get updated pdb with offset between heavy chain and light chain
        weight_fwr_cdr_map_cg_resnum_uv = None

        # Update with pdb file information 
        if pdb_path is not None:
            pdb_dict = self.pdb_to_tensor(
                # ADDED: change file type
                uid + "_" + chain_id + ".pdb",
                # uid + "_" + chain_id + ".cif", # use cif as file type
                pdb_path,
                # ADDED: remove trailing integar
                chain_id.split('_')[0],
                # chain_id,
                len(seq1),
                model_type=model_type,
                unique_chain_id=unique_chain_id
            )

            # parse pdb_dict for information previous extracted from csv
            cg_resnum=pdb_dict["cg_resnum"]
            cg_cgidx=pdb_dict["cg_cgidx"]
            scatter_index=pdb_dict['scatter_index']
            scatter_w=pdb_dict['scatter_w']
            dst_resnum=pdb_dict['dst_resnum']
            dst_atom=pdb_dict['dst_atom']
            dst_resname=pdb_dict['dst_resname']
            offset=pdb_dict["final_offset"]

            # Scaling
            if self.CONFIG["preprocessing"]["differential_weight_fwr_cdr_FAPE"] != "":
                weightage = self.CONFIG.getfloat("preprocessing", "differential_weight_fwr_cdr_FAPE")
                
                weight_cdr_cdr_pairs = False
                if self.CONFIG["preprocessing"]["weight_cdr_cdr_pairs_FAPE"] != "":
                    weight_cdr_cdr_pairs = self.CONFIG.getboolean("preprocessing", "weight_cdr_cdr_pairs_FAPE")
                    
                seq, annot = get_annot_VH_VL(uid, seq1, None)
                if annot is not None:
                    weight_fwr_cdr_map_cg_resnum_uv = create_weight_fwr_cdr_map_cg_resnum_uv(
                        annot, cg_resnum, weightage=weightage, num_of_atoms=cg_X0.shape[1], weight_cdr_cdr_pairs=weight_cdr_cdr_pairs
                    )
        else:
            pdb_dict = defaultdict(lambda: None)

        dtype = torch.float32
        cg_cgidx=torch.from_numpy(cg_cgidx)
        cg_resnum=torch.from_numpy(cg_resnum)
        scatter_index=torch.from_numpy(scatter_index)
        scatter_w=torch.from_numpy(scatter_w)
        dst_resnum=torch.from_numpy(dst_resnum)

        # DEBUG: num_nodes should match size of cg_resnum and size of cg_cgidx
        # if cg_resnum.shape[0] != pdb_dict['cg_mask'].shape[0]: 
        #     tqdm.write(f"Shape mismatch: cg_resnum = {cg_resnum.shape} and cg_mask = {pdb_dict['cg_mask'].shape}.\n")

        # ADDED: changed this part to use pdb data 
        data = Data(
            num_nodes=len(cg_cgidx),
            cg_resnum=cg_resnum,
            scatter_index=scatter_index,
            scatter_w= scatter_w,
            dst_resnum=dst_resnum,
            cg_cgidx=cg_cgidx,
            cg_X0=cg_X0[cg_cgidx],
            dst_atom=dst_atom,
            dst_resname=dst_resname,
            uid=uid,
            chain_id=chain_id,
            # from pdb
            # res/full atoms level
            sequence=pdb_dict["sequence"],
            resnum=pdb_dict["resnum"],
            # cg
            cg_pdb_resnum=pdb_dict["cg_resnum"],
            cg_pdb_cgidx=pdb_dict["cg_cgidx"],
            cg_mask=pdb_dict["cg_mask"],
            # 0
            cg_T=pdb_dict["cg_T"],
            cg_R=pdb_dict["cg_R"],
            cg_atom_mask=pdb_dict["cg_atom_mask"],
            cg_X=pdb_dict["cg_X"],
            cg_amb=pdb_dict["cg_amb"],
            # alt
            cg_T_alt=pdb_dict["cg_T_alt"],  # should always be identical to cg_T
            cg_R_alt=pdb_dict["cg_R_alt"],
            cg_atom_mask_alt=pdb_dict["cg_atom_mask_alt"],
            cg_X_alt=pdb_dict["cg_X_alt"],
            cg_amb_alt=pdb_dict["cg_amb_alt"],
            # CG to "PDB" mapping
            scatter_pdb_index=pdb_dict["scatter_index"],
            scatter_pdb_w=pdb_dict["scatter_w"],
            dst_pdb_resnum=pdb_dict["dst_resnum"],
            dst_pdb_atom=pdb_dict["dst_atom"],
            dst_pdb_resname=pdb_dict["dst_resname"],
            final_offset=pdb_dict["final_offset"],
            # struct
            dst_bonds_i1=pdb_dict["dst_bonds_i1"],
            dst_bonds_i2=pdb_dict["dst_bonds_i2"],
            dst_bonds_l=pdb_dict["dst_bonds_l"],
            dst_bonds_tol=pdb_dict["dst_bonds_tol"],
            dst_bonds_mask=pdb_dict["dst_bonds_mask"],
            dst_angles_i1=pdb_dict["dst_angles_i1"],
            dst_angles_i2=pdb_dict["dst_angles_i2"],
            dst_angles_i3=pdb_dict["dst_angles_i3"],
            dst_angles_mid=pdb_dict["dst_angles_mid"],
            dst_angles_tol=pdb_dict["dst_angles_tol"],
            dst_atom_widths=pdb_dict["dst_atom_widths"],
            dst_atom_mask=pdb_dict["dst_atom_mask"],
            weight_fwr_cdr_map_cg_resnum_uv=weight_fwr_cdr_map_cg_resnum_uv,
            torsion_angles_sin_cos=pdb_dict["torsion_angles_sin_cos"],
            alt_torsion_angles_sin_cos=pdb_dict["alt_torsion_angles_sin_cos"],
            torsion_angles_mask=pdb_dict["torsion_angles_mask"],
        )
        
        return data


    # get post-processed pdb(simply combined the above 4 rows together)
    def pdb_to_tensor(self, pdb_name, path, chain_id, len_chain_1, model_type="ab", is_train=True, unique_chain_id=None):
        path = os.path.join(path, pdb_name)

        if not is_train:
            path = "./test/" + pdb_name

        chain_1, chain_2 = None, None
        if model_type == "ab":
            chain_1 = chain_id[0]  # heavy chain
            chain_2 = chain_id[1]  # light chain
        else:
            chain_1 = chain_id

        result_1 = process_pdb(path, chain_id=chain_1, unique_chain_id=unique_chain_id)

        returned_pdb_data_1 = default_convert(pdb_feats_to_data(result_1, True))
        if self.CONFIG.getboolean("preprocessing", "compute_torsion_angles"):
            returned_pdb_data_1 = atom37_to_torsion_angles()(returned_pdb_data_1)
        else:
            returned_pdb_data_1["torsion_angles_sin_cos"] = None
            returned_pdb_data_1["alt_torsion_angles_sin_cos"] = None
            returned_pdb_data_1["torsion_angles_mask"] = None
        if chain_2:
            result_2 = process_pdb(path, chain_id=chain_2)
            seq2_offset = len_chain_1 + MAX_DIST
            returned_pdb_data_2 = default_convert(
                pdb_feats_to_data(result_2, True, dst_idx_offset=seq2_offset)
            )

            # save updated pdb in a dict
            feats = {
                # res/full atoms level
                "sequence": returned_pdb_data_1["sequence"] + returned_pdb_data_2["sequence"],
                "resnum": torch.cat(
                    (
                        returned_pdb_data_1["resnum"],
                        returned_pdb_data_2["resnum"] + seq2_offset,
                    )
                ),  # offset ADDED
                # cg
                "cg_resnum": torch.cat(
                    (
                        returned_pdb_data_1["cg_resnum"],
                        returned_pdb_data_2["cg_resnum"] + seq2_offset,
                    )
                ),  # offset ADDED
                "cg_cgidx": torch.cat(
                    (returned_pdb_data_1["cg_cgidx"], returned_pdb_data_2["cg_cgidx"])
                ),
                "cg_mask": torch.cat((returned_pdb_data_1["cg_mask"], returned_pdb_data_2["cg_mask"])),
                # 0
                "cg_T": torch.cat((returned_pdb_data_1["cg_T"], returned_pdb_data_2["cg_T"])),
                "cg_R": torch.cat((returned_pdb_data_1["cg_R"], returned_pdb_data_2["cg_R"])),
                "cg_atom_mask": torch.cat(
                    (
                        returned_pdb_data_1["cg_atom_mask"],
                        returned_pdb_data_2["cg_atom_mask"],
                    )
                ),
                "cg_X": torch.cat((returned_pdb_data_1["cg_X"], returned_pdb_data_2["cg_X"])),
                "cg_amb": torch.cat((returned_pdb_data_1["cg_amb"], returned_pdb_data_2["cg_amb"])),
                # alt
                "cg_T_alt": torch.cat(
                    (returned_pdb_data_1["cg_T_alt"], returned_pdb_data_2["cg_T_alt"])
                ),  # should always be identical to cg_T
                "cg_R_alt": torch.cat(
                    (returned_pdb_data_1["cg_R_alt"], returned_pdb_data_2["cg_R_alt"])
                ),
                "cg_atom_mask_alt": torch.cat(
                    (
                        returned_pdb_data_1["cg_atom_mask_alt"],
                        returned_pdb_data_2["cg_atom_mask_alt"],
                    )
                ),
                "cg_X_alt": torch.cat(
                    (returned_pdb_data_1["cg_X_alt"], returned_pdb_data_2["cg_X_alt"])
                ),
                "cg_amb_alt": torch.cat(
                    (returned_pdb_data_1["cg_amb_alt"], returned_pdb_data_2["cg_amb_alt"])
                ),
                # CG to "PDB" mapping
                "scatter_index": torch.cat(
                    (
                        returned_pdb_data_1["scatter_index"],
                        returned_pdb_data_2["scatter_index"],
                    )
                ),
                "scatter_w": torch.cat(
                    (returned_pdb_data_1["scatter_w"], returned_pdb_data_2["scatter_w"])
                ),
                "dst_resnum": torch.cat(
                    (
                        returned_pdb_data_1["dst_resnum"],
                        returned_pdb_data_2["dst_resnum"] + seq2_offset,
                    )
                ),  # offset ADDED
                "dst_atom": np.concatenate(
                    [returned_pdb_data_1["dst_atom"], returned_pdb_data_2["dst_atom"]]
                ),
                "dst_resname": np.concatenate(
                    [returned_pdb_data_1["dst_resname"], returned_pdb_data_2["dst_resname"]]
                ),
                "final_offset": returned_pdb_data_1["final_offset"]
                + returned_pdb_data_2["final_offset"],
                # struct
                "dst_bonds_i1": torch.cat(
                    (
                        returned_pdb_data_1["dst_bonds_i1"],
                        returned_pdb_data_2["dst_bonds_i1"],
                    )
                ),  # w/o the offset, incorrect violation loss
                "dst_bonds_i2": torch.cat(
                    (
                        returned_pdb_data_1["dst_bonds_i2"],
                        returned_pdb_data_2["dst_bonds_i2"],
                    )
                ),  # w/o the offset, incorrect violation loss
                "dst_bonds_l": torch.cat(
                    (returned_pdb_data_1["dst_bonds_l"], returned_pdb_data_2["dst_bonds_l"])
                ),
                "dst_bonds_tol": torch.cat(
                    (
                        returned_pdb_data_1["dst_bonds_tol"],
                        returned_pdb_data_2["dst_bonds_tol"],
                    )
                ),
                "dst_bonds_mask": torch.cat(
                    (
                        returned_pdb_data_1["dst_bonds_mask"],
                        returned_pdb_data_2["dst_bonds_mask"],
                    )
                ),
                "dst_angles_i1": torch.cat(
                    (
                        returned_pdb_data_1["dst_angles_i1"],
                        returned_pdb_data_2["dst_angles_i1"],
                    )
                ),  # w/o the offset, incorrect violation loss
                "dst_angles_i2": torch.cat(
                    (
                        returned_pdb_data_1["dst_angles_i2"],
                        returned_pdb_data_2["dst_angles_i2"],
                    )
                ),  # w/o the offset, incorrect violation loss
                "dst_angles_i3": torch.cat(
                    (
                        returned_pdb_data_1["dst_angles_i3"],
                        returned_pdb_data_2["dst_angles_i3"],
                    )
                ),  # w/o the offset, incorrect violation loss
                "dst_angles_mid": torch.cat(
                    (
                        returned_pdb_data_1["dst_angles_mid"],
                        returned_pdb_data_2["dst_angles_mid"],
                    )
                ),
                "dst_angles_tol": torch.cat(
                    (
                        returned_pdb_data_1["dst_angles_tol"],
                        returned_pdb_data_2["dst_angles_tol"],
                    )
                ),
                "dst_atom_widths": torch.cat(
                    (
                        returned_pdb_data_1["dst_atom_widths"],
                        returned_pdb_data_2["dst_atom_widths"],
                    )
                ),
                "dst_atom_mask": torch.cat(
                    (
                        returned_pdb_data_1["dst_atom_mask"],
                        returned_pdb_data_2["dst_atom_mask"],
                    )
                ),
            }
            return feats
        return returned_pdb_data_1


# main
def pdb_feats_to_data(pdb_feats, use_kabsch, real_pdb=False, dst_idx_offset=0):
    try:
        sequence = pdb_feats["sequence"][0].decode()
    except:
        sequence = pdb_feats["sequence"].decode()

    # residue type. should match restypes
    aatype = np.where(pdb_feats["aatype"] == 1)[1].astype(np.int64)

    # items to save
    pos = pdb_feats["all_atom_positions"]  # [N, 37, 3]
    mask = pdb_feats["all_atom_mask"]  # [N, 37]
    
    # -- all atoms to cg mapping
    # - ground truth
    # Ncg := total number of CG nodes
    resnum = np.arange(len(sequence), dtype=np.int64)
    cg_resnums = []  # [Ncg]; for edge attributes
    cg_cgidxs = []  # [Ncg]; for node attributes
    cg_Xs = []  # [Ncg, N_CG_MAX, 3]
    cg_atom_masks = (
        []
    )  # [Ncg, N_CG_MAX]; atom level mask; 1.0 if both atom experimentally present and belongs to the CG node else 0.0
    cg_masks = []  # [Ncg]; CG level mask; 1.0 if the first three atoms present else 0.0
    for res, cgs in cg_dict.items():
        res_idx = resname_to_idx[res]

        # get all residues of the type "res"
        ii = np.where(aatype == res_idx)[0]

        # DEBUG: Catch cases where ii's first dimension doesn't match Ncg
        # tqdm.write(f"""DEBUG: ii = {ii}\n""")

        # do the mapping
        for j in range(len(cgs)):
            cg = (res_idx, j)

            # relevant atoms among 37 heavy atoms
            icg_atoms = cg_to_np[cg]

            # DEBUG: 
            # tqdm.write("icg_atoms: ", icg_atoms.shape)

            # (TODO) ADD: Mask residues containing atoms with coordinates as np.NaN

            # get CG mask
            atom_mask = np.zeros((len(ii), N_CG_MAX), dtype=mask.dtype)
            if real_pdb:
                mask_ = mask[ii][:, icg_atoms]
                cg_mask = ~(mask_[:, :3] == 0).any(axis=1)  # true here means good
                atom_mask[cg_mask, : len(icg_atoms)] = mask_[cg_mask]
                atom_mask[~cg_mask, : len(icg_atoms)] = 1.0  # fill with dummy values
            else:
                # since these are de novo all atoms are assumed to be present (except for H)
                cg_mask = np.ones(len(ii), dtype=bool)  # true here means good
                atom_mask[:, : len(icg_atoms)] = 1.0

            cg_idxs = np.full(len(ii), cg_to_idx[cg], dtype=int)

            # get pos
            pos_ = pos[ii][:, icg_atoms]

            if use_kabsch:
                pos_[~cg_mask] = cg_X0[cg_idxs[~cg_mask], : len(icg_atoms)]
            else:
                # filling in dummy values for first three if not present
                pos_[~cg_mask, :3] = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

            # CG frame
            cg_X = np.zeros((len(pos_), N_CG_MAX, 3), dtype=mask.dtype)
            cg_X[:, : len(icg_atoms)] = pos_

            # collect info
            cg_Xs.append(cg_X)
            cg_resnums.append(ii)
            cg_cgidxs.append(cg_idxs)
            cg_masks.append(cg_mask)
            cg_atom_masks.append(atom_mask)

    # DEBUG: Catch cases where ii's first dimension doesn't match Ncg
    # tqdm.write(f"""DEBUG: cg_resnums = {cg_resnums}\n""")

    cg_resnum = np.concatenate(cg_resnums, axis=0)
    cg_cgidx = np.concatenate(cg_cgidxs, axis=0)
    cg_mask = np.concatenate(cg_masks, axis=0).astype(np.float32)
    cg_atom_mask = np.concatenate(cg_atom_masks, axis=0).astype(np.float32)
    cg_X = np.concatenate(cg_Xs, axis=0)

    # DEBUG 
    # tqdm.write(f"""DEBUG: N_cg = {cg_cgidx.shape[0]}\n""")

    # ADDED: make sure number of CG groups matches
    assert cg_resnum.shape[0] == cg_cgidx.shape[0] and cg_resnum.shape[0] == cg_X.shape[0]

    if cg_mask.sum() < 30:
        assert False

    cg_X, cg_T, cg_R = get_cg_RT(cg_cgidx, cg_X, cg_mask, cg_atom_mask, use_kabsch)
    cg_amb = cg_atom_ambiguous_np[cg_cgidx]
    if real_pdb:
        # reject CG nodes that have too big of rmsd from template coords
        cg_X_fit = torch.einsum(
            "rij,rkj->rki", torch.from_numpy(cg_R), cg_X0[cg_cgidx]
        ) + torch.from_numpy(cg_T).unsqueeze(1)
        d = ((cg_X_fit - cg_X).square().sum(-1) + 1e-6).sqrt()
        ireject = ((d * cg_atom_mask).sum(-1) / cg_atom_mask.sum(-1)) > 1**2
        cg_mask[ireject] = 0.0

    # -- alternative truth to account for possible 180 deg symmetry
    # permute atoms
    permut = cg_atom_rename_np[cg_cgidx]
    cg_X_alt = np.transpose(cg_X[np.arange(len(permut)), permut.T], (1, 0, 2))
    cg_amb_alt = cg_amb[np.arange(len(permut)), permut.T].T
    cg_atom_mask_alt = cg_atom_mask[np.arange(len(permut)), permut.T].T
    cg_X_alt, cg_T_alt, cg_R_alt = get_cg_RT(
        cg_cgidx, cg_X_alt, cg_mask, cg_atom_mask_alt, use_kabsch
    )

    # -- indices for scatter reduction and structure violation calculation
    # compute residue based offsets
    dst_bonds = []
    dst_angles = []
    dst_atom_widths = []
    resnum_to_offset = {}
    offset = 0
    for i, aa in enumerate(sequence):
        resnum_to_offset[i] = offset
        resname = restype_1to3[aa]

        # ADDED: Skip UNK
        # if resname == "UNK": 
        #     continue

        # precompute arrs for struct violation loss
        i1, i2, l, tol, mask = bonds_np[resname]
        dst_bonds.append((i1 + offset, i2 + offset, l, tol, mask))
        i1, i2, i3, mid, tol = bond_angles_np[resname]
        dst_angles.append((i1 + offset, i2 + offset, i3 + offset, mid, tol))
        dst_atom_widths.append(atom_width_np[resname])

        offset_increment = len(residue_atoms[resname])

        # add peptide bond constraints
        if i < len(sequence) - 1: 
            resname_next = restype_1to3[sequence[i + 1]]
            ca_i = residue_atoms[resname].index("CA")
            c_i = residue_atoms[resname].index("C")
            n_ip1 = residue_atoms[resname_next].index("N")
            ca_ip1 = residue_atoms[resname_next].index("CA")
            i1 = [ca_i, c_i]
            i2 = [ca_ip1, n_ip1]
            c_n, c_n_stddev = get_peptide_bond_lengths(resname)
            # ca-ca / C[i] - N[i+1] bond
            dst_bonds.append(
                (
                    np.asarray(i1) + offset,
                    np.asarray(i2) + offset + offset_increment,
                    np.asarray([ca_ca, c_n]),
                    np.asarray([0.05, c_n_stddev * tol_factor]),  # first element is handpicked
                    np.asarray([1.0, 1.0]),
                )
            )
            # inter-residue angles
            i1 = [c_i, ca_i]
            i2 = [n_ip1 + offset_increment, c_i]
            i3 = [ca_ip1 + offset_increment, n_ip1 + offset_increment]
            mid = [between_res_cos_angles_c_n_ca[0], between_res_cos_angles_ca_c_n[0]]
            tol = [between_res_cos_angles_c_n_ca[1], between_res_cos_angles_ca_c_n[1]]
            dst_angles.append(
                (
                    np.asarray(i1) + offset,
                    np.asarray(i2) + offset,
                    np.asarray(i3) + offset,
                    np.asarray(mid),
                    np.asarray(tol),
                )
            )
        offset += offset_increment

    dst_bonds = [np.concatenate(x) for x in zip(*dst_bonds)]
    dst_angles = [np.concatenate(x) for x in zip(*dst_angles)]
    dst_atom_widths = np.concatenate(dst_atom_widths)

    # reduction index
    N_CG = len(cg_cgidx)
    scatter_index = np.zeros(N_CG * N_CG_MAX, dtype=int)
    scatter_w = np.zeros(N_CG * N_CG_MAX, dtype=float)
    dst_resnum = np.zeros(offset, dtype=int)
    dst_atom = np.zeros(offset, dtype=">U3")
    dst_resname = np.zeros(offset, dtype=">U3")
    dst_atom_mask = np.ones(offset, dtype=float)  # for rmsd calc against gt
    for i, (cgidx, resnum_) in enumerate(zip(cg_cgidx, cg_resnum)):
        atomidxs = cgidx_to_atomidx[cgidx]
        for k, (resname_, atom, atomidx, w) in enumerate(atomidxs):
            src_idx = i * N_CG_MAX + k
            dst_idx = resnum_to_offset[resnum_] + atomidx
            scatter_index[src_idx] = dst_idx + dst_idx_offset
            scatter_w[src_idx] = 1 / w
            dst_resnum[dst_idx] = resnum_
            dst_atom[dst_idx] = atom
            dst_resname[dst_idx] = resname_
            dst_atom_mask[dst_idx] = (
                cg_atom_mask[i][k] * cg_mask[i]
            )  # necessary due to dummy value filling

    # ADDED: make sure number of CG groups matches
    assert cg_resnum.shape[0] == cg_cgidx.shape[0] and cg_resnum.shape[0] == cg_mask.shape[0] 

    # DEBUG: Catch cases where ii's first dimension doesn't match Ncg
    # tqdm.write(f"""DEBUG: (After Reduction) N_cg = {cg_cgidx.shape[0]}\n""")

    if cg_mask.shape[0] != cg_resnum.shape[0]: 
        tqdm.write("Mismatch in number of CG groups and residues.\n")
        tqdm.write(f"sequence: {sequence} \n")
        tqdm.write(f"""cg_resnum: {cg_resnum.shape[0]}, cg_cgidx: {cg_cgidx.shape[0]}, cg_X: {cg_X.shape[0]}\n""")

    # save
    feats = {
        # res/full atoms level
        "sequence": sequence,
        "resnum": resnum,
        # cg
        "cg_resnum": cg_resnum,
        "cg_cgidx": cg_cgidx,
        "cg_mask": cg_mask,
        # 0
        "cg_T": cg_T,
        "cg_R": cg_R,
        "cg_atom_mask": cg_atom_mask,
        "cg_X": cg_X,
        "cg_amb": cg_amb,
        # alt
        "cg_T_alt": cg_T_alt,  # should always be identical to cg_T
        "cg_R_alt": cg_R_alt,
        "cg_atom_mask_alt": cg_atom_mask_alt,
        "cg_X_alt": cg_X_alt,
        "cg_amb_alt": cg_amb_alt,
        # CG to "PDB" mapping
        "scatter_index": scatter_index,
        "scatter_w": scatter_w,
        "dst_resnum": dst_resnum,
        "dst_atom": dst_atom,
        "dst_resname": dst_resname,
        "final_offset": offset,
        # struct
        "dst_bonds_i1": dst_bonds[0] + dst_idx_offset,  # w/o the offset, incorrect violation loss
        "dst_bonds_i2": dst_bonds[1] + dst_idx_offset,  # w/o the offset, incorrect violation loss
        "dst_bonds_l": dst_bonds[2],
        "dst_bonds_tol": dst_bonds[3],
        "dst_bonds_mask": dst_bonds[4],
        "dst_angles_i1": dst_angles[0]
        + dst_idx_offset,  # w/o the offset, incorrect violation loss
        "dst_angles_i2": dst_angles[1]
        + dst_idx_offset,  # w/o the offset, incorrect violation loss
        "dst_angles_i3": dst_angles[2]
        + dst_idx_offset,  # w/o the offset, incorrect violation loss
        "dst_angles_mid": dst_angles[3],
        "dst_angles_tol": dst_angles[4],
        "dst_atom_widths": dst_atom_widths,
        "dst_atom_mask": dst_atom_mask,
        "aatype": aatype,
        "all_atom_positions": pdb_feats["all_atom_positions"],
        "all_atom_mask": pdb_feats["all_atom_mask"],
    }
        
    return feats


def get_annot_VH_VL(uid, seq, light, dVH=False, numbering='kabat'):
    tmp_file_name = ''.join(random.choice(string.ascii_letters) for i in range(20))
    # heavy
    write_seqs_into_fasta([0], [seq], tmp_file_name+".fasta")
    subprocess.call(
        ["ANARCI", "-i", tmp_file_name+".fasta", "-o", tmp_file_name, "-s", numbering, "-r", "heavy", "--csv"]
    )
    try:
        df_vh = pd.read_csv(tmp_file_name+"_H.csv")
        subprocess.call(["rm", tmp_file_name+"_H.csv"])
        subprocess.call(["rm", tmp_file_name+".fasta"])
        annots = globals()[f'get_annot_{numbering}'](seq, df_vh, "heavy")
        # light
        if light is not None: sys.exit("Need to implement light chain handling properly!")
        # if light is not None:
        #     write_seqs_into_fasta([0], [light], tmp_file_name+".fasta")
        #     if dVH:
        #         subprocess.call("ANARCI -i tmp.fasta -o tmp -s imgt -r heavy --csv", shell=True)
        #         df_vl = pd.read_csv("tmp_H.csv")
        #     else:
        #         subprocess.call("ANARCI -i tmp.fasta -o tmp -s imgt -r light --csv", shell=True)
        #         df_vl = pd.read_csv("tmp_KL.csv")
        #     annots = np.concatenate([annots, get_annot(light, df_vl, "light")])
        #     seq += light
    except Exception as e:
        # ADDED: better print statement so the par doesn't break
        tqdm.write(f"No {tmp_file_name}_H.csv found for {uid}")
        annots = None
    return seq, annots


def get_annot_imgt(seq, df, label):
    # FR1-IMGT: positions 1 to 26, FR2-IMGT: 39 to 55, FR3-IMGT: 66 to 104 and FR4-IMGT: 118 to 128) and of the complementarity determining regions: CDR1-IMGT: 27 to 38, CDR2-IMGT: 56 to 65 and CDR3-IMGT: 105 to 117
    names = [
        f"fwr1_{label}",
        f"cdr1_{label}",
        f"fwr2_{label}",
        f"cdr2_{label}",
        f"fwr3_{label}",
        f"cdr3_{label}",
        f"fwr4_{label}",
    ]
    ranges = [(1, 26), (27, 38), (39, 55), (56, 65), (66, 104), (105, 117), (118, 128)]
    cols = df.columns.tolist()
    annots = []
    s_valid = ""
    for name, rng in zip(names, ranges):
        st = rng[0]
        e = rng[1]
        idx_st = cols.index(str(st))
        idx_e = cols.index(str(e + 1)) if "fwr4" not in name else None
        s = "".join(df.loc[0, cols[idx_st:idx_e]].tolist()).replace("-", "")
        annots.extend([name] * len(s))
        s_valid += s

    assert s_valid == seq[: len(s_valid)] 
    annots.extend([f"C_{label}"] * (len(seq) - len(s_valid)))

    return np.asarray(annots)

def get_annot_kabat(seq, df, label):
    # FR1-KABAT: positions 1 to 25, FR2-KABAT: 36 to 49, FR3-KABAT: 59 to 92 and FR4-KABAT: 103 to 113 and of the complementarity determining regions: CDR1-KABAT: 26 to 36-1, CDR2-KABAT: 50 to 58 and CDR3-KABAT: 93 to 102
    names = [
        f"fwr1_{label}",
        f"cdr1_{label}",
        f"fwr2_{label}",
        f"cdr2_{label}",
        f"fwr3_{label}",
        f"cdr3_{label}",
        f"fwr4_{label}",
    ]
    ranges = [(1, 25), (26, 36-1), (36, 49), (50, 58), (59, 92), (93, 102), (103, 113)]
    cols = df.columns.tolist()
    annots = []
    s_valid = ""
    for name, rng in zip(names, ranges):
        st = rng[0]
        e = rng[1]
        idx_st = cols.index(str(st))

        if "fwr4" not in name:
            
            if "cdr1" in name:
                idx_e = cols.index(str(e+1))
            else:
                idx_e = cols.index(str(e)) + 1
        else:
            idx_e = None

        s = "".join(df.loc[0, cols[idx_st:idx_e]].tolist()).replace("-", "")
        annots.extend([name] * len(s))
        s_valid += s

    assert s_valid == seq[: len(s_valid)]
    annots.extend([f"C_{label}"] * (len(seq) - len(s_valid)))

    return np.asarray(annots)


def write_seqs_into_fasta(seq_ids, seqs, fname):
    with open(fname, "w") as f:
        for seq_id, seq in zip(seq_ids, seqs):
            f.write(f">{seq_id}\n")
            for i in range(0, len(seq), 60):
                seq_chunk = seq[i : min(len(seq), i + 60)]
                f.write(f"{seq_chunk}\n")


def create_weight_fwr_cdr_map_cg_resnum_uv(annot, cg_resnum, weightage=3, num_of_atoms=9, weight_cdr_cdr_pairs=False):
    # (TODO) DEBUG: What does this function do, specifically for cg_resnum? 
    cdr_locs = np.flatnonzero(np.core.defchararray.find(annot, "cdr") != -1)
    weight_fwr_cdr = np.ones([len(annot)])
    weight_fwr_cdr[cdr_locs] = weightage
    weight_fwr_cdr_map_cg_resnum = weight_fwr_cdr[cg_resnum]
    weight_fwr_cdr_map_cg_resnum_uv = np.outer(
        weight_fwr_cdr_map_cg_resnum, weight_fwr_cdr_map_cg_resnum
    )
    np.fill_diagonal(weight_fwr_cdr_map_cg_resnum_uv, 1)

    if weight_cdr_cdr_pairs:
        weight_fwr_cdr_map_cg_resnum_uv = np.where(weight_fwr_cdr_map_cg_resnum_uv > weightage, weightage, weight_fwr_cdr_map_cg_resnum_uv)
    else:
        weight_fwr_cdr_map_cg_resnum_uv = np.where(weight_fwr_cdr_map_cg_resnum_uv > weightage, 1, weight_fwr_cdr_map_cg_resnum_uv)
 
    return torch.tensor(weight_fwr_cdr_map_cg_resnum_uv)


################################### Main #########################
if __name__ == "__main__":

    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", default="./config.ini", help="Location to your global config file")
    args = vars(parser.parse_args())

    CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    print('CONFIG file being used: ', args["config"])
    CONFIG.read(args["config"])

    seq_file_path = CONFIG["preprocessing"]["sequence_path"]
    pdb_path = CONFIG["preprocessing"]["pdb_path"]
    pickle_path = CONFIG["preprocessing"]["pickle_path"]
    
    print(f'The save pickle file will be: {pickle_path}')
    
    output = data_process(CONFIG).process_input(seq_file_path, pdb_path, CONFIG["preprocessing"]["model_type"])
    with open(pickle_path, "ab") as f:
        pickle.dump(output, f)
