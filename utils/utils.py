import math
import shutil
from math import pi

import numpy as np
import torch
from e3nn import nn, o3
from torch_cluster import radius_graph
from torch_scatter import scatter

from openfold_light import residue_constants
from openfold_light.residue_constants import (
    atom_types,
    residue_atoms,
    resnames,
    restype_1to3,
    restype_3to1,
    rigid_group_atom_positions,
)
from utils.refine import refine

PI_div_2 = pi / 2.0


def compose_rotations(R1, R2):
    return torch.einsum("rij,rjk->rik", R1, R2)


def similarity_transform(R, R_update):
    return torch.einsum("rij,rjk,rlk->ril", R_update, R, R_update)


def get_euclidean(pos):
    # pos [N, 3, 3]
    T = pos[:, 1]
    v1 = pos[:, 0] - T
    v2 = pos[:, 2] - T
    R, _ = get_rot_6D(v1, v2)

    return T, R


def compute_d_ijab(X, mask_atom, mask_amb, eps=1e-4):
    """
    ij: CG index
    ab: atom index
    """
    with torch.no_grad():
        d_ijab = ((X[:, None, :, None] - X[None, :, None, :]).square().sum(-1) + eps).sqrt()
        # this mask tells which atoms are present
        mask_atom_ijab = mask_atom[:, None, :, None] * mask_atom[None, :, None, :]
        # this mask is 1 if "ia" is an ambiguous atom and "jb" is nonambiguous
        mask_nonamb = 1 - mask_amb
        # final mask
        mask_ijab = mask_atom_ijab * mask_amb[:, None, :, None] * mask_nonamb[None, :, None, :]

    return (d_ijab, mask_ijab)


def compute_d_ijab_pred(X, eps=1e-4):
    """
    ij: CG index
    ab: atom index
    """
    with torch.no_grad():
        d_ijab = ((X[:, None, :, None] - X[None, :, None, :]).square().sum(-1) + eps).sqrt()

    return d_ijab


def compute_X_uv(
    mask,
    X_v,
    R_v,
    T_v,
    mask_atom,
    X_v_alt,
    R_v_alt,
    T_v_alt,
    mask_atom_alt,
    d_ijab,
    mask_ijab,
    d_ijab_alt,
    mask_ijab_alt,
    d_ijab_pred,
):
    d_i = (mask_ijab * (d_ijab - d_ijab_pred).abs()).sum(dim=[1, 2, 3])
    d_i_alt = (mask_ijab_alt * (d_ijab_alt - d_ijab_pred).abs()).sum(dim=[1, 2, 3])
    ialt = d_i > d_i_alt
    # replace with alt
    if ialt.sum() > 0:
        X_v, R_v, T_v = X_v.clone(), R_v.clone(), T_v.clone()  # TODO: make this more efficient
        X_v[ialt], R_v[ialt], T_v[ialt] = X_v_alt[ialt], R_v_alt[ialt], T_v_alt[ialt]

    # compute ground truth Xuv
    with torch.no_grad():
        X_uv = apply_inverse_euclidean_uv(X_v, R_v, T_v)  # R_v, T_v = R_u, T_u
        mask_u = mask.unsqueeze(1)  # this is needed
        mask_v = mask.unsqueeze(0)
        mask_atom_v = mask_atom.unsqueeze(0)
        mask_atom_uv = (mask_u * mask_v).unsqueeze(-1) * mask_atom_v

    return X_uv, mask_atom_uv


def compute_X_v_pred(X0, R_pred_v, T_pred_v):
    X_v_pred = apply_euclidean(X0, R_pred_v, T_pred_v)

    return X_v_pred


def compute_X_uv_pred(X_v_pred, R_pred_u, T_pred_u):
    X_uv_pred = apply_inverse_euclidean_uv(X_v_pred, R_pred_u, T_pred_u)

    return X_uv_pred


def compute_FAPE_uv(
    X_uv,
    mask_atom_uv,
    X_uv_pred,
    eps=1e-4,
    d_max=10.0,
    Z=10.0,
    weights=None,
    return_count=False,
    scale_only_clamp_value=True,
):
    d_uv = ((X_uv - X_uv_pred).square().sum(-1) + eps).sqrt().clamp(max=d_max)
    if weights is not None:
        if scale_only_clamp_value:
            weights = torch.repeat_interleave(
                torch.unsqueeze(weights, -1), X_uv.size(dim=2), dim=-1
            )
            mask_d_max_scale = torch.where(d_uv >= d_max, 1.0, 0.0)
            weights = weights * mask_d_max_scale
            weights = torch.where(weights == 0.0, 1.0, weights)
            natom_pairs = mask_atom_uv.sum()
        else:
            weights = torch.repeat_interleave(
                torch.unsqueeze(weights, -1), X_uv.size(dim=2), dim=-1
            )
            d_uv = weights * d_uv
            natom_pairs = (weights * mask_atom_uv).sum()
    else:
        natom_pairs = mask_atom_uv.sum()

    if not return_count:
        loss = (d_uv * mask_atom_uv).sum() / natom_pairs / Z
        return loss
    else:
        loss = (d_uv * mask_atom_uv).sum() / Z
        return loss, natom_pairs


def apply_euclidean(x, R, T):
    """
    R [num_nodes, 3, 3]
    T [num_nodes, 3]
    x [num_nodes, Na, 3]
    """
    Rx = torch.einsum("rkl,rml->rmk", R, x)
    return Rx + T.unsqueeze(1)


def apply_inverse_euclidean(x, R, T):
    """
    R [num_nodes, 3, 3]
    T [num_nodes, 3]
    x [num_nodes, Na, 3]
    """
    return torch.einsum("rlk,rml->rmk", R, x - T.unsqueeze(1))


def apply_inverse_euclidean_uv(x_v, R_u, T_u):
    # x_v [N, Na, 3]
    # R_u [N, 3, 3]
    # T_u [N, 3]
    # x_uv [N, N, Na, 3]
    return torch.einsum(
        "uvpq,uvkp->uvkq", R_u.unsqueeze(1), x_v.unsqueeze(0) - T_u[:, None, None, :]
    )


def R_from_quaternion_u(u):
    norm = (1 + u.square().sum(dim=1)).sqrt()
    b, c, d = u.T / norm
    a = 1 / norm

    a2 = a.square()
    b2 = b.square()
    c2 = c.square()
    d2 = d.square()
    bc2 = 2 * b * c
    ad2 = 2 * a * d
    bd2 = 2 * b * d
    ac2 = 2 * a * c
    cd2 = 2 * c * d
    ab2 = 2 * a * b
    # R = [[a**2 + b**2 - c**2 - d**2, 2*b*c - 2*a*d, 2*b*d + 2*a*c],
    #      [2*b*c + 2*a*d, a**2 - b**2 + c**2 - d**2, 2*c*d - 2*a*b],
    #      [2*b*d - 2*a*c, 2*c*d + 2*a*b, a**2 - b**2 - c**2 + d**2]]
    m12 = bc2 - ad2
    m32 = cd2 + ab2
    m22 = a2 - b2 + c2 - d2
    m21 = bc2 + ad2
    m23 = cd2 - ab2
    R = torch.stack(
        [
            torch.stack([a2 + b2 - c2 - d2, m12, bd2 + ac2], dim=1),
            torch.stack([m21, m22, m23], dim=1),
            torch.stack([bd2 - ac2, m32, a2 - b2 - c2 + d2], dim=1),
        ],
        dim=1,
    )

    return R


def get_euclidean_kabsch(pos, ref, pos_mask):
    # pos [N,M,3]
    # ref [N,M,3]
    # pos_mask [N,M]
    # N : number of examples
    # M : number of atoms
    # R,T maps local reference onto global pos
    if pos_mask is None:
        pos_mask = torch.ones(pos.shape[:2], device=pos.device)
    else:
        if pos_mask.shape[0] != pos.shape[0]:
            raise ValueError(
                "pos_mask should have same number of rows as number of input vectors."
            )
        if pos_mask.shape[1] != pos.shape[1]:
            raise ValueError(
                "pos_mask should have same number of cols as number of input vector dimensions."
            )
        if pos_mask.ndim != 2:
            raise ValueError("pos_mask should be 2 dimensional.")

    # Center point clouds
    denom = torch.sum(pos_mask, dim=1, keepdim=True)
    denom[denom == 0] = 1.0
    pos_mu = torch.sum(pos * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
    ref_mu = torch.sum(ref * pos_mask[:, :, None], dim=1, keepdim=True) / denom[:, :, None]
    pos_c = pos - pos_mu
    ref_c = ref - ref_mu

    # Covariance matrix
    H = torch.einsum("bji,bjk->bik", ref_c, pos_mask[:, :, None] * pos_c)

    # DEBUG: Cov. Mat. will become NaN if the input is all zeros 
    # ADDED: Option 1: fill nan to zero
    H = torch.nan_to_num(H, nan=0.0)
    # (Didn't work) Option 2: init UNK reference coord to be none zero, e.g. random (in equifold_process_input.py)

    U, S, Vh = torch.linalg.svd(H)
    # Decide whether we need to correct rotation matrix to ensure right-handed coord system
    locs = torch.linalg.det(U @ Vh) < 0
    S[locs, -1] = -S[locs, -1]
    U[locs, :, -1] = -U[locs, :, -1]
    # Rotation matrix
    R = torch.einsum("bji,bkj->bik", Vh, U)

    # Translation vector
    T = pos_mu - torch.einsum("bij,bkj->bki", R, ref_mu)
    return T.squeeze(1), R


def quaternion_slerp2(R0, R1, t):
    q0 = o3.matrix_to_quaternion(R0)  # returns a unit q
    q1 = o3.matrix_to_quaternion(R1)
    dot = torch.bmm(q0[:, None], torch.transpose(q1[:, None], 1, 2)).squeeze()
    q1[dot < 0.0] = -q1[dot < 0.0]
    dot[dot < 0.0] = -dot[dot < 0.0]

    dot = torch.clamp(dot, -1.0, 1.0)[:, None]
    theta = torch.acos(dot)

    qslerp = (q0 * torch.sin((1 - t) * theta) + q1 * torch.sin(t * theta)) / torch.sin(theta)
    torch._assert(
        torch.all(torch.logical_or(theta == 0, torch.remainder(theta, torch.pi / 2) != 0)),
        "theta which is multiple of pi/2 needs to be handled in quaternion_slerp2",
    )
    torch._assert(
        torch.all(torch.logical_or(theta == 0, torch.remainder(theta, torch.pi) != 0)),
        "At least one theta is multiple of pi in quaternion_slerp2",
    )
    # deal with very small angles, and sin(0)=0
    for k, d in enumerate(dot):
        if d > 0.999:
            # print("performing linear interpolation")
            qslerp[k] = q0[k] * (1 - t) + q1[k] * t

    return o3.quaternion_to_matrix(qslerp)


def quaternion_power2(q, t):
    axis, angle = o3.quaternion_to_axis_angle(q)
    exp_tlnq = torch.cat(
        (torch.cos(t * angle / 2)[:, None], torch.sin(t * angle[:, None] / 2) * axis), 1
    )
    return exp_tlnq


def quaternion_slerp(R0, R1, t):
    """returns rotation matrix"""
    q0 = o3.matrix_to_quaternion(R0)  # returns a unit q
    q1 = o3.matrix_to_quaternion(R1)

    # https://en.wikipedia.org/wiki/Slerp#Quaternion_Slerp
    # use the first formula
    # q0(q0^-1q1)^t
    q0_inv = o3.inverse_quaternion(q0)
    # print(o3.compose_quaternion(q0, q0_inv))
    q0_inv_q1 = o3.compose_quaternion(q0_inv, q1)
    q0_inv_q1_to_t = quaternion_power2(q0_inv_q1, t)
    # print(quaternion_norm(q0_inv_q1_to_t))
    q = o3.compose_quaternion(q0, q0_inv_q1_to_t)

    return o3.quaternion_to_matrix(q)


def compute_struct_loss(
    X, data, eps=1e-6, return_full=False, bond_tol_scale=1.0, apply_mask=False
):
    """
    X [natoms, 3]
    data: single data instance

    if return_full is True, then return full loss gathered by atoms
    """
    dst_bonds_i1 = data["dst_bonds_i1"]
    dst_bonds_i2 = data["dst_bonds_i2"]
    dst_bonds_l = data["dst_bonds_l"]
    dst_bonds_tol = data["dst_bonds_tol"]
    dst_angles_i1 = data["dst_angles_i1"]
    dst_angles_i2 = data["dst_angles_i2"]
    dst_angles_i3 = data["dst_angles_i3"]
    dst_angles_mid = data["dst_angles_mid"]
    dst_angles_tol = data["dst_angles_tol"]
    dst_atom_widths = data["dst_atom_widths"]
    dst_bonds_mask = data["dst_bonds_mask"]
    if apply_mask:
        dst_atom_mask = data["dst_atom_mask"]

    # bond
    l_pred = ((X[dst_bonds_i1] - X[dst_bonds_i2]).square().sum(-1) + eps).sqrt()
    loss_bond = (dst_bonds_l - l_pred).abs() - bond_tol_scale * dst_bonds_tol
    mask_bond = dst_bonds_mask  # mask unambiguous
    # mask missing atoms
    if apply_mask:
        mask_bond = mask_bond * dst_atom_mask[dst_bonds_i1] * dst_atom_mask[dst_bonds_i2]
    loss_bond = loss_bond * mask_bond
    loss_bond = loss_bond.clamp(min=0)
    if return_full:
        # symmeterize
        loss_bond = scatter(loss_bond, dst_bonds_i1, dim=0, dim_size=len(X)) + scatter(
            loss_bond, dst_bonds_i2, dim=0, dim_size=len(X)
        )
        mask_bond = scatter(mask_bond, dst_bonds_i1, dim=0, dim_size=len(X)) + scatter(
            mask_bond, dst_bonds_i2, dim=0, dim_size=len(X)
        )
        loss_bond = loss_bond / mask_bond.clamp(min=1.0)
    else:
        loss_bond = loss_bond.sum() / mask_bond.sum()

    # angle; numerical stability?
    v1 = X[dst_angles_i1] - X[dst_angles_i2]
    v2 = X[dst_angles_i3] - X[dst_angles_i2]
    norm = ((v1.square().sum(-1) + eps) * (v2.square().sum(-1) + eps)).sqrt()
    cosa_pred = (v1 * v2).sum(-1) / norm
    loss_angle = (dst_angles_mid - cosa_pred).abs() - dst_angles_tol
    if apply_mask:
        mask_angle = (
            dst_atom_mask[dst_angles_i1]
            * dst_atom_mask[dst_angles_i2]
            * dst_atom_mask[dst_angles_i3]
        )
        loss_angle = loss_angle * mask_angle
        norm_angle = mask_angle.sum()
    else:
        norm_angle = len(loss_angle)
    loss_angle = loss_angle.clamp(min=0)
    if return_full:
        loss_angle = scatter(loss_angle, dst_angles_i2, dim=0, dim_size=len(X))
    else:
        loss_angle = loss_angle.sum() / norm_angle

    # clash
    d = ((X[:, None] - X[None, :]).square().sum(-1) + eps).sqrt()
    d_min = dst_atom_widths[:, None] + dst_atom_widths[None, :]
    clash_tol = 0.1  # previously, tried 1.5 and then 0.5
    loss_clash = (d_min - d - clash_tol).clamp(min=0)
    # exclude bonds
    mask_clash = (d < 8.0).type(d_min.dtype)  # only consider nearby
    mask_clash[dst_bonds_i1, dst_bonds_i2] = 0.0
    mask_clash[dst_bonds_i2, dst_bonds_i1] = 0.0
    # mask self
    mask_clash.fill_diagonal_(0.0)
    # mask missing
    if apply_mask:
        m = dst_atom_mask == 0  # want to mask out missing so set eq to zero
        mask_clash[m, :] = 0
        mask_clash[:, m] = 0
    loss_clash = loss_clash * mask_clash
    if not return_full:
        loss_clash = loss_clash.sum() / mask_clash.sum()
    else:
        loss_clash = loss_clash.sum(dim=1)
        mask_clash = mask_clash.sum(dim=1)
        loss_clash = loss_clash / mask_clash.clamp(min=1.0)

    return loss_bond, loss_angle, loss_clash


def compute_x_pdb(X_v_pred, scatter_index, scatter_w, natoms):
    X_pred_flat = X_v_pred.reshape(-1, 3) * scatter_w.reshape(-1, 1)
    X_pred_pdb = scatter(X_pred_flat, scatter_index, dim=0, dim_size=natoms)

    return X_pred_pdb


def compute_rmsd(x1, x2, niter=4, retain_frac=0.95, mask=None):
    """
    x1, x2 [N, 3]; torch.Tensors

    returns transformation that bring x1 to x2

    mask applies symmetrically to both
    """
    if retain_frac >= 1.0:
        niter = 1
        retain_frac = 1.0

    x1_, x2_ = x1.clone(), x2.clone()
    if mask is None:
        mask = torch.ones(len(x1_), device=x1_.device)
    mask_ = mask.clone()

    for i in range(niter):
        T, R = get_euclidean_kabsch(
            x2_.reshape(1, -1, 3), x1_.reshape(1, -1, 3), mask_.reshape(1, -1)
        )
        if i < niter - 1:
            x1_shifted = torch.einsum("ij,rj->ri", R.squeeze(0), x1_) + T
            d2 = (x2_ - x1_shifted).square().sum(-1)
            d2_max = torch.quantile(d2, retain_frac)
            isubset = (d2 < d2_max) & (mask_ == 1.0)
            x1_, x2_, mask_ = x1_[isubset], x2_[isubset], mask_[isubset]

    # use the last to transform x1
    x1_shifted = torch.einsum("ij,rj->ri", R.squeeze(0), x1) + T
    d2 = (x2 - x1_shifted).square().sum(-1)
    d2_max = torch.quantile(d2, retain_frac)
    isubset = (d2 < d2_max) & (mask == 1.0)
    rmsd = ((x2[isubset] - x1_shifted[isubset]).square().sum() / isubset.sum()).sqrt()

    return x1_shifted, R.squeeze(0), T.squeeze(0), rmsd


# from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


import os

import numpy as np
import requests
import torch
from IPython.core.debugger import set_trace

from openfold_light.residue_constants import (
    atom_order,
    atom_types,
    res_to_num,
    residue_atoms,
    restype_1to3,
    restypes,
)


def find_alignment_transform(traces):
    centers = traces.mean(-2, keepdim=True)
    traces = traces - centers

    p1, p2 = traces[0], traces[1:]

    C = torch.einsum("i j k, j l -> i k l", p2, p1)
    V, _, W = torch.linalg.svd(C)
    U = torch.matmul(V, W)
    U = torch.matmul(
        torch.stack(
            [
                torch.ones(len(p2), device=U.device),
                torch.ones(len(p2), device=U.device),
                torch.linalg.det(U),
            ],
            dim=1,
        )[:, :, None]
        * V,
        W,
    )

    return torch.cat([torch.eye(3, device=U.device)[None], U]), centers


def atom37_to_pdb(numbered_sequences, all_atoms, chain_ids="HL"):
    atom_index = 0
    pdb_lines = []
    record_type = "ATOM"
    seq = numbered_sequences[chain_ids[0]] + numbered_sequences[chain_ids[1]]
    chain_index = [0] * len(numbered_sequences[chain_ids[0]]) + [1] * len(
        numbered_sequences[chain_ids[1]]
    )
    chain_id = chain_ids[0]

    for i, amino in enumerate(seq):
        res_name_3 = restype_1to3[amino[1]]
        for atom in atom_types:
            if atom in residue_atoms[res_name_3]:
                j = atom_order[atom]
                pos = all_atoms[i, j]
                if pos.mean() != pos.mean():
                    continue
                name = f" {atom}"
                alt_loc = ""

                if chain_id != chain_ids[chain_index[i]]:
                    chain_id = chain_ids[chain_index[i]]
                occupancy = 1.00
                b_factor = 0.00
                element = atom[0]
                charge = ""
                # PDB is a columnar format, every space matters here!
                atom_line = (
                    f"{record_type:<6}{atom_index:>5} {name:<4}{alt_loc:>1}"
                    f"{res_name_3:>3} {chain_id:>1}"
                    f"{(amino[0][0]):>4}{amino[0][1]:>1}   "
                    f"{pos[0]:>8.3f}{pos[1]:>8.3f}{pos[2]:>8.3f}"
                    f"{occupancy:>6.2f}{b_factor:>6.2f}          "
                    f"{element:>2}{charge:>2}"
                )
                pdb_lines.append(atom_line)
                atom_index += 1

    return "\n".join(pdb_lines)


def add_errors_as_bfactors(filename, errors, header=[]):
    with open(filename) as file:
        txt = file.readlines()

    new_txt = header.copy()
    residue_index = -1
    position = "  "

    for line in txt:
        if line[:4] == "ATOM":
            current_res = line[22:27]
            if current_res != position:
                position = current_res
                residue_index += 1
            line = line.replace("  0.00  ", f"{errors[residue_index]:>6.2f}  ")
        elif "REMARK   1 CREATED WITH OPENMM" in line:
            line = line.replace(" 1 CREATED WITH OPENMM", "STRUCTURE REFINED USING OPENMM")
            line = line[:-1] + (81 - len(line)) * " " + "\n"
        new_txt.append(line)

    with open(filename, "w+") as file:
        file.writelines(new_txt)


def to_atom37(X_coord, dst_resnum, dst_atom, dst_resname):
    device = X_coord.get_device()
    if device == -1:
        device = "cpu"
    dst_atom_idx = torch.tensor([residue_constants.atom_order[i] for i in dst_atom.tolist()]).to(
        device
    )

    X_atom37 = []
    mask_atom37 = []
    aatype = []

    for i in range(torch.amax(dst_resnum).item() + 1):
        tmp = torch.where(dst_resnum == i)[0]

        res_shortname = residue_constants.restype_3to1.get(dst_resname[tmp[0]], "X")

        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num
        )

        x = torch.zeros(37, 3).to(device)
        x.index_copy_(0, dst_atom_idx[tmp], X_coord[tmp, :])
        y = torch.zeros(37).to(device)
        y.index_fill_(0, dst_atom_idx[tmp], 1.0)
        X_atom37.append(x)
        mask_atom37.append(y)
        aatype.append(restype_idx)

    X_atom37 = torch.stack(X_atom37).to(device)
    mask_atom37 = torch.stack(mask_atom37).to(device)

    return (
        torch.unsqueeze(X_atom37, dim=0),
        torch.unsqueeze(mask_atom37, dim=0),
        torch.unsqueeze(torch.tensor(aatype), dim=0),
    )


def torsion_angle_loss(a, a_gt, a_alt_gt, mask):
    # [*, N, 7, 2]  # [*, N, 7, 2]  # [*, N, 7, 2]

    # [*, N, 7]
    diff_norm_gt = torch.norm(a - a_gt, dim=-1) * mask
    diff_norm_alt_gt = torch.norm(a - a_alt_gt, dim=-1) * mask
    min_diff = torch.minimum(diff_norm_gt**2, diff_norm_alt_gt**2)

    # [*]
    l_torsion = torch.mean(min_diff, dim=(-1, -2))
    return l_torsion


class compute_prediction_error:
    def __init__(self, numbered_sequences, predictions, refine=True):
        self.numbered_sequences = numbered_sequences
        self.atoms = predictions
        self.refine = refine

        with torch.no_grad():
            traces = torch.stack([x[:, 0] for x in self.atoms])
            self.R, self.t = find_alignment_transform(traces)
            self.aligned_traces = (traces - self.t) @ self.R
            self.error_estimates = (
                (self.aligned_traces - self.aligned_traces.mean(0)).square().sum(-1)
            )
            self.ranking = [x.item() for x in self.error_estimates.mean(-1).argsort()]

    def save_single_unrefined(self, filename, index=0):
        atoms = (self.atoms[index] - self.t[index]) @ self.R[index]
        #         atoms = self.atoms[index]
        unrefined = atom37_to_pdb(self.numbered_sequences, atoms)

        with open(filename, "w+") as file:
            file.write(unrefined)

    def save_all(
        self,
        uid,
        dirname=None,
        filename=None,
        check_for_strained_bonds=True,
        n_threads=-1,
        header="",
    ):
        if dirname is None:
            dirname = "NanoFold_output"
        if filename is None:
            filename = f"{uid}_final_model.pdb"
        os.makedirs(dirname, exist_ok=True)

        for i in range(len(self.atoms)):
            unrefined_filename = os.path.join(
                dirname, f"{uid}_rank{self.ranking.index(i)}_unrefined.pdb"
            )

            self.save_single_unrefined(unrefined_filename, index=i)
            

        np.save(
            os.path.join(dirname, f"{uid}_error_estimates"),
            self.error_estimates.mean(0).cpu().numpy(),
        )
        final_filename = os.path.join(dirname, filename)
        if self.refine:
            refine(
                os.path.join(dirname, f"{uid}_rank0_unrefined.pdb"),
                final_filename,
                check_for_strained_bonds=check_for_strained_bonds,
                n_threads=n_threads,
            )
        else:
            shutil.copy(os.path.join(dirname, f"{uid}_rank0_unrefined.pdb"), final_filename)
        add_errors_as_bfactors(
            final_filename, self.error_estimates.mean(0).sqrt().cpu().numpy(), header=[header]
        )
