import dataclasses
import io
import os
import string
import sys
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from data import protein_preprocessing as protein
import torch
from Bio.PDB import PDBParser, MMCIFParser
# pylint: disable=wrong-import-position
from openfold_light import parsers, residue_constants

FeatureDict = Mapping[str, np.ndarray]

TensorDict = Dict[str, torch.Tensor]


def process_pdb(
    pdb_path: str,
    #         alignment_dir: str,
    is_distillation: bool = True,
    chain_id: Optional[str] = None,
    _structure_index: Optional[str] = None,
    alignment_index: Optional[str] = None,
    unique_chain_id: Optional[str] = None
) -> FeatureDict:
    """
    Assembles features for a protein in a PDB file.
    """
    if _structure_index is not None:
        db_dir = os.path.dirname(pdb_path)
        db = _structure_index["db"]
        db_path = os.path.join(db_dir, db)
        fp = open(db_path, "rb")
        _, offset, length = _structure_index["files"][0]
        fp.seek(offset)
        pdb_str = fp.read(length).decode("utf-8")
        fp.close()
    else:
        with open(pdb_path, "r") as f:
            pdb_str = f.read()
    protein_object = protein.from_pdb_string(pdb_str, chain_id, unique_chain_id)
    input_sequence = _aatype_to_str_sequence(protein_object.aatype)
    description = os.path.splitext(os.path.basename(pdb_path))[0].upper()
    pdb_feats = make_pdb_features(protein_object, description, is_distillation=is_distillation)
    return {**pdb_feats}


def make_protein_features(
    protein_object: protein.Protein,
    description: str,
    _is_distillation: bool = False,
) -> FeatureDict:
    pdb_feats = {}
    aatype = protein_object.aatype
    sequence = _aatype_to_str_sequence(aatype)

    # DEBUG
    # print("make_protein_feature: ", protein_object, " \n", description)

    pdb_feats.update(
        make_sequence_features(
            sequence=sequence,
            description=description,
            num_res=len(protein_object.aatype),
        )
    )

    all_atom_positions = protein_object.atom_positions
    all_atom_mask = protein_object.atom_mask

    pdb_feats["all_atom_positions"] = all_atom_positions.astype(np.float32)
    pdb_feats["all_atom_mask"] = all_atom_mask.astype(np.float32)

    pdb_feats["resolution"] = np.array([0.0]).astype(np.float32)
    pdb_feats["is_distillation"] = np.array(1.0 if _is_distillation else 0.0).astype(np.float32)

    return pdb_feats


def make_sequence_features(sequence: str, description: str, num_res: int) -> FeatureDict:
    """Construct a feature dict of sequence features."""
    features = {}
    features["aatype"] = residue_constants.sequence_to_onehot(
        sequence=sequence,
        mapping=residue_constants.restype_order_with_x,
        map_unknown_to_x=True,
    )
    features["between_segment_residues"] = np.zeros((num_res,), dtype=np.int32)
    features["domain_name"] = np.array([description.encode("utf-8")], dtype=np.object_)
    features["residue_index"] = np.array(range(num_res), dtype=np.int32)
    features["seq_length"] = np.array([num_res] * num_res, dtype=np.int32)
    features["sequence"] = np.array([sequence.encode("utf-8")], dtype=np.object_)
    return features


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    # Cartesian coordinates of atoms in angstroms. The atom types correspond to
    # residue_constants.atom_types, i.e. the first three are N, CA, CB.
    atom_positions: np.ndarray  # [num_res, num_atom_type, 3]

    # Amino-acid type for each residue represented as an integer between 0 and
    # 20, where 20 is 'X'.
    aatype: np.ndarray  # [num_res]

    # Binary float mask to indicate presence of a particular atom. 1.0 if an atom
    # is present and 0.0 if not. This should be used for loss masking.
    atom_mask: np.ndarray  # [num_res, num_atom_type]

    # Residue index as used in PDB. It is not necessarily continuous or 0-indexed.
    residue_index: np.ndarray  # [num_res]

    # B-factors, or temperature factors, of each residue (in sq. angstroms units),
    # representing the displacement of the residue from its ground truth mean
    # value.
    b_factors: np.ndarray  # [num_res, num_atom_type]

    # Chain indices for multi-chain predictions
    chain_index: Optional[np.ndarray] = None

    # Optional remark about the protein. Included as a comment in output PDB
    # files
    remark: Optional[str] = None

    # Templates used to generate this protein (prediction-only)
    parents: Optional[Sequence[str]] = None

    # Chain corresponding to each parent
    parents_chain_index: Optional[Sequence[int]] = None


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Takes a PDB string and constructs a Protein object.
    WARNING: All non-standard residue types will be converted into UNK. All
      non-standard atoms will be ignored.
    Args:
      pdb_str: The contents of the pdb file
      chain_id: If None, then the pdb file must contain a single chain (which
        will be parsed). If chain_id is specified (e.g. A), then only that chain
        is parsed.
    Returns:
      A new `Protein` parsed from the pdb contents.
    """
    pdb_fh = io.StringIO(pdb_str)
    # parser = PDBParser(QUIET=True)
    # TODO: Modified to MMCIFParser 
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("none", pdb_fh)
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(f"Only single model PDBs are supported. Found {len(models)} models.")
    model = models[0]

    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            if res.id[2] != " ":
                raise ValueError(
                    f"PDB contains an insertion code at chain {chain.id} and residue "
                    f"index {res.id[1]}. These are not supported."
                )
            res_shortname = residue_constants.restype_3to1.get(res.resname, "X")
            restype_idx = residue_constants.restype_order.get(
                res_shortname, residue_constants.restype_num
            )
            pos = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            res_b_factors = np.zeros((residue_constants.atom_type_num,))
            for atom in res:
                if atom.name not in residue_constants.atom_types:
                    continue  
                pos[residue_constants.atom_order[atom.name]] = atom.coord
                res_b_factors[residue_constants.atom_order[atom.name]] = atom.bfactor
                # ADDED: maunally mask out atom with NaN coord
                if np.all(np.isnan(atom.coord)):
                    mask[residue_constants.atom_order[atom.name]] = 0.0
                    print("Atom mask altered")
                else:  
                    mask[residue_constants.atom_order[atom.name]] = 1.0
            if np.sum(mask) < 0.5:
                # If no known atom positions are reported for the residue then skip it.
                continue
            aatype.append(restype_idx)
            atom_positions.append(pos)
            atom_mask.append(mask)
            residue_index.append(res.id[1])
            chain_ids.append(chain.id)
            b_factors.append(res_b_factors)

    parents = None
    parents_chain_index = None
    if "PARENT" in pdb_str:
        parents = []
        parents_chain_index = []
        chain_id = 0
        for l in pdb_str.split("\n"):
            if "PARENT" in l:
                if not "N/A" in l:
                    parent_names = l.split()[1:]
                    parents.extend(parent_names)
                    parents_chain_index.extend([chain_id for _ in parent_names])
                chain_id += 1

    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {cid: n for n, cid in enumerate(string.ascii_uppercase)}
    chain_index = np.array([chain_id_mapping[cid] for cid in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
        parents=parents,
        parents_chain_index=parents_chain_index,
    )


def _aatype_to_str_sequence(aatype):
    return "".join([residue_constants.restypes_with_x[aatype[i]] for i in range(len(aatype))])


def make_pdb_features(
    protein_object: protein.Protein,
    description: str,
    is_distillation: bool = True,
    confidence_threshold: float = 50.0,
) -> FeatureDict:
    pdb_feats = make_protein_features(protein_object, description, _is_distillation=True)

    if is_distillation:
        high_confidence = protein_object.b_factors > confidence_threshold
        high_confidence = np.any(high_confidence, axis=-1)
        pdb_feats["all_atom_mask"] *= high_confidence[..., None]

    return pdb_feats


def _parse_template_hits(
    self, alignment_dir: str, alignment_index: Optional[Any] = None
) -> Mapping[str, Any]:
    all_hits = {}
    if alignment_index is not None:
        fp = open(os.path.join(alignment_dir, alignment_index["db"]), "rb")

        def read_template(start, size):
            fp.seek(start)
            return fp.read(size).decode("utf-8")

        for name, start, size in alignment_index["files"]:
            ext = os.path.splitext(name)[-1]

            if ext == ".hhr":
                hits = parsers.parse_hhr(read_template(start, size))
                all_hits[name] = hits

        fp.close()
    else:
        for f in os.listdir(alignment_dir):
            path = os.path.join(alignment_dir, f)
            ext = os.path.splitext(f)[-1]

            if ext == ".hhr":
                with open(path, "r") as fp:
                    hits = parsers.parse_hhr(fp.read())
                all_hits[f] = hits

    return all_hits
