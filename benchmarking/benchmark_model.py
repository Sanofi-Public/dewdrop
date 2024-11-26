import configparser
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import csv
import glob
import json
import os
import pickle
import subprocess
import warnings
from copy import deepcopy
from multiprocessing import Pool
from shutil import rmtree

import Bio
import Bio.PDB
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_fasta_plddt(fasta, plddt):
    seq = fasta
    if plddt is not None:
        pass
        # with open(plddt, "rb") as f:
        #     plddt = pickle.load(f)["plddt"]
    else:
        plddt = np.full(len(seq), -1, dtype=np.float32)

    # assumes heavy-U-light
    if "U" in seq:
        seqs = [x for x in seq.split("U") if len(x) > 0]
        lengths = [len(s) for s in seqs]
        starts = [sum(lengths[:i]) + 32 * i if i > 0 else 0 for i in range(len(seqs))]
        ends = [s + l for s, l in zip(starts, lengths)]
        plddts = [plddt[st:e] for st, e in zip(starts, ends)]
    else:
        seqs = [seq]
        plddts = [plddt]

    return seqs, plddts


def get_annot_fasta_plddt(fasta, plddt, dVH=False):
    seqs, plddts = get_fasta_plddt(fasta, plddt)
    seq = seqs[0]
    light = None if len(seqs) == 1 else seqs[1]
    plddt = np.concatenate(plddts)
    seq, annots = get_annot_VH_VL(seq, light, dVH)
    return seq, annots, plddt


def get_annot_VH_VL(seq, light, dVH=False, numbering='kabat'):
    # heavy
    write_seqs_into_fasta([0], [seq], "tmp.fasta")
    
    subprocess.call(
        ["ANARCI", "-i", "tmp.fasta", "-o", "tmp", "-s", numbering, "-r", "heavy", "--csv"]
    )
    df_vh = pd.read_csv("tmp_H.csv")
    annots = globals()[f'get_annot_{numbering}'](seq, df_vh, "heavy")
    
    subprocess.call(["rm", "tmp_H.csv"])
    subprocess.call(["rm", "tmp.fasta"])
    
    # light
    if light is not None:
        write_seqs_into_fasta([0], [light], "tmp.fasta")
        if dVH:
            subprocess.call(f"ANARCI -i tmp.fasta -o tmp -s {numbering} -r heavy --csv", shell=True)
            df_vl = pd.read_csv("tmp_H.csv")
            subprocess.call(["rm", "tmp_H.csv"])
            
        else:
            subprocess.call(f"ANARCI -i tmp.fasta -o tmp -s {numbering} -r light --csv", shell=True)
            df_vl = pd.read_csv("tmp_KL.csv")
            subprocess.call(["rm", "tmp_KL.csv"])
        
        subprocess.call(["rm", "tmp.fasta"])

        annots = np.concatenate([annots, globals()[f'get_annot_{numbering}'](light, df_vl, "light")])
        seq += light

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


def read_fasta(fasta):
    with open(fasta, "r") as f:
        key = "dummy"
        chains = {}
        tmp = []
        for x in f:
            x = x.rstrip()
            if x.startswith(">"):
                chains[key] = "".join(tmp)
                key = x[1:].split()[0][-1]
                tmp = []
            else:
                tmp.append(x)
    # last
    chains[key] = "".join(tmp)
    # Clean up dummy
    del chains["dummy"]

    return chains


def get_CA_atoms(structure):
    atoms = [
        (res["CA"], Bio.PDB.Polypeptide.protein_letters_3to1[res.resname])
        for chain in structure
        for res in chain
    ]
    seq = "".join([x for _, x in atoms])
    atoms = [x for x, _ in atoms]

    return atoms, seq


def match_seqs(ref_seq, pred_seq, seq_str, ref_atoms, plddt, annots):
    """Reference and annotations are being matched to predictions"""
    if ref_seq != pred_seq:
        idx_st = 0
        idx_e = 1
        chunks = []
        while idx_e < len(pred_seq):
            while pred_seq[idx_st:idx_e] in ref_seq and idx_e <= len(pred_seq):
                idx_e += 1
            idx_e -= 1
            found_chunk = pred_seq[idx_st:idx_e]
            st = ref_seq.find(found_chunk)
            chunks.append((st, st + len(found_chunk)))
            idx_st = idx_e
            idx_e += 1
        ref_atoms = np.concatenate([ref_atoms[st:e] for st, e in chunks])
        ref_seq = "".join([ref_seq[st:e] for st, e in chunks])

    if seq_str != pred_seq:
        idx_st = 0
        idx_e = 1
        chunks = []
        while idx_e < len(pred_seq):
            while pred_seq[idx_st:idx_e] in seq_str and idx_e <= len(pred_seq):
                idx_e += 1
            idx_e -= 1
            found_chunk = pred_seq[idx_st:idx_e]
            st = seq_str.find(found_chunk)
            chunks.append((st, st + len(found_chunk)))
            idx_st = idx_e
            idx_e += 1
        annots = np.concatenate([annots[st:e] for st, e in chunks])
        plddt = np.concatenate([plddt[st:e] for st, e in chunks])

    return ref_atoms, ref_seq, annots, plddt


def match_seqs_complex(ref_seq, pred_seq, seq_str, pred_atoms, plddt, annots):
    """predictions and annotations are being matched to reference"""
    if ref_seq != pred_seq:
        idx_st = 0
        idx_e = 1
        chunks = []
        while idx_e < len(ref_seq):
            while ref_seq[idx_st:idx_e] in pred_seq and idx_e <= len(ref_seq):
                idx_e += 1
            idx_e -= 1
            found_chunk = ref_seq[idx_st:idx_e]
            st = pred_seq.find(found_chunk)
            chunks.append((st, st + len(found_chunk)))
            idx_st = idx_e
            idx_e += 1
        pred_atoms = np.concatenate([pred_atoms[st:e] for st, e in chunks])
        annots = np.concatenate([annots[st:e] for st, e in chunks])
        plddt = np.concatenate([plddt[st:e] for st, e in chunks])
        pred_seq = "".join([pred_seq[st:e] for st, e in chunks])

    return pred_atoms, pred_seq, annots, plddt


def align_pdbs(pdb_ref, pdb_pred, seq_str, annots, plddt, ag_only=False, complex_hack=False):
    """
    seq_str, annots, and plddt are
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_ref = Bio.PDB.PDBParser().get_structure("ref", pdb_ref)[0]
        model_pred = Bio.PDB.PDBParser().get_structure("pred", pdb_pred)[0]

    ref_atoms, ref_seq = get_CA_atoms(model_ref)
    pred_atoms, pred_seq = get_CA_atoms(model_pred)
    if complex_hack:
        # for complexes, there are issues with ags missing residues
        pred_atoms, pred_seq, annots, plddt = match_seqs_complex(
            ref_seq, pred_seq, seq_str, pred_atoms, plddt, annots
        )
    else:
        ref_atoms, ref_seq, annots, plddt = match_seqs(
            ref_seq, pred_seq, seq_str, ref_atoms, plddt, annots
        )
    assert ref_seq == pred_seq
    # hack
    if len(annots) < len(pred_seq):
        annots = np.concatenate([annots, ["C_light"] * (len(pred_seq) - len(annots))])
        plddt = np.full(len(ref_seq), -1, dtype=np.float32)
    else:
        annots = annots[: len(pred_seq)]
    if len(plddt) != len(ref_seq):
        plddt = np.full(len(ref_seq), -1, dtype=np.float32)

    # filter to Fv regions
    if ag_only:
        iselect = annots == "Ag"
    else:
        iselect = (annots != "C_light") & (annots != "C_heavy")

    pred_atoms = np.asarray(pred_atoms)[iselect]
    ref_atoms = np.asarray(ref_atoms)[iselect]
    annots_trimmed = annots[iselect]
    plddt_trimmed = plddt[iselect]
    RMSD95_iter, RMSD95, RMSDs, deltas, super_imposer = get_iterative_alignment(
        ref_atoms, pred_atoms, annots_trimmed, niter=5
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        io = Bio.PDB.PDBIO()
        structure_pred = Bio.PDB.PDBParser().get_structure("pred", pdb_pred)
        super_imposer.apply(structure_pred[0].get_atoms())
        io.set_structure(structure_pred)
        pdb_out_filename = pdb_pred.replace(".pdb", "_aligned.pdb")
        io.save(pdb_out_filename)

    return RMSD95_iter, RMSD95, RMSDs, pdb_out_filename, (deltas, annots_trimmed, plddt_trimmed)


def get_deltas(ref_atoms, pred_atoms):
    deltas = []
    for a1, a2 in zip(ref_atoms, pred_atoms):
        deltas.append(np.sum(np.square(a1.get_coord() - a2.get_coord())))
    deltas = np.asarray(deltas)

    return deltas


def get_RMSDs(ref_atoms, pred_atoms, return_deltas=False):
    # Find 95% with least mistakes
    deltas = get_deltas(ref_atoms, pred_atoms)
    idxs = np.where(deltas < np.percentile(deltas, 95))[0]
    # assert False
    RMSD95 = np.sqrt(np.mean(deltas[idxs]))
    RMSD = np.sqrt(np.mean(deltas))

    if return_deltas:
        return RMSD, RMSD95, idxs, deltas
    else:
        return RMSD, RMSD95, idxs


def get_iterative_alignment(ref_atoms, pred_atoms, annots, niter=5):
    """From AF2: We also report accuracies by the RMSD95 (Cα RMSD at 95% coverage).
    We perform 5 iterations of (a) least-squares alignment of the predicted structure
    and the PDB structure on the currently chosen Cα atoms (using all Cα atoms at the
    first iteration); (b) selecting 95% Cα atoms with the lowest alignment error.
    The root mean squared deviation (RMSD) of the atoms chosen at the final iterations
    is the RMSD95. This metric is more robust to apparent errors that can originate
    from crystal structure artefacts, though of course in some cases the removed 5%
    of residues will contain genuine modelling errors.
    """
    ref_atoms_ = np.asarray(deepcopy(ref_atoms))
    pred_atoms_ = np.asarray(deepcopy(pred_atoms))

    for l in range(niter):
        # Align these paired atom lists:
        super_imposer = Bio.PDB.Superimposer()
        super_imposer.set_atoms(ref_atoms_, pred_atoms_)
        pred_atoms__ = deepcopy(
            pred_atoms_
        )  # make a copy so that we retain original global frame for pred
        super_imposer.apply(pred_atoms__)

        RMSD, RMSD95, idxs = get_RMSDs(ref_atoms_, pred_atoms__)
        if l == 0:
            RMSD95o = RMSD95

        ref_atoms_ = ref_atoms_[idxs]
        pred_atoms_ = pred_atoms_[idxs]

    super_imposer.apply(pred_atoms)
    _, RMSD95_iter, _, deltas = get_RMSDs(ref_atoms, pred_atoms, return_deltas=True)
    df_ = pd.DataFrame({"delta": deltas, "annot": annots})
    RMSDs = df_.groupby("annot").apply(lambda x: np.sqrt(np.mean(x["delta"])))

    return RMSD95_iter, RMSD95o, RMSDs, deltas, super_imposer


def get_rsmds_two_structs(pdb_ref, pdb_pred, seq_str, annots, plddt, complex_hack=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model_ref = Bio.PDB.PDBParser().get_structure("ref", pdb_ref)[0]
        model_pred = Bio.PDB.PDBParser().get_structure("pred", pdb_pred)[0]

    ref_atoms, ref_seq = get_CA_atoms(model_ref)
    pred_atoms, pred_seq = get_CA_atoms(model_pred)
    if complex_hack:
        # for complexes, there are issues with ags missing residues
        pred_atoms, pred_seq, annots, plddt = match_seqs_complex(
            ref_seq, pred_seq, seq_str, pred_atoms, plddt, annots
        )
    else:
        ref_atoms, ref_seq, annots, plddt = match_seqs(
            ref_seq, pred_seq, seq_str, ref_atoms, plddt, annots
        )
    assert ref_seq == pred_seq

    iselect = (annots != "C_light") & (annots != "C_heavy")
    pred_atoms = np.asarray(pred_atoms)[iselect]
    ref_atoms = np.asarray(ref_atoms)[iselect]
    annots_trimmed = annots[iselect]
    plddt_trimmed = plddt[iselect]

    _, RMSD95_iter, _, deltas = get_RMSDs(ref_atoms, pred_atoms, return_deltas=True)
    df_ = pd.DataFrame({"delta": deltas, "annot": annots_trimmed})
    RMSDs = df_.groupby("annot").apply(lambda x: np.sqrt(np.mean(x["delta"])))

    return RMSDs, deltas, plddt_trimmed, annots_trimmed


def absoluteFilePaths(directory):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            yield os.path.abspath(os.path.join(dirpath, f))


if __name__ == "__main__":
    
    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--config", default="./config.ini", help="Location to your global config file")
    args = vars(parser.parse_args())

    CONFIG = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    print('CONFIG file being used: ', args["config"])
    CONFIG.read(args["config"])
    
    rmsd_cols = [
        "cdr1_heavy",
        "cdr2_heavy",
        "cdr3_heavy",
        "fwr1_heavy",
        "fwr2_heavy",
        "fwr3_heavy",
        "fwr4_heavy",
        "cdr1_light",
        "cdr2_light",
        "cdr3_light",
        "fwr1_light",
        "fwr2_light",
        "fwr3_light",
        "fwr4_light",
    ]
    pdbs_ref = []
    pdbs_pred = []
    fastas_included = []
    csv_path = CONFIG["benchmarking"]["sequence_path"]
    # csv_path = "./ckpt_and_data/7_5_nb/equifold_int_nb_test_input_59_7_3.csv"

    with open(csv_path) as fd:
        reader = csv.reader(fd, delimiter="\t", quotechar='"')
        next(reader, None)
        for i, row in enumerate(reader):
            row = row[0].split(",")
            fastas_included.append(row[1])

            ground_truth_pdb_path = CONFIG["benchmarking"]["ground_truth_pdb_path"]
            pdbs_ref.append(os.path.join(ground_truth_pdb_path, row[0] + "_" + row[2] + ".pdb"))

            # fine-tuned pdbs
            model_pred_path = CONFIG["benchmarking"]["predicted_pdb_path"]
            pdbs_pred.append(
                os.path.join(model_pred_path, row[0] + "_" + row[2] + "_final_model.pdb")
            )

    table = []
    for pdb_ref, pdb_pred, fasta in tqdm(
        zip(pdbs_ref, pdbs_pred, fastas_included), total=len(pdbs_ref)
    ):
        seq_str, annots, plddt = get_annot_fasta_plddt(fasta, None, dVH="2xVH" in pdb_ref)
        RMSD95_iter, RMSD95o, RMSDs, pdb_out_filename, rs = align_pdbs(
            pdb_ref, pdb_pred, seq_str, annots, plddt
        )
        name = os.path.basename(pdb_pred)[0:-9]

        table.append(
            [name, pdb_ref, fasta, RMSD95o, RMSD95_iter, pdb_out_filename]
            + [RMSDs.get(x, -1) for x in rmsd_cols]
        )

    prefix = CONFIG["benchmarking"]["prefix"]
    df_ = pd.DataFrame(
        table,
        columns=[
            "name",
            "ref pdb",
            "input fasta",
            "RMSD95 CA",
            "RMSD95 CA iter",
            "pred pdb aligned",
        ]
        + [f"RSMD_{x}" for x in rmsd_cols],
    )
    df_.to_csv(
        os.path.join(CONFIG["benchmarking"]["predicted_pdb_path"], f"{prefix}_benchmarks.csv"),
        index=False,
    )
