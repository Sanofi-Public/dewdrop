from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from Bio import PDB, Seq, pairwise2
import pandas as pd
import numpy as np
import subprocess
import sys
import shutil
import os
import gzip
from tqdm import tqdm
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from openfold_light.residue_constants import restype_3to1, restype_1to3 # map 3 letters amino acid code to 1 letter code
from cg import cg_dict 

def parse_structure(name, subcid, curated_seq, cid, aligned_sequences, outfp='./data/pdb_1700_raw/', destfp='./data/pdb_1700/'):
    # parse the target file 
    parser = PDB.MMCIFParser(QUIET=True)
    try: # 7evw-assembly1 isn't downloaded so skip with try-except
        structure = parser.get_structure(name, os.path.join(outfp, name+".cif"))
        model = structure[0]

        # extract chain 
        chain = model[cid]    

        # extract both residue objects and their string sequences
        chain_residues = [(restype_3to1[residue.get_resname()], residue) for residue in chain if residue.get_resname() in restype_3to1]
        chain_seq = ''.join([restype_3to1[residue.get_resname()] for residue in chain if residue.get_resname() in restype_3to1])


        # align curacted seq with chain seq
        if chain_seq != curated_seq: 

            # note on parameters: 
            # [seq1, seq2, 
            #  reward given to identical character,
            #  deduction for non-identical character, 
            #  deduction for open gap,
            #  deduction for extending sequence]
            alignment = pairwise2.align.globalms(chain_seq, curated_seq, 5, -3, -1., -.5, gap_char='X')[0]

            # EX: 
            # len(seqA) == len(seqB)
            # chain_seq   = EVQLQESGGGLVYDSLRLSCASSRSIDGINIMRWYRQAPGKQRGMVAVVTGWGSTNYVDSVKGRFIISRDSAKDTVYLQMNNLKPEDTAVYSCNAIYRGSEYWGQGTQVTVS 
            # curated_seq = ESGEMLFTVKKDSLRLSCASSRSIDGINIMRWYRQAPGKQRGMVAVVTGWGSTNYVDSVKGRFIISRDSAKDTVYLQMNNLKPEDTAVYSCNAIYRGSEYWGQGTQVTVS 
            # seqA = EVQLQESGGGXXLXXVYXXDSLRLSCASSRSIDGINIMRWYRQAPGKQRGMVAVVTGWGSTNYVDSVKGRFIISRDSAKDTVYLQMNNLKPEDTAVYSCNAIYRGSEYWGQGTQVTVS
            # seqB = XXXXXESXXGEMLFTVXKKDSLRLSCASSRSIDGINIMRWYRQAPGKQRGMVAVVTGWGSTNYVDSVKGRFIISRDSAKDTVYLQMNNLKPEDTAVYSCNAIYRGSEYWGQGTQVTVS
            aligned_chain_seq = alignment.seqA
            aligned_curated_seq = alignment.seqB

            # Need to re-index the chain_residues according to the aligned chain sequence
            aligned_chain_residues = []
            chain_indx = 0
            for aligned_indx, aligned_restype in enumerate(aligned_chain_seq):
                # (1) inserted unknown residue
                if aligned_restype == 'X':
                    aligned_chain_residues.append((aligned_restype, None)) # No residue assigned to X, will create a new one later 
                # (2) keep original chain residue 
                else: 
                    chain_restype, chain_residue = chain_residues[chain_indx]
                    aligned_chain_residues.append((aligned_restype, chain_residue))
                    chain_indx += 1    

        else: 
            aligned_chain_seq = chain_seq 
            aligned_chain_residues = chain_residues
            aligned_curated_seq = curated_seq


        # construct new structure
        new_structure = PDB.Structure.Structure(id=name[:4])
        new_model = PDB.Model.Model(id=model.id)
        new_chain = PDB.Chain.Chain(id=cid)

        # insert aligned residue according to aligned curated sequence  
        rid = 1 # new residue id 
        for ri, curated_restype in enumerate(aligned_curated_seq): 
            chain_residue = aligned_chain_residues[ri][1]
            chain_restype = aligned_chain_seq[ri]

            # if residue is X or unknown, ignore the residue in the original protein
            if curated_restype == 'X' or curated_restype not in restype_1to3: 
                continue
            # if residue is the same as the original protein, copy the residue
            elif curated_restype == chain_restype:
                new_residue = PDB.Residue.Residue((' ', rid, ' '), restype_1to3[curated_restype], chain_residue.segid)

                # copy the atoms
                for atom in chain_residue:
                    new_residue.add(atom.copy()) 

                # add residue to the chain 
                new_chain.add(new_residue)
                rid += 1
            # if residue is different from the original protein, create a new residue with the curated residue type
            else:
                assert chain_restype == 'X', "Chain residue is not X when curated residue is different but not X."
                new_residue = PDB.Residue.Residue((' ', rid, ' '), restype_1to3[curated_restype], "")

                # common atoms as placeholder
                coord = np.array([np.nan, np.nan, np.nan]) # TODO: change this to the average of the neighboring atoms
                for serial_num, atom_name in enumerate(("C", "CA", "N")):
                    new_atom = PDB.Atom.Atom(name=atom_name, 
                                            coord=coord, 
                                            bfactor=0.0, 
                                            occupancy=1.0, 
                                            altloc=" ",
                                            fullname=f" {atom_name} ", 
                                            serial_number=serial_num,
                                            element=atom_name[0])
                    new_residue.add(new_atom)

                new_chain.add(new_residue)
                rid += 1

        # add new chain to model and model to structure 
        new_model.add(new_chain)
        new_structure.add(new_model)
        
        # save aligned seq to a list 
        aligned_sequences.append(aligned_curated_seq)
        
        # write the curated structure to file 
        cifio = PDB.mmcifio.MMCIFIO()
        cifio.set_structure(new_structure)
        cifio.save(os.path.join(destfp,f'{name}_{subcid}.cif'))

    except FileNotFoundError as e: 
        tqdm.write(f"\nFileNotFound: {name}")



if __name__ == "__main__": 

    csv_default_filepath = "./data/master_pdb_parsed_v2_curated_v4_multichain_Ag_filtered_v2_separated_all_VH_complexes_unique_seq_level_with_seq.csv"
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--csv_file",
                        default=csv_default_filepath, 
                        type=str,
                        help="Relative path to the csv with PDB ids.")
    parser.add_argument("-r", "--raw_dir",
                        default="./data/pdb_1700_raw",
                        type=str, 
                        help="Directory where unprocessed files are stored.")
    parser.add_argument("-o", "--out_dir",
                        default="./data/pdb_1700",
                        type=str, 
                        help="Target output directory.")
    parser.add_argument("-csv", "--csv_name",
                        default="./data/nanobody_1700_unique_seq.csv",
                        type=str, 
                        help="Relative path to the csv with PDB ids.")
    parser.add_argument("-nd", "--no_download",
                        action="store_true",
                        help="Wether to download the file or not, if you don't have the pdb/cif files downloaded, don't add this flag.")
    parser.add_argument("-ne", "--no_extract",
                        action="store_true",
                        help="Wether to extract the chain id specific files or not, if you don't have the pdb/cif files parsed, don't add this flag.")
    parser.add_argument("-ns", "--no_sequence", 
                        action="store_true",
                        help="Wether to generate sequence file or not, if you don't have the csv sequence file parsed, don't add this flag.")
    args = parser.parse_args()
 
    # ======= Get protein uid from the csv file =======
    csvfp = args.csv_file
    df = pd.read_csv(csvfp)

    # get all the files names from "Name"
    names = df["Name"].str.split('>').str[0].to_list() # e.g. "7lx5-assembly1" or "5ku2" 
    # df["Name"] = df["Name"].str.split('>').str[0]
    subchain_id = df["Chain_VH"].to_list() # e.g. "B_0" or "B-2_0"
    sequence_vh = df["Sequence_VH"].to_list()

    # ======= Download all mmCif files with the bash script =======
    if not args.no_download:
        # save target filenames
        raw_dir = "./data/pdb_1700_raw/"
        with open("./data/cache_filenames.txt", "w") as file: 
            file.write(", ".join(names))
        os.makedirs(raw_dir, exist_ok=True)

        tqdm.write(f"Downlaoding files to {raw_dir}.")

        # change permission 
        subprocess.run(["chmod", "+x", "./data/batch_download.sh"], check=True)

        # run the shell script 
        # subprocess.run(["bash", "./data/batch_download.sh", "-f", "./data/cache_filenames.txt", "-o", raw_dir, "-c"], stdout=subprocess.DEVNULL)
        subprocess.run(["./data/batch_download.sh", "-f", "./data/cache_filenames.txt", "-o", raw_dir, "-c"])

        # delete cached files names 
        os.remove("./data/cache_filenames.txt")

        # gzip decompress all files 
        for fp in tqdm(os.listdir(raw_dir), desc="Decompress .gz files..."):
            with gzip.open(os.path.join(raw_dir, fp), "rb") as f_in, \
                open(os.path.join(raw_dir, fp.strip(".gz")), "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            # delete gzip files 
            os.remove(os.path.join(raw_dir, fp))
    
    # ======= Extract target from mmCif files and save them as PDB files =======
    destfp = "./data/pdb_1700/"
    os.makedirs(destfp, exist_ok=True)
    aligned_sequences = []


    # TODO: Improve runtime to use multiprocessing
    pbar = tqdm(names)
    for indx, name in enumerate(pbar): 
        # get the corresponding chain and sequence
        subcid = subchain_id[indx] # e.g. K_0
        curated_seq = sequence_vh[indx] # e.g. QVQLQESGGGLV...
        cid, subid = subcid.split('_')
        parse_structure(name, subcid, curated_seq, cid, aligned_sequences, outfp=args.raw_dir, destfp=destfp)

    # ======= Move sequence into a csv file =======

    df = pd.read_csv(args.csv_file)
    df = df.drop(df.loc[df["Name"].str.contains("7evw-assembly1")].index).reset_index(drop=True) # this one is not working 

    df = df.rename(columns={
                        "Chain_VH": "chain_id",
                        "Name": "uid",
                        "Sequence_VH": "seq",
                    })
    df["uid"] = df["uid"].str.split(">").str[0]
    df.to_csv(args.csv_name, index=None)