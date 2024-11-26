import tqdm
import pickle
import os
import requests 
import zipfile
import Bio.PDB.DSSP as DSSP  
from Bio.PDB import PDBParser, MMCIFParser, MMCIFIO
import shutil
import matplotlib.pyplot as plt 
import seaborn as sns
# import cg as CG
import numpy as np 
import pandas as pd
from collections import defaultdict
from openfold_light import residue_constants

root = os.path.join("data", "ATLAS")

def download_protein(uid, chain, rmsf=True, rmsd=True, bfactor=True, plddt=True):
    """
    Equivalent to calling the api: 
    Example: To get uid=16pk, chain=A, run "curl -X 'GET' \
              'https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/protein/16pk_A' \
              -H 'accept: */*'""
    """
    header = {
        "accept": "*/*"
    }
    url = f"https://www.dsimb.inserm.fr/ATLAS/api/ATLAS/analysis/{uid}_{chain}"
    r = requests.get(url, headers=header)
    # write to local file 
    compressed_file_name = os.path.join(root, f"{uid}_{chain}.zip")
    if r.status_code == 200:
        with open(compressed_file_name, 'wb') as f:
            f.write(r.content)
        extracted_folder_name = compressed_file_name.replace('.zip', '')
        with zipfile.ZipFile(compressed_file_name, 'r') as zip_ref:
            zip_ref.extractall(extracted_folder_name)
        os.remove(compressed_file_name)
        
    # (TODO) retain only the target files
    print(f"Downloaded {uid}_{chain}")
    
def plot_terminal_rmsf(target_uid, target_chain):
    # define all relevent fp
    target_root_fp = os.path.join("data", "ATLAS", target_uid+'_'+target_chain)
    bfactor_fp = os.path.join(target_root_fp, f"{target_uid}_{target_chain}_Bfactor.tsv")
    rmsd_fp = os.path.join(target_root_fp, f"{target_uid}_{target_chain}_RMSD.tsv")
    rmsf_fp = os.path.join(target_root_fp, f"{target_uid}_{target_chain}_RMSF.tsv")
    bfactor_df = pd.read_csv(bfactor_fp, sep='\t')
    rmsd_df = pd.read_csv(rmsd_fp, sep='\t')
    rmsf_df = pd.read_csv(rmsf_fp, sep='\t')
    # Assuming df1, df2, and df3 are your dataframes
    df_merged = pd.concat([rmsf_df, rmsd_df, bfactor_df], axis=1, join='inner')
    # use the first 5 and the last 5 as the N-terminal residue and C-terminal residue
    n_terminal = df_merged.iloc[:5, :]
    c_terminal = df_merged.iloc[-5:, :]
    non_terminal = df_merged.iloc[5:-5, :]

    bins = np.arange(0.0, 1.0, 0.05)
    plt.figure()
    plt.title(f"avg. RMSF in  {target_uid}_{target_chain}")
    plt.hist(non_terminal[['RMSF_R1','RMSF_R2','RMSF_R3']].mean(axis=1), density=True, bins=bins, label="non-terminal", alpha=0.9)
    plt.hist(n_terminal[['RMSF_R1','RMSF_R2','RMSF_R3']].mean(axis=1), density=True, bins=bins, label="N-terminal", alpha=0.7)
    plt.hist(c_terminal[['RMSF_R1','RMSF_R2','RMSF_R3']].mean(axis=1), density=True, bins=bins, label="C-terminal", alpha=0.7)
    plt.legend()
#     plt.show()
    
def plot_terminal_rmsd(target_uid, target_chain):
    # define all relevent fp
    target_root_fp = os.path.join("data", "ATLAS", target_uid+'_'+target_chain)
    bfactor_fp = os.path.join(target_root_fp, f"{target_uid}_{target_chain}_Bfactor.tsv")
    rmsd_fp = os.path.join(target_root_fp, f"{target_uid}_{target_chain}_RMSD.tsv")
    rmsf_fp = os.path.join(target_root_fp, f"{target_uid}_{target_chain}_RMSF.tsv")
    bfactor_df = pd.read_csv(bfactor_fp, sep='\t')
    rmsd_df = pd.read_csv(rmsd_fp, sep='\t')
    rmsf_df = pd.read_csv(rmsf_fp, sep='\t')
    # Assuming df1, df2, and df3 are your dataframes
    df_merged = pd.concat([rmsf_df, rmsd_df, bfactor_df], axis=1, join='inner')
    # use the first 5 and the last 5 as the N-terminal residue and C-terminal residue
    n_terminal = df_merged.iloc[:5, :]
    c_terminal = df_merged.iloc[-5:, :]
    non_terminal = df_merged.iloc[5:-5, :]

    bins = np.arange(0.0, max(df_merged[['RMSD_R1','RMSD_R2','RMSD_R3']].mean(axis=1)), 0.05)
    plt.figure()
    plt.title(f"avg. RMSD in  {target_uid}_{target_chain}")
    plt.hist(non_terminal[['RMSD_R1','RMSD_R2','RMSD_R3']].mean(axis=1), density=True, bins=bins, label="non-terminal", alpha=0.9)
    plt.hist(n_terminal[['RMSD_R1','RMSD_R2','RMSD_R3']].mean(axis=1), density=True, bins=bins, label="N-terminal", alpha=0.7)
    plt.hist(c_terminal[['RMSD_R1','RMSD_R2','RMSD_R3']].mean(axis=1), density=True, bins=bins, label="C-terminal", alpha=0.7)
    plt.legend()
#     plt.show()
    
def extract_df(target_uid, target_chain): 
    # define all relevent fp
    target_root_fp = os.path.join("data", "ATLAS", target_uid+'_'+target_chain)
    bfactor_fp = os.path.join(target_root_fp, f"{target_uid}_{target_chain}_Bfactor.tsv")
    rmsd_fp = os.path.join(target_root_fp, f"{target_uid}_{target_chain}_RMSD.tsv")
    rmsf_fp = os.path.join(target_root_fp, f"{target_uid}_{target_chain}_RMSF.tsv")
    bfactor_df = pd.read_csv(bfactor_fp, sep='\t')
    rmsd_df = pd.read_csv(rmsd_fp, sep='\t')
    rmsf_df = pd.read_csv(rmsf_fp, sep='\t')
    # Assuming df1, df2, and df3 are your dataframes
    df_merged = pd.concat([rmsf_df, rmsd_df, bfactor_df], axis=1, join='inner')
    return df_merged

def extract_coord(target_uid, target_chain): 
    target_root_fp = os.path.join("data", "ATLAS", target_uid+'_'+target_chain)
    pdb_fp = os.path.join(target_root_fp, f"{target_uid}_{target_chain}.pdb")
    parser = PDBParser()
    struct = parser.get_structure(target_uid+"_"+target_chain, pdb_fp)
    calpha_coords = []
    model = struct[0]
    for chain in model:
        for res in chain:
            if 'CA' in res:
                calpha_atom = res['CA']
                calpha_coords.append(calpha_atom.get_coord())
    return calpha_coords
    
def gather_pdb(atlas_root, target_fp, save_cif=True):
    # move all pdb files to a new folder 
    if not os.path.exists(target_fp):
        os.makedirs(target_fp)
    folders = os.listdir(atlas_root)
    for folder in folders:
        if not os.path.isfile(os.path.join(atlas_root, folder)) and not folder == ".ipynb_checkpoints":
            files = os.listdir(os.path.join(atlas_root, folder))
            for file in files: 
                file = os.path.join(atlas_root, folder, file)
                if os.path.splitext(file)[1] == '.pdb': 
                    print(f"move {file}")
                    file_name = os.path.basename(file)
                    destination_path = os.path.join(target_fp, file_name)
                    shutil.copy2(file, destination_path)
                    
                    if save_cif: 
                        parser = PDBParser(QUIET=True)
                        structure = parser.get_structure('structure', destination_path)
                        for chain in structure[0]:
                            chain.id = os.path.splitext(file)[0].split('_')[-1]
                        io = MMCIFIO()
                        io.set_structure(structure)
                        io.save(destination_path.replace('.pdb', '.cif'))

def generate_sequence_csv(pdb_root, cif=True): 
    df_dict = {"uid":[], "seq":[], "chain_id":[]}
    for fp in os.listdir(pdb_root):
        if os.path.isfile(os.path.join(pdb_root, fp)) and (fp[-3:] == 'cif'): 
            uid, chain_id = (fp[:-4]).split('_')
            df_dict['uid'].append(uid)
            df_dict['chain_id'].append(chain_id)
            seq = []
            if cif: 
                parser = MMCIFParser(QUIET=True)
            else: 
                parser = PDBParser(QUIET=True)
            struct = parser.get_structure(uid+chain_id, os.path.join(pdb_root, fp))
            for model in struct:
                for chain in model: 
                    for res in chain: 
                        seq.append(res.get_resname())
            df_dict['seq'].append(''.join(seq))
    df = pd.DataFrame(df_dict)
    return df

def to_atom37(X_coord, dst_resnum, dst_atom, dst_resname):
    dst_atom_idx = np.array([residue_constants.atom_order[i] for i in dst_atom])

    X_atom37 = []
    mask_atom37 = []
    aatype = []
    
    dst_resnum = dst_resnum.cpu().numpy()
    
    for i in range(np.max(dst_resnum) + 1):
        tmp = np.where(dst_resnum == i)[0]
        
        res_shortname = residue_constants.restype_3to1.get(dst_resname[tmp[0]], "X")

        restype_idx = residue_constants.restype_order.get(
            res_shortname, residue_constants.restype_num
        )

        x = np.zeros((37, 3))
        x[dst_atom_idx[tmp], :] = X_coord[tmp, :]
        y = np.zeros(37)
        
        y[dst_atom_idx[tmp]] = 1.0
        X_atom37.append(x)
        mask_atom37.append(y)
        aatype.append(restype_idx)

    X_atom37 = np.stack(X_atom37)
    mask_atom37 = np.stack(mask_atom37)

    return (
        np.expand_dims(X_atom37, axis=0),
        np.expand_dims(mask_atom37, axis=0),
        np.expand_dims(np.array(aatype), axis=0),
    )

def get_chunckwise_statistics(ensembles):
    """
    Give a couple ensembles, calculate the partial statistics. 
    :param ensemble: np.array[dict]
    :return: Tuple(List[np.array], List[np.array])
    """
    # modify data structure
    X_ensemble = []
    dst_resnum_ensemble = []
    dst_resname_ensemble = []
    dst_atom_ensemble = []
    x_ensemble = []
    for ensemble in ensembles: 
        X_preds = []
        dst_resnums = []
        dst_resnames = []
        dst_atoms = []
        x_preds = []
        for data in ensemble: 
            X_preds.append(data['X_pred'])
            dst_resnums.append(data['dst_resnum'])
            dst_resnames.append(data['dst_resname'])
            dst_atoms.append(data['dst_atom'])
            x_preds.append(data['x_pred'])
        X_ensemble.append(X_preds)
        dst_resnum_ensemble.append(dst_resnums)
        dst_resname_ensemble.append(dst_resnames)
        dst_atom_ensemble.append(dst_atoms)
        x_ensemble.append(x_preds)
        
    # flatten the batch and change tensor to numpy array 
    X_numpy = []
    # keep ensemble dimension 
    for ensemble in tqdm.tqdm(x_ensemble, leave=False):
        X_numpy_flat = []
        # flatten the next two 
        for i, batch in enumerate(ensemble):
            for j, data in enumerate(batch): 
                # get the last block output and change it to numpy array 
                pred = data[-1].cpu().numpy()
                all_atom_position, _, _ = to_atom37(
                    pred, 
                    dst_resnum_ensemble[i][j][0], 
                    dst_atom_ensemble[i][j][0], 
                    dst_resname_ensemble[i][j][0]
                   )
                X_numpy_flat.append(all_atom_position)
        X_numpy.append(X_numpy_flat)
    
    # switch protein and ensemble dimension (numProtein, ensembleSize)
    ensemble_size, num_protein = len(X_numpy), len(X_numpy[0])
    X_numpy = [[X_numpy[dim1][dim2] for dim1 in range(ensemble_size)] for dim2 in range(num_protein)]
    
    # calculate varaince at each position as an uncertainty measure 
    # out shape = (numProtein, atom, coordinates)
    X_numpy = [np.stack(data) for data in X_numpy]
    X_sum = [np.sum(data, axis=0) for data in X_numpy]
    X_sum_squared = [np.sum(data**2, axis=0) for data in X_numpy]
    return X_sum, X_sum_squared

def get_uid_chain(ensembles): 
    uid_ensemble, chain_ensemble = [], []
    for ensemble in ensembles: 
        uids, chain_ids = [], []
        for data in ensemble: 
            uids.append(data['uid'])
            chain_ids.append(data['chain_id'])
        uid_ensemble.append(uids)
        chain_ensemble.append(chain_ids)
    return uid_ensemble, chain_ensemble

def batched(iterable, n):
    """
    Yield successive n-sized chunks from iterable.
    
    Parameters:
        iterable: An iterable to split into chunks.
        n: The size of each chunk.
        
    Yields:
        Chunks of the iterable of size n.
    """
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]