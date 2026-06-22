import os
import pickle
import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser


AA_THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def calculate_dihedral(p1, p2, p3, p4):
    """
    手动计算四个原子的二面角（弧度制）
    :param p1-p4: 四个原子的坐标（np.array，形状(3,)）
    :return: 二面角（弧度），无有效原子时返回None
    """
    try:
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)
        p3 = np.array(p3, dtype=np.float64)
        p4 = np.array(p4, dtype=np.float64)

        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3

        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)

        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)

        v2_unit = v2 / np.linalg.norm(v2)

        m1 = np.cross(n1, v2_unit)
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)

        return np.arctan2(y, x)
    except:
        return None


def load_fasta_sequences(fasta_path):
    """从FASTA文件读取参考序列，返回{ID: 序列字符串}字典"""
    seq_dict = {}
    for record in SeqIO.parse(fasta_path, 'fasta'):
        seq_dict[record.id] = str(record.seq)
    return seq_dict


def extract_dssp_with_alignment(pdb_path, pdb_id, chain_id, ref_seq, dssp_exec='mkdssp'):
    """
    提取 DSSP 特征并根据参考序列对齐
    特征构成：4个二面角三角函数值 + 9个二级结构One-Hot
    """
    SS_dict = {'H': 0, 'B': 1, 'E': 2, 'G': 3, 'I': 4, 'T': 5, 'S': 6, '-': 7}

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, pdb_path)
    model = structure[0]

    dssp_matrix = []
    dssp_seq = ""
    dssp_map = {}

    chain = None
    for c in model.get_chains():
        if c.id == chain_id:
            chain = c
            break
    if not chain:
        print(f"Chain {chain_id} not found in PDB file")
        pad = [
            np.sin(360 * np.pi / 180),
            np.sin(360 * np.pi / 180),
            np.cos(360 * np.pi / 180),
            np.cos(360 * np.pi / 180),
        ] + [0]*8 + [1]
        return [pad for _ in range(len(ref_seq))]

    valid_residues = []
    for res in chain.get_residues():
        if res.get_id()[0] == ' ':
            three_letter = res.get_resname()
            if three_letter in AA_THREE_TO_ONE:
                try:
                    res['N']
                    res['CA']
                    res['C']
                    valid_residues.append(res)
                except:
                    continue

    if not valid_residues:
        print(f"Chain {chain_id} has no valid residues with complete backbone atoms")
        pad = [
            np.sin(360 * np.pi / 180),
            np.sin(360 * np.pi / 180),
            np.cos(360 * np.pi / 180),
            np.cos(360 * np.pi / 180),
        ] + [0]*8 + [1]
        return [pad for _ in range(len(ref_seq))]

    for idx, curr_res in enumerate(valid_residues):
        three_letter = curr_res.get_resname()
        aa = AA_THREE_TO_ONE.get(three_letter, 'X')
        dssp_seq += aa

        try:
            curr_N = curr_res['N'].get_coord()
            curr_CA = curr_res['CA'].get_coord()
            curr_C = curr_res['C'].get_coord()
        except:
            phi_val = 360.0
            psi_val = 360.0
            dssp_matrix.append([
                np.sin(phi_val * np.pi / 180),
                np.sin(psi_val * np.pi / 180),
                np.cos(phi_val * np.pi / 180),
                np.cos(psi_val * np.pi / 180),
            ] + [0]*9)
            continue

        phi = None
        if idx > 0:
            prev_res = valid_residues[idx-1]
            try:
                prev_C = prev_res['C'].get_coord()
                phi = calculate_dihedral(prev_C, curr_N, curr_CA, curr_C)
            except:
                phi = None
        phi_val = phi if phi is not None else 360.0

        psi = None
        if idx < len(valid_residues) - 1:
            next_res = valid_residues[idx+1]
            try:
                next_N = next_res['N'].get_coord()
                psi = calculate_dihedral(curr_N, curr_CA, curr_C, next_N)
            except:
                psi = None
        psi_val = psi if psi is not None else 360.0

        vec = [
            np.sin(phi_val * np.pi / 180),
            np.sin(psi_val * np.pi / 180),
            np.cos(phi_val * np.pi / 180),
            np.cos(psi_val * np.pi / 180),
        ]

        if phi_val != 360.0 and psi_val != 360.0:
            if abs(phi_val) < 90 and abs(psi_val) < 90:
                ss = 'H'
            elif abs(phi_val) > 90 and abs(psi_val) > 90:
                ss = 'E'
            elif abs(phi_val - 60) < 30 and abs(psi_val + 30) < 30:
                ss = 'G'
            else:
                ss = '-'
        else:
            ss = '-'
        ss = ss if ss in SS_dict else '-'

        ss_onehot = [0] * 9
        ss_onehot[SS_dict[ss]] = 1
        vec.extend(ss_onehot)

        dssp_matrix.append(vec)
        dssp_map[idx] = vec

    pad = [
        np.sin(360 * np.pi / 180),
        np.sin(360 * np.pi / 180),
        np.cos(360 * np.pi / 180),
        np.cos(360 * np.pi / 180),
    ] + [0]*8 + [1]

    aligned_features = []
    p_dssp = 0
    for i in range(len(ref_seq)):
        if p_dssp < len(dssp_seq) and ref_seq[i] == dssp_seq[p_dssp]:
            aligned_features.append(dssp_matrix[p_dssp])
            p_dssp += 1
        else:
            aligned_features.append(pad)

    return aligned_features


def extract_features_and_save(fasta_path, output_folder, pdb_dataset_dir):
    """
    提取特征并保存为dssp.pkl
    """
    os.makedirs(output_folder, exist_ok=True)
    ref_seq_dict = load_fasta_sequences(fasta_path)
    dssp_dict = {}

    for record in SeqIO.parse(fasta_path, "fasta"):
        header = record.id
        sequence = str(record.seq).upper()
        pdbid, chain = header.split("_")
        pdbid_chain = f"{pdbid}_{chain}"
        print(f"Processing {pdbid_chain}...")

        target_filename = f"{pdbid}_{chain}.pdb".lower()
        real_filename = None
        for f in os.listdir(pdb_dataset_dir):
            if f.lower() == target_filename:
                real_filename = f
                break
        pdb_path = os.path.join(pdb_dataset_dir, real_filename) if real_filename else ""

        header_lower = header.lower()
        ref_seq_dict_lower = {k.lower(): v for k, v in ref_seq_dict.items()}
        ref_seq_valid = header_lower in ref_seq_dict_lower
        pdb_valid = os.path.exists(pdb_path)

        if ref_seq_valid and pdb_valid:
            try:
                dssp_feat = extract_dssp_with_alignment(pdb_path, pdbid, chain, ref_seq_dict_lower[header_lower])
                dssp_dict[header] = dssp_feat
            except Exception as e:
                print(f"DSSP extraction failed {header}: {e}")
        else:
            print(f"Skip {header}, missing reference sequence or PDB file.")

    with open(os.path.join(output_folder, "dssp.pkl"), "wb") as f:
        pickle.dump(dssp_dict, f)

    print(f"All DSSP features (13D) have been saved to {output_folder}")
    print(f"Total valid entries: {len(dssp_dict)}\n")


if __name__ == "__main__":
    BASE_FOLDER = "/home/GCAT-BCE"

    fasta_187 = os.path.join(BASE_FOLDER, "RoBep_187.fasta")
    output_187 = "features_RoBep_187(pdb)"
    pdb_187_dir = os.path.join(BASE_FOLDER, "RoBep_187_antigen_chain_pdbs")

    extract_features_and_save(fasta_187, output_187, pdb_187_dir)


