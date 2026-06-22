#######################################################################################################################
import os
import numpy as np
from Bio import SeqIO
from Bio.PDB import PDBParser, DSSP
import freesasa
import math
import pickle


def get_one_hot(sequence):
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_dict = {aa: i for i, aa in enumerate(amino_acids)}
    one_hot = np.zeros((len(sequence), 20), dtype=int)
    for i, aa in enumerate(sequence):
        if aa in aa_dict:
            one_hot[i, aa_dict[aa]] = 1
    return one_hot.tolist()


def get_pssm_matrix(sequence, pdb_id, seq_len, PSSM_dir):
    PSSM_matrix = []
    pssm_file_path = os.path.join(PSSM_dir, f"{pdb_id}.pssm")

    if not os.path.isfile(pssm_file_path):
        print(f"[WARNING] PSSM 文件不存在: {pssm_file_path}")
        # return []
        return [[0.0] * 20 for _ in range(seq_len)]

    try:
        with open(pssm_file_path, 'r') as PSSM_file:
            pssm_lines = PSSM_file.readlines()

        pssm_values = []
        for line in pssm_lines[3:]:
            if len(line.split()) > 40:
                values = list(map(float, line.split()[2:22]))
                pssm_values.extend(values)

        if not pssm_values:
            return []

        max_vals = max(pssm_values)
        min_vals = min(pssm_values)

        for idx, residue in enumerate(sequence):
            line = pssm_lines[idx + 3]
            raw_pssm_values = list(map(float, line.split()[2:22]))
            normalized_pssm = [(x - min_vals) / (max_vals - min_vals) for x in raw_pssm_values]
            PSSM_matrix.append(normalized_pssm)

    except Exception as e:
        print(f"[ERROR] 读取或解析 {pdb_id}.pssm 时出错: {e}")
        return []

    return PSSM_matrix


def get_rsa(pdbid, chain, seq_len, pdb_dir="RoBep_187_antigen_chain_pdbs"):
    pdb_path = os.path.join(pdb_dir, f"{pdbid}_{chain}.pdb")
    if not os.path.exists(pdb_path):
        print(f"[WARNING] PDB file not found: {pdb_path}")
        return []

    try:
        structure = freesasa.Structure(pdb_path)
        result = freesasa.calc(structure, freesasa.Parameters({
            'algorithm': freesasa.LeeRichards, 'n-slices': 100, 'probe-radius': 1.4}))
        residue_areas = result.residueAreas()
    except Exception as e:
        print(f"[ERROR] Failed to process {pdbid}_{chain}: {e}")
        return []

    chain_key = ' ' if chain == '*' else chain
    if chain_key not in residue_areas:
        print(f"[WARNING] Chain {chain_key} not found in {pdbid}")
        return []

    rsa_list = []
    for res_key in residue_areas[chain_key].keys():
        area = residue_areas[chain_key][res_key]
        rsa_aa = [
            min(1.0, area.relativeTotal),
            min(1.0, area.relativePolar),
            min(1.0, area.relativeApolar),
            min(1.0, area.relativeMainChain),
            0.0 if math.isnan(area.relativeSideChain) else min(1.0, area.relativeSideChain)
        ]
        rsa_list.append(rsa_aa)

    if len(rsa_list) < seq_len:
        avg_rsa = [sum(x[i] for x in rsa_list) / len(rsa_list) for i in range(5)] if rsa_list else [0.5] * 5
        rsa_list += [avg_rsa] * (seq_len - len(rsa_list))
    elif len(rsa_list) > seq_len:
        rsa_list = rsa_list[:seq_len]

    return rsa_list


def get_side_chain_properties(sequence):
    side_chain_atom_num = {
        'A': 5.0, 'C': 6.0, 'D': 8.0, 'E': 9.0, 'F': 11.0, 'G': 4.0, 'H': 10.0, 'I': 8.0, 'K': 9.0,
        'L': 8.0, 'M': 8.0, 'N': 8.0, 'P': 7.0, 'Q': 9.0, 'R': 11.0, 'S': 6.0, 'T': 7.0, 'V': 7.0,
        'W': 14.0, 'Y': 12.0
    }
    side_chain_charge_num = {
        'A': 0.0, 'C': 0.0, 'D': -1.0, 'E': -1.0, 'F': 0.0, 'G': 0.0, 'H': 1.0, 'I': 0.0, 'K': 1.0,
        'L': 0.0, 'M': 0.0, 'N': 0.0, 'P': 0.0, 'Q': 0.0, 'R': 1.0, 'S': 0.0, 'T': 0.0, 'V': 0.0,
        'W': 0.0, 'Y': 0.0
    }
    side_chain_hbond_num = {
        'A': 2.0, 'C': 2.0, 'D': 4.0, 'E': 4.0, 'F': 2.0, 'G': 2.0, 'H': 4.0, 'I': 2.0, 'K': 2.0, 'L': 2.0,
        'M': 2.0, 'N': 4.0, 'P': 2.0, 'Q': 4.0, 'R': 4.0, 'S': 4.0, 'T': 4.0, 'V': 2.0, 'W': 3.0, 'Y': 3.0
    }
    side_chain_pka = {
        'A': 7.0, 'C': 7.0, 'D': 3.65, 'E': 3.22, 'F': 7.0, 'G': 7.0, 'H': 6.0, 'I': 7.0, 'K': 10.53,
        'L': 7.0, 'M': 7.0, 'N': 8.18, 'P': 7.0, 'Q': 7.0, 'R': 12.48, 'S': 7.0, 'T': 7.0, 'V': 7.0,
        'W': 7.0, 'Y': 10.07
    }
    hydrophobicity = {
        'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9,
        'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2,
        'W': -0.9, 'Y': -1.3
    }

    features = []
    for aa in sequence:
        features.append([
            side_chain_atom_num.get(aa, 0.0),
            side_chain_charge_num.get(aa, 0.0),
            side_chain_hbond_num.get(aa, 0.0),
            side_chain_pka.get(aa, 7.0),  # 默认中性
            hydrophobicity.get(aa, 0.0)
        ])
    return features


def load_fasta_sequences(fasta_path):
    """从 FASTA 文件读取参考序列，返回一个字典 {ID: 序列字符串}"""
    seq_dict = {}
    for record in SeqIO.parse(fasta_path, 'fasta'):
        seq_dict[record.id] = str(record.seq)
    return seq_dict


def compute_dist_adj(pdb_path):
    def dictance(xyz, position):
        xyz = xyz - xyz[position]
        dist = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2 + xyz[:, 2] ** 2)
        return dist.tolist()

    p = PDBParser(QUIET=1)
    structure = p.get_structure("protein", pdb_path)

    ca_coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    ca_coords.append(residue["CA"].get_vector().get_array())

    ca_coords = np.array(ca_coords)
    num_residues = len(ca_coords)

    distance_list = []
    for i in range(num_residues):
        distance_list.append(dictance(ca_coords, i))

    return distance_list


def extract_features_and_save(fasta_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    ref_seq_dict = load_fasta_sequences(fasta_path)

    onehot_dict = {}
    rsa_dict = {}
    dist_dict = {}

    for record in SeqIO.parse(fasta_path, "fasta"):
        header = record.id
        sequence = str(record.seq).upper()
        seq_len = len(sequence)
        pdbid, chain = header.split("_")
        pdbid_chain = f"{pdbid}_{chain}"
        print(f"Processing {pdbid_chain}...")

        # onehot_dict[header] = get_one_hot(sequence)
        # rsa_dict[header] = get_rsa(pdbid, chain, seq_len)

        dist_dict[header] = compute_dist_adj(os.path.join("RoBep_187_antigen_chain_pdbs", f"{pdbid}_{chain}.pdb"))

    # with open(os.path.join(output_folder, "one_hot.pkl"), "wb") as f:
    #     pickle.dump(onehot_dict, f)
    # with open(os.path.join(output_folder, "rsa.pkl"), "wb") as f:
    #     pickle.dump(rsa_dict, f)
    # with open(os.path.join(output_folder, "dist_adj.pkl"), "wb") as f:
    #     pickle.dump(dist_dict, f)

    print(f"All features have been saved to {output_folder}")


# if __name__ == "__main__":
#     extract_features_and_save("../pp-draw/case/RoBep_187.fasta", output_folder="features_RoBep_187(pdb)")
#     # extract_features_and_save("../pp-draw/case_pdb/PDB2526_28.fasta", output_folder="features_PDB2526_28(pdb)")


#######################################################################################################################
import os
import pickle
import numpy as np
from collections import defaultdict
from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_3to1


TEST_FASTA = "RoBep_187.fasta"
TEST_PDB_DIR = "RoBep_187_antigen_chain_pdbs"
TEST_OUT = "features_RoBep_187(pdb)"

RADII = list(range(3, 16))  # 3–15 Å
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY")

propensity_dict = {
    'A': 0.6790,
    'C': 0.6126,
    'D': 1.2877,
    'E': 1.2678,
    'F': 0.8572,
    'G': 0.9487,
    'H': 1.2014,
    'I': 0.7724,
    'K': 1.3499,
    'L': 0.7253,
    'M': 0.9507,
    'N': 1.2183,
    'P': 1.0843,
    'Q': 1.3019,
    'R': 1.5103,
    'S': 0.9584,
    'T': 0.9908,
    'V': 0.6607,
    'W': 1.0233,
    'Y': 1.1598
}


def load_fasta_with_labels(fasta_file):
    seq_dict = {}
    label_dict = {}

    for record in SeqIO.parse(fasta_file, "fasta"):
        seq = str(record.seq)
        seq_dict[record.id] = seq.upper()
        label_dict[record.id] = np.array(
            [1 if aa.isupper() else 0 for aa in seq]
        )

    return seq_dict, label_dict


def load_ca_coordinates(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_file)

    coords = []
    resnames = []

    for model in structure:
        for chain in model:
            for res in chain:
                if "CA" in res:
                    coords.append(res["CA"].get_coord())
                    resnames.append(res.get_resname())

    return np.array(coords), resnames


def res3to1(resname):
    return protein_letters_3to1.get(resname.capitalize(), "X")


def compute_radius_scan_feature(pdb_path, seq_len, propensity):
    coords, resnames = load_ca_coordinates(pdb_path)
    n = len(coords)

    features = np.zeros((n, len(RADII) * 4))

    for i in range(n):
        center = coords[i]
        dists = np.linalg.norm(coords - center, axis=1)

        vec = []
        for r in RADII:
            mask = (dists <= r) & (dists > 0)
            vals = []

            for idx in np.where(mask)[0]:
                aa = res3to1(resnames[idx])
                if aa in propensity:
                    vals.append(propensity[aa])

            if len(vals) == 0:
                vec.extend([0.0, 0.0, 0.0, 0.0])
            else:
                vals = np.array(vals)
                vec.extend([
                    vals.mean(),
                    vals.max(),
                    vals.min(),
                    vals.std()
                ])

        features[i] = vec

    if features.shape[0] < seq_len:
        pad = np.zeros((seq_len - features.shape[0], features.shape[1]))
        features = np.vstack([features, pad])
    return features[:seq_len]


def extract_and_save_radius_feature(fasta_file, pdb_dir, out_dir, propensity):
    os.makedirs(out_dir, exist_ok=True)
    seqs, _ = load_fasta_with_labels(fasta_file)

    radius_dict = {}

    for header, seq in seqs.items():
        if "_" not in header:
            print(f"Skip {header}: invalid format")
            continue

        pdbid, chain = header.split("_")

        target_file = f"{pdbid}_{chain}.pdb".lower()
        real_file = None
        for f in os.listdir(pdb_dir):
            if f.lower() == target_file:
                real_file = f
                break

        if not real_file:
            print(f"Skip {header}, PDB not found")
            continue

        pdb_path = os.path.join(pdb_dir, real_file)
        print(f"Processing {header}...")

        try:
            feat = compute_radius_scan_feature(
                pdb_path,
                len(seq),
                propensity
            )
            radius_dict[header] = feat
        except Exception as e:
            print(f"  Failed: {e}")

    out_path = os.path.join(out_dir, "composition.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(radius_dict, f)

    print(f"\n成功生成 composition.pkl")
    print(f"路径: {out_path}")
    print(f"有效蛋白数量: {len(radius_dict)}")


if __name__ == "__main__":
    print("\n开始提取抗原的倾向特征...")
    extract_and_save_radius_feature(
        fasta_file=TEST_FASTA,
        pdb_dir=TEST_PDB_DIR,
        out_dir=TEST_OUT,
        propensity=propensity_dict
    )


#######################################################################################################################
import os
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from Bio import SeqIO


def normalized_Laplacian(matrix):
    degree_sum = np.array((matrix.sum(1)))
    D_diag = (degree_sum ** -0.5).flatten()
    D_diag[np.isinf(D_diag)] = 0
    D_inv = np.diag(D_diag)
    Lap = D_inv @ matrix @ D_inv
    return Lap


def edge_weight(dist):
    matrix = dist.clone()
    softmax = torch.nn.Softmax(dim=0)
    dist = softmax(1./(torch.log(torch.log(dist+2))))
    dist[matrix > 12] = 0   # 14
    return dist


def get_label_dict_from_fasta(fasta_path):
    label_dict = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        header = record.id
        seq = str(record.seq)
        label = [1 if aa.isupper() else 0 for aa in seq]
        label_dict[header] = label
    return label_dict


def build_graph_dataset(fasta_path, feature_dir, output_path, mode="weighted"):
    assert mode in ["weighted", "unweighted", "upper_weighted", "upper_unweighted", "laplacian_weighted", "laplacian_unweighted"], \
        f"Unknown mode: {mode}"

    label_dict = get_label_dict_from_fasta(fasta_path)

    with open(os.path.join(feature_dir, "one_hot.pkl"), "rb") as f:
        onehot_dict = pickle.load(f)
    with open(os.path.join(feature_dir, "rsa.pkl"), "rb") as f:
        rsa_dict = pickle.load(f)
    with open(os.path.join(feature_dir, "dssp.pkl"), "rb") as f:
        dssp_dict = pickle.load(f)
    with open(os.path.join(feature_dir, "composition.pkl"), "rb") as f:
        composition_dict = pickle.load(f)
    with open(os.path.join(feature_dir, "dist_adj.pkl"), "rb") as f:
        dist_dict = pickle.load(f)

    dataset = []
    for prot_id in label_dict:
        if prot_id not in rsa_dict or prot_id not in dssp_dict or prot_id not in dist_dict:
            print(f"Skipping {prot_id} due to missing features.")
            continue

        rsa_len = len(rsa_dict[prot_id])
        dssp_len = len(dssp_dict[prot_id])
        label_len = len(label_dict[prot_id])

        dist_list = dist_dict[prot_id]
        dist_np = np.array(dist_list)
        dist_len = dist_np.shape[0]

        feature_lengths = [rsa_len, dssp_len, label_len, dist_len]
        if len(set(feature_lengths)) != 1:
            print(
                f"Skipping {prot_id} due to feature length mismatch: RSA={rsa_len}, DSSP={dssp_len}, Label={label_len}, Dist={dist_len}")
            continue

        features = []
        rsa_vals = []

        num_residues = len(rsa_dict[prot_id])
        for i in range(num_residues):
            onehot = onehot_dict[prot_id][i]
            rsa = rsa_dict[prot_id][i]
            dssp = dssp_dict[prot_id][i]
            composition = composition_dict[prot_id][i]

            if i == 0:
                print(f"\n{prot_id}:")
                print(f" - onehot: {len(onehot)}")
                print(f" - rsa: {len(rsa)}")
                print(f" - dssp: {len(dssp)}")
                print(f" - composition: {len(composition)}")
                print(f" - total feature dim: {len(onehot) + len(rsa) + len(dssp) + len(composition)}")

            feat = []
            feat.extend(onehot)
            feat.extend(rsa)
            feat.extend(dssp)
            feat.extend(composition)

            features.append(feat)
            rsa_vals.append(rsa[0])

        pos = np.where(np.array(rsa_vals) >= 0.15)[0].tolist()  # 0  0.15

        if not pos:
            print(f"No exposed residues found in {prot_id}, skipping.")
            continue

        x = torch.tensor(np.array(features)[pos], dtype=torch.float)
        y = torch.tensor(np.array(label_dict[prot_id])[pos], dtype=torch.float)

        # dist
        dist = torch.tensor(dist_dict[prot_id])
        dist = dist[pos, :][:, pos]

        # 构建图结构
        if mode == "weighted":
            dist_edge = edge_weight(dist)
            adj = torch.tensor((dist.numpy() < 12).astype(int), dtype=torch.long)  # 14
            data = Data(x=x, y=y, adj=adj, dist=dist_edge)
        elif mode == "unweighted":
            adj = torch.tensor((dist.numpy() < 14).astype(int), dtype=torch.long)
            dist_edge = torch.ones_like(adj, dtype=torch.float)
            data = Data(x=x, y=y, adj=adj, dist=dist_edge)
        elif mode == "upper_weighted":
            dist_edge = edge_weight(dist)
            triu_mask = torch.triu(torch.ones_like(dist_edge), diagonal=1).bool()
            dist_edge = dist_edge * triu_mask
            adj = (dist_edge > 0).long()
            data = Data(x=x, y=y, adj=adj, dist=dist_edge)
        elif mode == "laplacian_weighted":
            dist_edge = edge_weight(dist)
            adj_np = dist_edge.numpy()
            laplacian = normalized_Laplacian(adj_np)
            adj = torch.tensor(laplacian, dtype=torch.float)
            data = Data(x=x, y=y, adj=adj, dist=dist_edge)
        elif mode == "upper_unweighted":
            adj_np = (dist.numpy() < 14).astype(np.float32)  # 14
            adj_tensor = torch.tensor(adj_np, dtype=torch.float32)
            triu_mask = torch.triu(torch.ones_like(adj_tensor), diagonal=1).bool()
            adj_tensor = adj_tensor * triu_mask
            adj = adj_tensor.long()
            dist_edge = torch.ones_like(adj_tensor)  # dummy edge weight
            data = Data(x=x, y=y, adj=adj, dist=dist_edge)
        elif mode == "laplacian_unweighted":
            adj_np = (dist.numpy() < 14).astype(np.float32)
            laplacian = normalized_Laplacian(adj_np)
            adj = torch.tensor(laplacian, dtype=torch.float)
            data = Data(x=x, y=y, adj=adj)

        data.name = prot_id
        data.POS = pos
        data.length = len(label_dict[prot_id])
        dataset.append(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"Saved dataset with {len(dataset)} samples to {output_path}")


# build_graph_dataset("RoBep_187.fasta", "features_RoBep_187(pdb)", "features_RoBep_187(pdb)/RoBep_187_ordc+weighted_rsa0.15_euclid12.pkl", mode="weighted")

#######################################################################################################################
# # 查看文件内容
# import pickle
#
# with open("features_RoBep_187(pdb)/RoBep_187_ordc+weighted_rsa0.15_euclid12.pkl", "rb") as f:
#     data = pickle.load(f)
#
# wrong_entries = []
#
# for d in data:
#     print(f"蛋白: {d.name}")
#     print(f" - x.shape: {tuple(d.x.shape)}")
#     print(f" - y.shape: {tuple(d.y.shape)}")
#     print(f" - adj.shape: {tuple(d.adj.shape)}")
#     print(f" - dist.shape: {tuple(d.dist.shape)}")
#     print(f" - POS: {len(d.POS)} 个暴露残基")
#     print(f" - 总残基数: {d.length}")
#     print("-" * 50)
#
# for d in data:
#     name = d.name
#     x_len = d.x.shape[0]
#     y_len = d.y.shape[0]
#     expected_len = d.length
#
#     if x_len != y_len:
#         wrong_entries.append((name, x_len, y_len))
#
# print(f"\n存在 x 和 y 长度不一致的样本数量: {len(wrong_entries)}")
#
# for entry in wrong_entries[:5]:
#     print(f"样本: {entry[0]}, x 长度: {entry[1]}, y 长度: {entry[2]}")


