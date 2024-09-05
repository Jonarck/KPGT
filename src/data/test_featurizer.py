from functools import partial
from itertools import permutations
from typing import Any, List

import networkx as nx
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import BondStereo, BondType, ChiralType, HybridizationType
from torch_geometric.data import Data


### basic feature generator
class FeatureGenerator:
    def __init__(self, feature_fns: List):
        self.feature_fns, self.feature_choice = list(zip(*feature_fns))
        self.feature_length = sum(self.feature_choice)

    def __call__(self, input: Any):
        features = []
        for fn in self.feature_fns:
            features.append(fn(input))
        features = np.array(features)
        return features

    def __repr__(self):
        fn_names = [fn.__name__ for fn in self.feature_fns]
        return f"{self.__class__.__name__}({fn_names})"


def safe_index(lst: list, e: Any) -> int:
    try:
        return lst.index(e)
    except ValueError:
        return len(lst) - 1


def safe_onehot(lst: list, e: Any) -> List[int]:
    try:
        return [int(i == e) for i in lst]
    except ValueError:
        return [0] * (len(lst) - 1) + [1]


### atom features
# TODO: use real value rather than one-hot encoding for some of the atomic features
def atom_atomic_number(
    atom: Chem.Atom, onehot=False, allowed_values=list(range(118)) + ["misc"]
):  # length: 119
    if onehot:
        return safe_onehot(allowed_values, atom.GetAtomicNum())
    return safe_index(allowed_values, atom.GetAtomicNum())


def atom_chirality(
    atom: Chem.Atom,
    allowed_values=[
        ChiralType.CHI_UNSPECIFIED,
        ChiralType.CHI_TETRAHEDRAL_CW,
        ChiralType.CHI_TETRAHEDRAL_CCW,
        ChiralType.CHI_OTHER,
        "misc",
    ],
):  # length: 5
    return safe_index(allowed_values, atom.GetChiralTag())


def atom_degree(
    atom: Chem.Atom, allowed_values=list(range(11)) + ["misc"]
):  # length: 12
    return safe_index(allowed_values, atom.GetTotalDegree())


def atom_formal_charge(
    atom: Chem.Atom, allowed_values=list(range(-5, 6)) + ["misc"]
):  # length: 12
    return safe_index(allowed_values, atom.GetFormalCharge())


def atom_num_h(atom: Chem.Atom, allowed_values=list(range(9)) + ["misc"]):  # length: 10
    return safe_index(allowed_values, atom.GetTotalNumHs())


def atom_num_radical_e(
    atom: Chem.Atom, allowed_values=list(range(5)) + ["misc"]
):  # length: 6
    return safe_index(allowed_values, atom.GetNumRadicalElectrons())


def atom_hybridization(
    atom: Chem.Atom,
    allowed_values=[
        HybridizationType.SP,
        HybridizationType.SP2,
        HybridizationType.SP2D,
        HybridizationType.SP3,
        HybridizationType.SP3D,
        HybridizationType.SP3D2,
        "misc",
    ],
):  # length: 7
    return safe_index(allowed_values, atom.GetHybridization())


def atom_is_aromatic(atom: Chem.Atom, allowed_values=[False, True]):  # length: 2
    return safe_index(allowed_values, atom.GetIsAromatic())


def atom_is_in_ring(atom: Chem.Atom, allowed_values=[False, True]):  # length: 2
    return safe_index(allowed_values, atom.IsInRing())


AtomFeature = FeatureGenerator(
    [
        (atom_atomic_number, 119),
        (atom_chirality, 5),
        (atom_degree, 12),
        (atom_formal_charge, 12),
        (atom_num_h, 10),
        (atom_num_radical_e, 6),
        (atom_hybridization, 7),
        (atom_is_aromatic, 2),
        (atom_is_in_ring, 2),
    ]
)  # length: 174


### bond features
def bond_type(
    bond: Chem.Bond,
    allowed_values=[
        BondType.SINGLE,
        BondType.DOUBLE,
        BondType.TRIPLE,
        BondType.AROMATIC,
        "misc",
    ],
):  # length: 5
    return safe_index(allowed_values, bond.GetBondType())


def bond_is_conjugated(bond: Chem.Bond, allowed_values=[True, False]):  # length: 2
    return safe_index(allowed_values, bond.GetIsConjugated())


def bond_is_in_ring(bond: Chem.Bond, allowed_values=[True, False]):  # length: 2
    return safe_index(allowed_values, bond.IsInRing())


def bond_stereo(
    bond: Chem.Bond,
    allowed_values=[
        BondStereo.STEREONONE,
        BondStereo.STEREOZ,
        BondStereo.STEREOE,
        BondStereo.STEREOCIS,
        BondStereo.STEREOTRANS,
        BondStereo.STEREOANY,
        "misc",
    ],
):  # length: 7
    return safe_index(allowed_values, bond.GetStereo())


BondFeature = FeatureGenerator(
    [
        (bond_type, 5),
        (bond_is_conjugated, 2),
        (bond_is_in_ring, 2),
        (bond_stereo, 7),
    ]
)  # length: 15


### mol to graph
## 1. mol to Basic Graph
def mol2graph(
    mol: Chem.Mol, atom_feature=AtomFeature, bond_feature=BondFeature, return_mol=False
):
    n_atoms = mol.GetNumAtoms()
    x = []
    mol.GetAtoms()
    for atom_id in range(n_atoms):
        atom = mol.GetAtomWithIdx(atom_id)
        x.append(atom_feature(atom))
    edge_index, edge_attr = [], []
    for bond in mol.GetBonds():
        begin_atom_id, end_atom_id = (
            bond.GetBeginAtom().GetIdx(),
            bond.GetEndAtom().GetIdx(),
        )
        edge_index.append([end_atom_id, begin_atom_id])
        edge_attr.append(bond_feature(bond))
    x = np.array(x, dtype=np.int64)
    if not len(edge_index):
        edge_index = np.empty((2, 0), dtype=np.int64)
        edge_attr = np.empty((0, len(BondFeature.feature_fns)), dtype=np.int64)
    else:
        edge_index = np.array(edge_index, dtype=np.int64).T
        edge_attr = np.array(edge_attr, dtype=np.int64)
    data = {"x": x, "edge_index": edge_index, "edge_attr": edge_attr}
    if return_mol:
        data["rdkit_mol"] = mol
    return data


## 2. mol to Line-Graph (Fragment: Triplets)
class Triplet_Vocab(object):
    def __init__(self, n_atom_types, n_bond_types):
        self.n_atom_types = n_atom_types
        self.n_bond_types = n_bond_types
        self.vocab = self.construct()

    def construct(self):
        vocab = {}
        # bonded Triplets
        atom_ids = list(range(self.n_atom_types))
        bond_ids = list(range(self.n_bond_types))
        id = 0
        for atom_id_1 in atom_ids:
            vocab[atom_id_1] = {}  # 一级字典嵌套
            for bond_id in bond_ids:
                vocab[atom_id_1][bond_id] = {}  # 二级字典嵌套
                for atom_id_2 in atom_ids:
                    if atom_id_2 >= atom_id_1:  # a-b和b-a等价
                        vocab[atom_id_1][bond_id][atom_id_2] = id  # 三元组id
                        id += 1
        for atom_id in atom_ids:
            vocab[atom_id][999] = {}
            vocab[atom_id][999][999] = id
            id += 1
        vocab[999] = {}
        vocab[999][999] = {}
        vocab[999][999][999] = id
        self.vocab_size = id
        return vocab

    def index(self, atom_type1, atom_type2, bond_type):
        atom_type1, atom_type2 = np.sort([atom_type1, atom_type2])
        try:
            return self.vocab[atom_type1][bond_type][atom_type2]
        except Exception as e:
            print(e)
            return self.vocab_size

    def one_hot_feature_index(
        self, atom_type_one_hot1, atom_type_one_hot2, bond_type_one_hot
    ):
        atom_type1, atom_type2 = np.sort(
            [atom_type_one_hot1.index(1), atom_type_one_hot2.index(1)]
        ).tolist()
        bond_type = bond_type_one_hot.index(1)
        return self.index([atom_type1, bond_type, atom_type2])


INF = 1e6
VIRTUAL_ATOM_INDICATOR = -1
VIRTUAL_ATOM_FEATURE_PLACEHOLDER = -1
VIRTUAL_BOND_FEATURE_PLACEHOLDER = -1
VIRTUAL_PATH_INDICATOR = -INF

N_ATOM_TYPES = 101
N_BOND_TYPES = 5


# 直接将mol转为PyG Data数据，没有经过transform.MolToPyGData函数的中介处理
def mol2linegraph(
    mol: Chem.Mol,
    vocab,
    atom_feature=AtomFeature,
    bond_feature=BondFeature,
    max_length=5,
    n_virtual_nodes=8,
    add_self_loop=True,
    return_mol=False,
):
    # 重新排列原子的编号以确保唯一性
    new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
    mol = Chem.rdmolops.RenumberAtoms(mol, new_order)

    # 获取分子中的原子数量
    n_atoms = mol.GetNumAtoms()
    mol_atom_features = []  # 原子特征

    # 定义原子特征和键特征的维度
    d_atom_feats = 9
    d_bond_feats = 4
    # 为每个原子生成特征
    for atom_id in range(n_atoms):
        atom = mol.GetAtomWithIdx(atom_id)
        mol_atom_features.append(atom_feature(atom))

    # 分割算法 —— 构建线图
    # 1. 线图的节点的构建
    # 初始化一个用于存储原子对到三元组 ID 的映射
    atomIDPair_to_tripletId = np.ones(shape=(n_atoms, n_atoms)) * np.nan

    # 初始化用于存储三元组标签和虚拟原子/虚拟节点标签的列表
    triplet_labels = []
    virtual_atom_and_virtual_node_labels = []

    # 初始化用于存储原子对特征、键特征的列表
    atom_pairs_features_in_triplets = (
        []
    )  # 三元组节点中的原子特征：元素——[atom_features[begin_atom_id], atom_features[end_atom_id]]
    bond_features_in_triplets = (
        []
    )  # 三元组节点中的键特征：元素——bond_featurizer_all(bond)

    bonded_atoms = set()  # 用于记录【键遍历】中已经进行了三元组链接的节点

    triplet_id = 0  # 三元组节点的【分子内索引】初始化为 0

    # 获取：【实】三元组 &【半实半虚】三元组的————【特征：atom_pairs_features_in_triplets/bond_features_in_triplets】&【标签：triplet_labels/virtual_atom_and_virtual_node_labels】
    # 【实】三元组：
    for bond in mol.GetBonds():
        # 原子在分子中的索引（原子节点的【分子内索引】）
        begin_atom_id, end_atom_id = np.sort(
            [bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()]
        )
        # 添加【三元组节点】的原子对特征和键特征
        atom_pairs_features_in_triplets.append(
            [mol_atom_features[begin_atom_id], mol_atom_features[end_atom_id]]
        )
        mol_bond_feature = bond_feature(bond)
        bond_features_in_triplets.append(mol_bond_feature)
        # 记录已通过“遍历键”实现特征化的原子
        bonded_atoms.add(begin_atom_id)
        bonded_atoms.add(end_atom_id)
        # 添加三元组标签（根据【原子1-键-原子2】的“【类型索引】”在【词典】中找到对应的【三元组词汇】的【词典索引】）
        # triplet_labels.append(vocab.index([begin_atom_id][:N_ATOM_TYPES].index(1), mol_atom_features[end_atom_id][:N_ATOM_TYPES].index(1), mol_bond_feature[:N_BOND_TYPES].index(1)))
        triplet_labels.append(
            vocab.index(
                mol_atom_features[begin_atom_id][0],
                mol_atom_features[end_atom_id][0],
                mol_bond_feature[0],
            )
        )
        # 标记节点类型：实三元组标记为0
        virtual_atom_and_virtual_node_labels.append(0)
        # 更新原子对（的【分子内索引】）到三元组 （的【分子内索引】） 的映射
        atomIDPair_to_tripletId[begin_atom_id, end_atom_id] = atomIDPair_to_tripletId[
            end_atom_id, begin_atom_id
        ] = triplet_id
        triplet_id += 1  # 更新三元组 ID
    # 孤立原子→【半实半虚】三元组
    for atom_id in range(n_atoms):
        if atom_id not in bonded_atoms:  # 【实+虚】三元组节点
            # 添加虚拟原子特征和虚拟键特征
            atom_pairs_features_in_triplets.append(
                [
                    mol_atom_features[atom_id],
                    [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats,
                ]
            )
            bond_features_in_triplets.append(
                [VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats
            )
            triplet_labels.append(
                vocab.index(
                    mol_atom_features[atom_id][:N_ATOM_TYPES].index(1), 999, 999
                )
            )
            virtual_atom_and_virtual_node_labels.append(VIRTUAL_ATOM_INDICATOR)

    # 2. 线图的边和路径构建
    # 构建三元组之间的边和路径
    edge_index = []  # 用于存储边

    paths = []  # 用于存储路径
    line_graph_path_labels = []  # line_graph图路径bool标签
    mol_graph_path_labels = []  # 分子图路径bool标签
    virtual_path_labels = []  # 虚拟路径bool标签
    self_loop_labels = []  # 自环路径bool标签

    # 2.1 构建线图的【原始边】+ 生成【第一类线图路径】
    for i in range(n_atoms):
        node_ids = atomIDPair_to_tripletId[
            i
        ]  # 该原子1涉及的所有三元组节点的【分子内索引的列表】
        node_ids = node_ids[~np.isnan(node_ids)]  # 过滤掉 NaN 值

        if (
            len(node_ids) >= 2
        ):  # 如果“原子1”不止存在于1个三元组节点中，则在该节点涉及的所有三元组之间建立全连接的【边】并生成路径
            # 在与某个【原子节点】相关的所有【三元组节点】之间，全连接地建立【线图的边】
            new_edges = list(
                permutations(node_ids, 2)
            )  # 生成列表中所有数的全部2长度排列元组——即：在该点相关的所有三元组之间建立全连接的【边】
            edge_index.extend(new_edges)  # 放入边表

            # new_edge：与该原子相关的【线图的边】
            # new_edge[0]：【线图的边】的【起点三元组】；
            # new_edge[1]：【线图的边】的【终点三元组】；
            # 将每一条【线图边】都构建为一条【线图路径】：这种“最简单”的【线图路径】被称为：【第一类线图路径】，即：“line_graph_path”
            new_paths = [
                [new_edge[0]]
                + [VIRTUAL_PATH_INDICATOR] * (max_length - 2)
                + [new_edge[1]]
                for new_edge in new_edges
            ]
            paths.extend(new_paths)

            # 标签设定：标记【线图路径】的类型为“line_graph_path”
            n_new_edges = len(new_edges)
            line_graph_path_labels.extend(
                [1] * n_new_edges
            )  # “最简单”的【线图路径】——由【线图的单条边】直接生成
            mol_graph_path_labels.extend([0] * n_new_edges)
            virtual_path_labels.extend([0] * n_new_edges)
            self_loop_labels.extend([0] * n_new_edges)

    # 2.2 生成【第二类线图路径】+ 构建线图的【长路径头尾直连边】
    # 获取分子图中所有原子节点的最短路径
    adj_matrix = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol))
    nx_g = nx.from_numpy_array(adj_matrix)
    paths_dict = dict(
        nx.algorithms.all_pairs_shortest_path(nx_g, max_length + 1)
    )  # paths_dict字典：键是分子图中某个节点（原子）的索引，值是另一个字典。内层字典的键是与外层键对应节点相连的其他节点的索引，值是从外层键节点到内层键节点的最短路径列表。

    for i in paths_dict.keys():  # 起点
        for j in paths_dict[i]:  # 终点
            path = paths_dict[i][j]  # 一条【分子图路径】
            path_length = len(path)

            if (
                3 < path_length <= max_length + 1
            ):  # 分子图路径上至少包含四个原子节点（至少涉及三个三元组节点，因为只涉及两个三元组节点的路径已经直接利用线图边生成了），节点数不小于最大线段数加1
                # 生成【第二类线图路径】
                triplet_ids = [
                    atomIDPair_to_tripletId[path[pi], path[pi + 1]]
                    for pi in range(len(path) - 1)
                ]  # 获取一条【分子图路径】上经过的所有三元组节点
                path_start_triplet_id = triplet_ids[0]  # 【第二类线图路径】的起点三元组
                path_end_triplet_id = triplet_ids[-1]  # 【第二类线图路径】的终点三元组
                triplet_path = triplet_ids[1:-1]  # 【第二类线图路径】的中间部分
                # 拼接得到【第二类线图路径】
                triplet_path = (
                    [path_start_triplet_id]
                    + triplet_path
                    + [VIRTUAL_PATH_INDICATOR] * (max_length - len(triplet_path) - 2)
                    + [path_end_triplet_id]
                )
                paths.append(triplet_path)

                # 构建线图路径起点和终点间的边连接
                edge_index.append([path_start_triplet_id, path_end_triplet_id])

                line_graph_path_labels.append(0)
                mol_graph_path_labels.append(1)
                virtual_path_labels.append(0)
                self_loop_labels.append(0)

    # 2.3 生成【第三类线图路径】+更新【虚节点的特征情况】+构建线图的【虚实相关边】
    for n in range(n_virtual_nodes):  # 遍历每一个【虚节点】（默认有8个）
        for i in range(len(atom_pairs_features_in_triplets) - n):  # 对每一个【实节点】

            # 将本轮涉及的【虚节点】与所有【实节点】双向连接起来
            edge_index.append([len(atom_pairs_features_in_triplets), i])
            edge_index.append([i, len(atom_pairs_features_in_triplets)])

            # 直接将每一条【虚实边】都构建为一条【线图路径】
            paths.append(
                [len(atom_pairs_features_in_triplets)]
                + [VIRTUAL_PATH_INDICATOR] * (max_length - 2)
                + [i]
            )
            paths.append(
                [i]
                + [VIRTUAL_PATH_INDICATOR] * (max_length - 2)
                + [len(atom_pairs_features_in_triplets)]
            )

            # 标记virtual_path_labels为对应的虚节点编号
            line_graph_path_labels.extend([0, 0])
            mol_graph_path_labels.extend([0, 0])
            virtual_path_labels.extend([n + 1, n + 1])
            self_loop_labels.extend([0, 0])

        # 添加新的虚原子（建立虚原子与虚原子的三元组特征）
        atom_pairs_features_in_triplets.append(
            [
                [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats,
                [VIRTUAL_ATOM_FEATURE_PLACEHOLDER] * d_atom_feats,
            ]
        )

        # 键特征
        bond_features_in_triplets.append(
            [VIRTUAL_BOND_FEATURE_PLACEHOLDER] * d_bond_feats
        )

        # 三元组标签
        triplet_labels.append(vocab.index(999, 999, 999))

        # 虚拟节点的具体类型
        virtual_atom_and_virtual_node_labels.append(n + 1)

    # 2.4 生成【第四类线图路径】 + 添加线图的【自环边】
    if add_self_loop:
        for i in range(len(atom_pairs_features_in_triplets)):
            edge_index.append([i, i])
            paths.append([i] + [VIRTUAL_PATH_INDICATOR] * (max_length - 2) + [i])
            line_graph_path_labels.append(0)
            mol_graph_path_labels.append(0)
            virtual_path_labels.append(0)
            self_loop_labels.append(1)

    # 3. 创建 PyG Data 对象:
    # 将列表转换为 tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # 创建 PyG Data 对象
    g = Data(
        edge_index=edge_index,  # Only Use the Edge indices
    )

    # Additional node feature
    g.tripletsNode_features_part1_inside_atom_pair = torch.FloatTensor(
        np.array(atom_pairs_features_in_triplets)
    )
    g.tripletsNode_features_part2_inside_bond = torch.LongTensor(
        np.array(bond_features_in_triplets)
    )
    g.tripletsNode_vocab_label = torch.LongTensor(
        np.array(triplet_labels)
    )
    g.tripletsNode_ifvan_label = torch.LongTensor(
        np.array(virtual_atom_and_virtual_node_labels)
    )  # Virtual or actual node indicator

    # Paths and path labels, add them as additional properties if needed
    g.tripletsPaths = torch.LongTensor(
        np.array(paths)
    )  # Example: Adding path information
    g.tripletsPaths_lgp_label = torch.BoolTensor(
        np.array(line_graph_path_labels)
    )
    g.tripletsPaths_mgp_label = torch.BoolTensor(
        np.array(mol_graph_path_labels)
    )
    g.tripletsPaths_vp_label = torch.BoolTensor(
        np.array(virtual_path_labels)
    )
    g.tripletsPaths_sl_label = torch.BoolTensor(
        np.array(self_loop_labels)
    )

    if return_mol:
        g.rdkit_mol = mol
    return g


## 3. mol to xxxx Graph (Fragment: xxxx)

if __name__ == "__main__":
    smi = "CCO"
    mol = Chem.MolFromSmiles(smi)
    print(mol2graph(mol))
    vocab = Triplet_Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    print(mol2linegraph(mol, vocab))
