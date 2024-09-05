import numpy as np
import torch
from rdkit import Chem
import dgl
from dgllife.utils.featurizers import ConcatFeaturizer, bond_type_one_hot, bond_is_conjugated, bond_is_in_ring, bond_stereo_one_hot, atomic_number_one_hot, atom_degree_one_hot, atom_formal_charge, atom_num_radical_electrons_one_hot, atom_hybridization_one_hot, atom_is_aromatic, atom_total_num_H_one_hot, atom_is_chiral_center, atom_chirality_type_one_hot, atom_mass
from functools import partial
from itertools import permutations
import networkx as nx
INF = 1e6
VIRTUAL_ATOM_INDICATOR = -1
VIRTUAL_ATOM_FEATURE_PLACEHOLDER = -1
VIRTUAL_BOND_FEATURE_PLACEHOLDER = -1
VIRTUAL_PATH_INDICATOR = -INF

N_ATOM_TYPES = 101
N_BOND_TYPES = 5
bond_featurizer_all = ConcatFeaturizer([ # 14
    partial(bond_type_one_hot, encode_unknown=True), # 5
    bond_is_conjugated, # 1
    bond_is_in_ring, # 1
    partial(bond_stereo_one_hot,encode_unknown=True) # 7
    ])
atom_featurizer_all = ConcatFeaturizer([ # 137
    partial(atomic_number_one_hot, encode_unknown=True), #101
    partial(atom_degree_one_hot, encode_unknown=True), # 12
    atom_formal_charge, # 1
    partial(atom_num_radical_electrons_one_hot, encode_unknown=True), # 6
    partial(atom_hybridization_one_hot, encode_unknown=True), # 6
    atom_is_aromatic, # 1
    partial(atom_total_num_H_one_hot, encode_unknown=True), # 6
    atom_is_chiral_center, # 1
    atom_chirality_type_one_hot, # 2
    atom_mass, # 1
    ])

# 词典构建
class Vocab(object):
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
            vocab[atom_id_1] = {} # 一级字典嵌套
            for bond_id in bond_ids:
                vocab[atom_id_1][bond_id] = {} # 二级字典嵌套
                for atom_id_2 in atom_ids:
                    if atom_id_2 >= atom_id_1: # a-b和b-a等价
                        vocab[atom_id_1][bond_id][atom_id_2]=id # 三元组id
                        id+=1
        for atom_id in atom_ids:
            vocab[atom_id][999] = {}
            vocab[atom_id][999][999] = id
            id+=1
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
    def one_hot_feature_index(self, atom_type_one_hot1, atom_type_one_hot2, bond_type_one_hot):
        atom_type1, atom_type2 = np.sort([atom_type_one_hot1.index(1),atom_type_one_hot2.index(1)]).tolist()
        bond_type = bond_type_one_hot.index(1)
        return self.index([atom_type1, bond_type, atom_type2])


# 分子图对象化
def smiles_to_graph(smiles, vocab, max_length=5, n_virtual_nodes=8, add_self_loop=True):    
    # 针对【一个分子】处理
    # 通过 SMILES 字符串生成分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # 如果生成失败，则返回 None
    
    # 重新排列原子的编号以确保唯一性
    new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
    mol = Chem.rdmolops.RenumberAtoms(mol, new_order)
    
    # 获取分子中的原子数量
    n_atoms = mol.GetNumAtoms()
    atom_features = []
    
    # 定义原子特征和键特征的维度
    d_atom_feats = 137
    d_bond_feats = 14

    # 为每个原子生成特征
    for atom_id in range(n_atoms):
        atom = mol.GetAtomWithIdx(atom_id)
        atom_features.append(atom_featurizer_all(atom))

    # 分割算法 —— 构建线图
    # 1. 线图的节点的构建
    # 初始化一个用于存储原子对到三元组 ID 的映射
    atomIDPair_to_tripletId = np.ones(shape=(n_atoms,n_atoms))*np.nan
    
    # 初始化用于存储三元组标签和虚拟原子/虚拟节点标签的列表
    triplet_labels = []
    virtual_atom_and_virtual_node_labels = []
    
    # 初始化用于存储原子对特征、键特征的列表
    atom_pairs_features_in_triplets = [] # 三元组节点中的原子特征：元素——[atom_features[begin_atom_id], atom_features[end_atom_id]]
    bond_features_in_triplets = [] # 三元组节点中的键特征：元素——bond_featurizer_all(bond)
    
    bonded_atoms = set()  # 用于记录【键遍历】中已经进行了三元组链接的节点

    triplet_id = 0  # 三元组节点的【分子内索引】初始化为 0
    
    # 遍历所有的键，获取：
    '''
    【三元组节点的特征】——实三元组
    atom_pairs_features_in_triplets：[原子1的特征, 原子2的特征]
    bond_features_in_triplets：键的特征
    【三元组节点的词典索引】——实三元组
    triplet_labels
    【原子对→三元组节点】的【分子内索引】的【映射关系】——【仅对】实三元组
    atomIDPair_to_tripletId
    【三元组节点的分子内索引】——【仅对】实三元组
    triplet_id
    【三元组节点的类型标记】——实三元组标记为0
    virtual_atom_and_virtual_node_labels
    '''
    for bond in mol.GetBonds():
        # 原子在分子中的索引（原子节点的【分子内索引】）
        begin_atom_id, end_atom_id = np.sort([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
        
        # 添加【三元组节点】的原子对特征和键特征
        atom_pairs_features_in_triplets.append([atom_features[begin_atom_id], atom_features[end_atom_id]])
        bond_feature = bond_featurizer_all(bond)
        bond_features_in_triplets.append(bond_feature)
        
        #记录已通过“遍历键”实现特征化的原子
        bonded_atoms.add(begin_atom_id) 
        bonded_atoms.add(end_atom_id)
        
        # 添加三元组标签（根据【原子1-键-原子2】的“【类型索引】”在【词典】中找到对应的【三元组词汇】的【词典索引】）
        triplet_labels.append(vocab.index(atom_features[begin_atom_id][:N_ATOM_TYPES].index(1), atom_features[end_atom_id][:N_ATOM_TYPES].index(1), bond_feature[:N_BOND_TYPES].index(1)))

        # 标记节点类型：实三元组标记为0
        virtual_atom_and_virtual_node_labels.append(0)
        
        # 更新原子对（的【分子内索引】）到三元组 （的【分子内索引】） 的映射
        atomIDPair_to_tripletId[begin_atom_id, end_atom_id] = atomIDPair_to_tripletId[end_atom_id, begin_atom_id] = triplet_id
        triplet_id += 1  # 更新三元组 ID
    
    # 处理【键遍历】中未连接的原子，处理模式：
    '''
    【三元组节点的特征】——虚三元组
    atom_pairs_features_in_triplets：[原子1的特征，虚原子的特征（全-1特征向量）]
    bond_features_in_triplets：虚键的特征（全-1特征向量）
    【三元组节点的词典索引】——虚三元组
    triplet_labels
    【三元组节点的类型标记】——虚三元组标记为-1
    virt1ual_atom_and_virtual_node_labels
    '''
    for atom_id in range(n_atoms):
        if atom_id not in bonded_atoms: # 【实+虚】三元组节点
            # 添加虚拟原子特征和虚拟键特征
            atom_pairs_features_in_triplets.append([atom_features[atom_id], [VIRTUAL_ATOM_FEATURE_PLACEHOLDER]*d_atom_feats])
            bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER]*d_bond_feats)
            triplet_labels.append(vocab.index(atom_features[atom_id][:N_ATOM_TYPES].index(1), 999, 999))
            virtual_atom_and_virtual_node_labels.append(VIRTUAL_ATOM_INDICATOR)

    # 2. 线图的边和路径构建
    # 构建三元组之间的边和路径
    edges = []  # 用于存储边
    paths = []  # 用于存储路径
    line_graph_path_labels = []  # line_graph图路径bool标签
    mol_graph_path_labels = []  # 分子图路径bool标签
    virtual_path_labels = []  # 虚拟路径bool标签
    self_loop_labels = []  # 自环路径bool标签
    
    # 2.1 构建线图的【原始边】+ 生成【第一类线图路径】
    for i in range(n_atoms):
        node_ids = atomIDPair_to_tripletId[i] # 该原子1涉及的所有三元组节点的【分子内索引的列表】
        node_ids = node_ids[~np.isnan(node_ids)]  # 过滤掉 NaN 值
        
        if len(node_ids) >= 2: # 如果“原子1”不止存在于1个三元组节点中，则在该节点涉及的所有三元组之间建立全连接的【边】并生成路径
            # 在与某个【原子节点】相关的所有【三元组节点】之间，全连接地建立【线图的边】
            new_edges = list(permutations(node_ids, 2)) # 生成列表中所有数的全部2长度排列元组——即：在该点相关的所有三元组之间建立全连接的【边】
            edges.extend(new_edges) # 放入边表
            
            # new_edge：与该原子相关的【线图的边】
            # new_edge[0]：【线图的边】的【起点三元组】；
            # new_edge[1]：【线图的边】的【终点三元组】；
            # 将每一条【线图边】都构建为一条【线图路径】：这种“最简单”的【线图路径】被称为：【第一类线图路径】，即：“line_graph_path”
            new_paths = [[new_edge[0]] + [VIRTUAL_PATH_INDICATOR]*(max_length-2) + [new_edge[1]] for new_edge in new_edges]
            paths.extend(new_paths)
            
            # 标签设定：标记【线图路径】的类型为“line_graph_path”
            n_new_edges = len(new_edges)
            line_graph_path_labels.extend([1]*n_new_edges)  # “最简单”的【线图路径】——由【线图的单条边】直接生成
            mol_graph_path_labels.extend([0]*n_new_edges)
            virtual_path_labels.extend([0]*n_new_edges)
            self_loop_labels.extend([0]*n_new_edges)
    
    # 2.2 生成【第二类线图路径】+ 构建线图的【长路径头尾直连边】
    # 获取分子图中所有原子节点的最短路径
    adj_matrix = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol))
    nx_g = nx.from_numpy_array(adj_matrix)
    paths_dict = dict(nx.algorithms.all_pairs_shortest_path(nx_g, max_length+1)) # paths_dict字典：键是分子图中某个节点（原子）的索引，值是另一个字典。内层字典的键是与外层键对应节点相连的其他节点的索引，值是从外层键节点到内层键节点的最短路径列表。
    
    for i in paths_dict.keys(): # 起点
        for j in paths_dict[i]: # 终点
            path = paths_dict[i][j] # 一条【分子图路径】
            path_length = len(path)
            
            if 3 < path_length <= max_length+1: # 分子图路径上至少包含四个原子节点（至少涉及三个三元组节点，因为只涉及两个三元组节点的路径已经直接利用线图边生成了），节点数不小于最大线段数加1
                # 生成【第二类线图路径】
                triplet_ids = [atomIDPair_to_tripletId[path[pi], path[pi+1]] for pi in range(len(path)-1)] # 获取一条【分子图路径】上经过的所有三元组节点
                path_start_triplet_id = triplet_ids[0] # 【第二类线图路径】的起点三元组
                path_end_triplet_id = triplet_ids[-1] # 【第二类线图路径】的终点三元组
                triplet_path = triplet_ids[1:-1] # 【第二类线图路径】的中间部分
                # 拼接得到【第二类线图路径】
                triplet_path = [path_start_triplet_id] + triplet_path + [VIRTUAL_PATH_INDICATOR]*(max_length-len(triplet_path)-2) + [path_end_triplet_id]
                paths.append(triplet_path) 
                
                # 构建线图路径起点和终点间的边连接
                edges.append([path_start_triplet_id, path_end_triplet_id]) 

                line_graph_path_labels.append(0) 
                mol_graph_path_labels.append(1) 
                virtual_path_labels.append(0) 
                self_loop_labels.append(0)
    
    # 2.3 生成【第三类线图路径】+更新【虚节点的特征情况】+构建线图的【虚实相关边】
    for n in range(n_virtual_nodes): # 遍历每一个【虚节点】（默认有8个）
        for i in range(len(atom_pairs_features_in_triplets)-n): # 对每一个【实节点】
            
            # 将本轮涉及的【虚节点】与所有【实节点】双向连接起来
            edges.append([len(atom_pairs_features_in_triplets), i])
            edges.append([i, len(atom_pairs_features_in_triplets)])
            
            # 直接将每一条【虚实边】都构建为一条【线图路径】
            paths.append([len(atom_pairs_features_in_triplets)] + [VIRTUAL_PATH_INDICATOR]*(max_length-2) + [i])
            paths.append([i] + [VIRTUAL_PATH_INDICATOR]*(max_length-2) + [len(atom_pairs_features_in_triplets)])

            # 标记virtual_path_labels为对应的虚节点编号
            line_graph_path_labels.extend([0,0])
            mol_graph_path_labels.extend([0,0])
            virtual_path_labels.extend([n+1, n+1])
            self_loop_labels.extend([0,0])
        
        # 添加新的虚原子（建立虚原子与虚原子的三元组特征）
        atom_pairs_features_in_triplets.append([[VIRTUAL_ATOM_FEATURE_PLACEHOLDER]*d_atom_feats, [VIRTUAL_ATOM_FEATURE_PLACEHOLDER]*d_atom_feats])

        # 键特征
        bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER]*d_bond_feats)
        
        # 三元组标签
        triplet_labels.append(vocab.index(999, 999, 999))
        
        # 虚拟节点的具体类型
        virtual_atom_and_virtual_node_labels.append(n+1)
    
    # 2.4 生成【第四类线图路径】 + 添加线图的【自环边】
    if add_self_loop:
        for i in range(len(atom_pairs_features_in_triplets)):
            edges.append([i, i])
            paths.append([i] + [VIRTUAL_PATH_INDICATOR]*(max_length-2) + [i])
            line_graph_path_labels.append(0)
            mol_graph_path_labels.append(0)
            virtual_path_labels.append(0)
            self_loop_labels.append(1)

    # 3. DGL化：    
    # 创建 DGL 图
    edges = np.array(edges, dtype=np.int64)
    data = (edges[:,0], edges[:,1]) #([所有起点],[所有终点])
    g = dgl.graph(data)

    # 节点的内部原子特征
    g.ndata['begin_end'] = torch.FloatTensor(atom_pairs_features_in_triplets)
    # 节点的内部键特征
    g.ndata['edge'] = torch.FloatTensor(bond_features_in_triplets)
    # 节点的vocab索引标签
    g.ndata['label'] = torch.LongTensor(triplet_labels)
    # 节点的虚实类型标记
    g.ndata['vavn'] = torch.LongTensor(virtual_atom_and_virtual_node_labels)
    # 路径
    g.edata['path'] = torch.LongTensor(paths)
    # 四类路径的标签
    g.edata['lgp'] = torch.BoolTensor(line_graph_path_labels)
    g.edata['mgp'] = torch.BoolTensor(mol_graph_path_labels)
    g.edata['vp'] = torch.BoolTensor(virtual_path_labels)
    g.edata['sl'] = torch.BoolTensor(self_loop_labels)
    
    return g  # 返回创建的图对象

def smiles_to_graph_tune(smiles, max_length=5, n_virtual_nodes=8, add_self_loop=True):
    d_atom_feats = 137
    d_bond_feats = 14
    # Canonicalize
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    new_order = Chem.rdmolfiles.CanonicalRankAtoms(mol)
    mol = Chem.rdmolops.RenumberAtoms(mol, new_order)
    # Featurize Atoms
    n_atoms = mol.GetNumAtoms()
    atom_features = []
    
    for atom_id in range(n_atoms):
        atom = mol.GetAtomWithIdx(atom_id)
        atom_features.append(atom_featurizer_all(atom))
    atomIDPair_to_tripletId = np.ones(shape=(n_atoms,n_atoms))*np.nan
    # Construct and Featurize Triplet  Nodes
    ## bonded atoms
    virtual_atom_and_virtual_node_labels = []
    
    atom_pairs_features_in_triplets = []
    bond_features_in_triplets = []
    
    bonded_atoms = set()
    triplet_id = 0
    for bond in mol.GetBonds():
        begin_atom_id, end_atom_id = np.sort([bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()])
        atom_pairs_features_in_triplets.append([atom_features[begin_atom_id], atom_features[end_atom_id]])
        bond_feature = bond_featurizer_all(bond)
        bond_features_in_triplets.append(bond_feature)
        bonded_atoms.add(begin_atom_id)
        bonded_atoms.add(end_atom_id)
        virtual_atom_and_virtual_node_labels.append(0)
        atomIDPair_to_tripletId[begin_atom_id,end_atom_id] = atomIDPair_to_tripletId[end_atom_id,begin_atom_id] = triplet_id
        triplet_id += 1
    ## unbonded atoms 
    for atom_id in range(n_atoms):
        if atom_id not in bonded_atoms:
            atom_pairs_features_in_triplets.append([atom_features[atom_id], [VIRTUAL_ATOM_FEATURE_PLACEHOLDER]*d_atom_feats])
            bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER]*d_bond_feats)
            virtual_atom_and_virtual_node_labels.append(VIRTUAL_ATOM_INDICATOR)
    # Construct and Featurize Paths between Triplets
    ## line graph paths
    edges = []
    paths = []
    line_graph_path_labels = []
    mol_graph_path_labels = []
    virtual_path_labels = []
    self_loop_labels = []
    for i in range(n_atoms):
        node_ids = atomIDPair_to_tripletId[i]
        node_ids = node_ids[~np.isnan(node_ids)]
        if len(node_ids) >= 2:
            new_edges = list(permutations(node_ids,2))
            edges.extend(new_edges)
            new_paths = [[new_edge[0]]+[VIRTUAL_PATH_INDICATOR]*(max_length-2)+[new_edge[1]] for new_edge in new_edges]
            paths.extend(new_paths)
            n_new_edges = len(new_edges)
            line_graph_path_labels.extend([1]*n_new_edges)
            mol_graph_path_labels.extend([0]*n_new_edges)
            virtual_path_labels.extend([0]*n_new_edges)
            self_loop_labels.extend([0]*n_new_edges)
    # # molecule graph paths
    adj_matrix = np.array(Chem.rdmolops.GetAdjacencyMatrix(mol))
    nx_g = nx.from_numpy_array(adj_matrix)
    paths_dict = dict(nx.algorithms.all_pairs_shortest_path(nx_g,max_length+1))
    for i in paths_dict.keys():
        for j in paths_dict[i]:
            path = paths_dict[i][j]
            path_length = len(path)
            if 3 < path_length <= max_length+1:
                triplet_ids = [atomIDPair_to_tripletId[path[pi], path[pi+1]] for pi in range(len(path)-1)]
                path_start_triplet_id = triplet_ids[0]
                path_end_triplet_id = triplet_ids[-1]
                triplet_path = triplet_ids[1:-1]
                # assert [path_start_triplet_id,path_end_triplet_id] not in edges
                triplet_path = [path_start_triplet_id]+triplet_path+[VIRTUAL_PATH_INDICATOR]*(max_length-len(triplet_path)-2)+[path_end_triplet_id]
                paths.append(triplet_path)
                edges.append([path_start_triplet_id, path_end_triplet_id])
                line_graph_path_labels.append(0)
                mol_graph_path_labels.append(1)
                virtual_path_labels.append(0)
                self_loop_labels.append(0)
    for n in range(n_virtual_nodes):
        for i in range(len(atom_pairs_features_in_triplets)-n):
            edges.append([len(atom_pairs_features_in_triplets), i])
            edges.append([i, len(atom_pairs_features_in_triplets)])
            paths.append([len(atom_pairs_features_in_triplets)]+[VIRTUAL_PATH_INDICATOR]*(max_length-2)+[i])
            paths.append([i]+[VIRTUAL_PATH_INDICATOR]*(max_length-2)+[len(atom_pairs_features_in_triplets)])
            line_graph_path_labels.extend([0,0])
            mol_graph_path_labels.extend([0,0])
            virtual_path_labels.extend([n+1,n+1])
            self_loop_labels.extend([0,0])
        atom_pairs_features_in_triplets.append([[VIRTUAL_ATOM_FEATURE_PLACEHOLDER]*d_atom_feats, [VIRTUAL_ATOM_FEATURE_PLACEHOLDER]*d_atom_feats])
        bond_features_in_triplets.append([VIRTUAL_BOND_FEATURE_PLACEHOLDER]*d_bond_feats)
        virtual_atom_and_virtual_node_labels.append(n+1)
    if add_self_loop:
        for i in range(len(atom_pairs_features_in_triplets)):
            edges.append([i, i])
            paths.append([i]+[VIRTUAL_PATH_INDICATOR]*(max_length-2)+[i])
            line_graph_path_labels.append(0)
            mol_graph_path_labels.append(0)
            virtual_path_labels.append(0)
            self_loop_labels.append(1)
    edges = np.array(edges, dtype=np.int64)
    data = (edges[:,0], edges[:,1])
    g = dgl.graph(data)
    g.ndata['begin_end'] = torch.FloatTensor(atom_pairs_features_in_triplets)
    g.ndata['edge'] = torch.FloatTensor(bond_features_in_triplets)
    g.ndata['vavn'] = torch.LongTensor(virtual_atom_and_virtual_node_labels)
    g.edata['path'] = torch.LongTensor(paths)
    g.edata['lgp'] = torch.BoolTensor(line_graph_path_labels)
    g.edata['mgp'] = torch.BoolTensor(mol_graph_path_labels)
    g.edata['vp'] = torch.BoolTensor(virtual_path_labels)
    g.edata['sl'] = torch.BoolTensor(self_loop_labels)
    return g