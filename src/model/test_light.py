import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
from torch import nn
from torch_geometric.data import Batch, Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax

VIRTUAL_ATOM_FEATURE_PLACEHOLDER = -1
VIRTUAL_BOND_FEATURE_PLACEHOLDER = -1


# Initializes the parameters of linear and embedding layers with normal distribution.
def init_params(module):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


# 1. Basic Module:
# Implements a multi-layer perceptron with optional hidden layer size configuration.
class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) with configurable number of layers and activation function.

    Args Needed:
        d_in_feats (int): Number of input features.
        d_out_feats (int): Number of output features.
        n_dense_layers (int): Number of dense layers in the MLP.
        activation (nn.Module): Activation function to use.
        d_hidden_feats (int, optional): Number of features in hidden layers. Defaults to d_out_feats.
    """

    def __init__(
        self, d_in_feats, d_out_feats, n_dense_layers, activation, d_hidden_feats=None
    ):
        super(MLP, self).__init__()
        self.n_dense_layers = n_dense_layers
        self.d_hidden_feats = d_out_feats if d_hidden_feats is None else d_hidden_feats
        self.dense_layer_list = nn.ModuleList()
        # Input Layer
        self.input_layer = nn.Linear(d_in_feats, self.d_hidden_feats)
        # Hidden Layers
        for _ in range(
            self.n_dense_layers - 2
        ):  # add gidden layer with the same num of dense nodes
            self.dense_layer_list.append(
                nn.Linear(self.d_hidden_feats, self.d_hidden_feats)
            )
        # Output Layer
        self.output_layer = nn.Linear(self.d_hidden_feats, d_out_feats)
        self.act = activation

    def forward(self, feats):
        feats = self.act(self.input_layer(feats))
        for i in range(self.n_dense_layers - 2):
            feats = self.act(self.dense_layer_list[i](feats))
        feats = self.output_layer(feats)
        return feats


# Implements a residual block with normalization, projection, and a feedforward network.
class Residual(nn.Module):
    """
    A residual block with layer normalization, linear projection, and a feedforward neural network (FFN).

    Args:
        d_in_feats (int): Number of input features.
        d_out_feats (int): Number of output features.
        n_ffn_dense_layers (int): Number of dense layers in the FFN.
        feat_drop (float): Dropout rate for feature dropout.
        activation (nn.Module): Activation function to use.
    """

    def __init__(
        self, d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, activation
    ):
        super(Residual, self).__init__()
        self.norm = nn.LayerNorm(d_in_feats)
        self.input_layer = nn.Linear(d_in_feats, d_out_feats)
        self.mlp = MLP(
            d_out_feats,
            d_out_feats,
            n_ffn_dense_layers,
            activation,
            d_hidden_feats=d_out_feats * 4,
        )
        self.feat_dropout = nn.Dropout(feat_drop)

    def forward(self, triplet_h, aggregated_message):
        module_input = triplet_h + self.feat_dropout(self.in_proj(aggregated_message))
        residual = module_input
        module_output = self.feat_dropout(self.ffn(self.norm(module_input)))
        module_output_with_residual = module_output + residual
        return module_output_with_residual


# 2. Embedding Module:
# Embeds atom features into a high-dimensional space.
class AtomEmbedding(nn.Module):
    """
    Embeds atom features into a high-dimensional space for use in graph neural networks.

    Args:
        d_atom_feats (int): Dimensionality of input atom features.
        d_g_feats (int): Dimensionality of graph features.
        input_drop (float): Dropout rate for input features.
    """

    def __init__(self, d_atom_feats, d_g_feats, input_drop):
        super(AtomEmbedding, self).__init__()
        self.input_layer = nn.Linear(d_atom_feats, d_g_feats)
        self.virtual_atom_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, pair_node_feats, indicators):
        pair_node_h = self.input_layer(pair_node_feats)
        pair_node_h[indicators == VIRTUAL_ATOM_FEATURE_PLACEHOLDER, 1, :] = (
            self.virtual_atom_emb.weight
        )  # .half()
        return torch.sum(self.input_dropout(pair_node_h), dim=-2)


# Embeds bond features into a high-dimensional space.
class BondEmbedding(nn.Module):
    """
    Embeds bond features into a high-dimensional space for use in graph neural networks.

    Args:
        d_bond_feats (int): Dimensionality of input bond features.
        d_g_feats (int): Dimensionality of graph features.
        input_drop (float): Dropout rate for input features.
    """

    def __init__(self, d_bond_feats, d_g_feats, input_drop):
        super(BondEmbedding, self).__init__()
        self.input_layer = nn.Linear(d_bond_feats, d_g_feats)
        self.virutal_bond_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)

    def forward(self, edge_feats, indicators):
        edge_h = self.input_layer(edge_feats)
        edge_h[indicators == VIRTUAL_BOND_FEATURE_PLACEHOLDER] = (
            self.virutal_bond_emb.weight
        )  # .half()
        return self.input_dropout(edge_h)


# Embeds triplet features into a high-dimensional space, combining atom, bond, and path features.
class TripletEmbedding(nn.Module):
    """
    Embeds triplet (atom-bond-atom) features into a high-dimensional space, combining atom, bond, and path features.

    Args:
        d_g_feats (int): Dimensionality of graph features.
        d_fp_feats (int): Dimensionality of fingerprint features.
        d_md_feats (int): Dimensionality of molecular descriptor features.
        activation (nn.Module): Activation function to use.
    """

    def __init__(self, d_g_feats, d_fp_feats, d_md_feats, activation=nn.GELU()):
        super(TripletEmbedding, self).__init__()
        self.input_layer_g = MLP(d_g_feats * 2, d_g_feats, 2, activation)
        self.input_layer_fp = MLP(d_fp_feats, d_g_feats, 2, activation)
        self.input_layer_md = MLP(d_md_feats, d_g_feats, 2, activation)

    def forward(self, node_h, edge_h, fp, md, indicators):
        triplet_h = torch.cat([node_h, edge_h], dim=-1)
        triplet_h = self.input_layer_g(triplet_h)
        triplet_h[indicators == 1] = self.input_layer_fp(fp)
        triplet_h[indicators == 2] = self.input_layer_md(md)
        return triplet_h


# 3. Transformer Module:
# Implements a transformer module designed for triplet-based graph processing.
class TripletTransformer(nn.Module):
    """
    A transformer-based module designed to process triplet (node-edge-node) features in a graph.

    Args:
        d_feats (int): Dimensionality of input features.
        d_hpath_ratio (int): Ratio to control the dimension of triplet path features.
        path_length (int): Maximum length of paths considered in the graph.
        n_heads (int): Number of attention heads.
        n_ffn_dense_layers (int): Number of dense layers in the feedforward network.
        feat_drop (float): Dropout rate for feature dropout.
        attn_drop (float): Dropout rate for attention dropout.
        activation (nn.Module): Activation function to use.
    """

    def __init__(
        self,
        d_feats,
        d_hpath_ratio,
        path_length,
        n_heads,
        n_ffn_dense_layers,
        feat_drop=0.0,
        attn_drop=0.0,
        activation=nn.GELU(),
    ):
        super(TripletTransformer, self).__init__()
        self.d_feats = d_feats
        self.d_trip_path = d_feats // d_hpath_ratio
        self.path_length = path_length
        self.n_heads = n_heads
        self.scale = d_feats ** (-0.5)

        # 使用 nn.TransformerEncoder 来替代手动实现的多头注意力和前馈层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_feats,
            nhead=n_heads,
            dim_feedforward=self.d_trip_path,
            dropout=feat_drop,
            activation=(
                activation.func_name if hasattr(activation, "func_name") else "gelu"
            ),
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_ffn_dense_layers
        )

        self.attention_norm = nn.LayerNorm(d_feats)
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)

    def forward(self, triplet_h, dist_attn, path_attn):
        new_triplet_h = self.attention_norm(triplet_h)

        # Transformer编码器处理节点特征
        transformed_h = self.transformer_encoder(new_triplet_h)

        # 应用自注意力机制
        attn_values = self.attn_dropout(
            F.softmax(
                transformed_h + dist_attn.unsqueeze(2) + path_attn.unsqueeze(2), dim=-1
            )
        )
        transformed_h = attn_values * transformed_h

        return transformed_h


# 4. Core Module:
# Core model that integrates multiple TripletTransformer layers for graph-based molecular representation learning.
class LiGhT(nn.Module):
    """
    The core model integrating multiple TripletTransformer layers for processing molecular graphs.

    Args:
        d_g_feats (int): Dimensionality of graph features.
        d_hpath_ratio (int): Ratio to control the dimension of triplet path features.
        path_length (int): Maximum length of paths considered in the graph.
        n_mol_layers (int): Number of molecular transformer layers.
        n_heads (int): Number of attention heads.
        n_ffn_dense_layers (int): Number of dense layers in the feedforward network.
        feat_drop (float): Dropout rate for feature dropout.
        attn_drop (float): Dropout rate for attention dropout.
        activation (nn.Module): Activation function to use.
    """

    def __init__(
        self,
        d_g_feats,
        d_hpath_ratio,
        path_length,
        n_mol_layers=2,
        n_heads=4,
        n_ffn_dense_layers=4,
        feat_drop=0.0,
        attn_drop=0.0,
        activation=nn.GELU(),
    ):
        super(LiGhT, self).__init__()
        self.n_mol_layers = n_mol_layers
        self.n_heads = n_heads
        self.path_length = path_length
        self.d_g_feats = d_g_feats
        self.d_trip_path = d_g_feats // d_hpath_ratio

        self.mask_emb = nn.Embedding(1, d_g_feats)
        # Distance Attention
        self.path_len_emb = nn.Embedding(path_length + 1, d_g_feats)
        self.virtual_path_emb = nn.Embedding(1, d_g_feats)
        self.self_loop_emb = nn.Embedding(1, d_g_feats)
        self.dist_attn_layer = nn.Sequential(
            nn.Linear(self.d_g_feats, self.d_g_feats),
            activation,
            nn.Linear(self.d_g_feats, n_heads),
        )
        # Path Attention
        self.trip_fortrans = nn.ModuleList(
            [
                MLP(d_g_feats, self.d_trip_path, 2, activation)
                for _ in range(self.path_length)
            ]
        )
        self.path_attn_layer = nn.Sequential(
            nn.Linear(self.d_trip_path, self.d_trip_path),
            activation,
            nn.Linear(self.d_trip_path, n_heads),
        )
        # Molecule Transformer Layers
        self.mol_T_layers = nn.ModuleList(
            [
                # TripletTransformer模块定义构造：__init__ 方法中接收静态的超参数
                TripletTransformer(
                    d_g_feats,
                    d_hpath_ratio,
                    path_length,
                    n_heads,
                    n_ffn_dense_layers,
                    feat_drop,
                    attn_drop,
                    activation,
                )
                for _ in range(n_mol_layers)
            ]
        )

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.act = activation

    def _featurize_path(self, g, path_indices):
        mask = (path_indices[:, :] >= 0).to(torch.int32)
        path_feats = torch.sum(mask, dim=-1)
        path_feats = self.path_len_emb(path_feats)
        path_feats[g.tripletsPaths_vp_label == 1] = (
            self.virtual_path_emb.weight
        )  # virtual path
        path_feats[g.tripletsPaths_sl_label == 1] = (
            self.self_loop_emb.weight
        )  # self loop
        return path_feats

    def _init_path(self, g, triplet_h, path_indices):
        g = g.local_var()
        path_indices[path_indices < -99] = -1
        path_h = []
        for i in range(self.path_length):
            path_h.append(
                torch.cat(
                    [
                        self.trip_fortrans[i](triplet_h),
                        torch.zeros(size=(1, self.d_trip_path)).to(self._device()),
                    ],
                    dim=0,
                )[path_indices[:, i]]
            )
        path_h = torch.stack(path_h, dim=-1)
        mask = (path_indices >= 0).to(torch.int32)
        path_size = torch.sum(mask, dim=-1, keepdim=True)
        path_h = torch.sum(path_h, dim=-1) / path_size
        return path_h

    def forward(self, g, triplet_h):
        path_indices = g.tripletsPaths  # 使用【路径列表】作为【模型输入】
        dist_h = self._featurize_path(g, path_indices)  # 利用路径——生成距离嵌入编码
        path_h = self._init_path(
            g, triplet_h, path_indices
        )  # 利用路径——生成路径嵌入编码

        dist_attn, path_attn = self.dist_attn_layer(dist_h), self.path_attn_layer(
            path_h
        )  # 生成距离和路径的两类注意力

        for i in range(self.n_mol_layers):
            # TripletTransformer模块调用传参： 调用forward方法传入的是动态的输入数据
            # # triplet_h = self.mol_T_layers[i](triplet_h, dist_attn, path_attn) 输入：g, triplet_h, dist_attn, path_attn
            triplet_h = self.mol_T_layers[i](data, triplet_h, dist_attn, path_attn)
        return triplet_h

    def _device(self):
        return next(self.parameters()).device


# Predicts molecular properties using the LiGhT model, incorporating node, edge, and triplet embeddings.
class LiGhTPredictor(nn.Module):
    """
    Predicts molecular properties using the LiGhT model, integrating node, edge, and triplet embeddings.

    Args:
        d_node_feats (int): Dimensionality of input node features.
        d_edge_feats (int): Dimensionality of input edge features.
        d_g_feats (int): Dimensionality of graph features.
        d_fp_feats (int): Dimensionality of fingerprint features.
        d_md_feats (int): Dimensionality of molecular descriptor features.
        d_hpath_ratio (int): Ratio to control the dimension of triplet path features.
        n_mol_layers (int): Number of molecular transformer layers.
        path_length (int): Maximum length of paths considered in the graph.
        n_heads (int): Number of attention heads.
        n_ffn_dense_layers (int): Number of dense layers in the feedforward network.
        input_drop (float): Dropout rate for input features.
        feat_drop (float): Dropout rate for feature dropout.
        attn_drop (float): Dropout rate for attention dropout.
        activation (nn.Module): Activation function to use.
        n_node_types (int): Number of node types to predict.
        readout_mode (str): Readout operation to aggregate node features ('mean', 'sum', etc.).
    """
    def __init__(
            self, 
            d_node_feats=40, 
            d_edge_feats=12, 
            d_g_feats=128, 
            d_fp_feats=512, 
            d_md_feats=200, 
            d_hpath_ratio=1, 
            n_mol_layers=2, 
            path_length=5, 
            n_heads=4, 
            n_ffn_dense_layers=2,
            input_drop=0., 
            feat_drop=0., 
            attn_drop=0., 
            activation=nn.GELU(), 
            n_node_types=1, 
            readout_mode='mean'):
        super(LiGhTPredictor, self).__init__()
        self.d_g_feats = d_g_feats
        self.readout_mode = readout_mode
        # Input
        self.node_emb = AtomEmbedding(d_node_feats, d_g_feats, input_drop)
        self.edge_emb = BondEmbedding(d_edge_feats, d_g_feats, input_drop)
        self.triplet_emb = TripletEmbedding(
            d_g_feats, d_fp_feats, d_md_feats, activation
        )
        self.mask_emb = nn.Embedding(1, d_g_feats)
        # Backbone model
        self.model = LiGhT(
            d_g_feats, 
            d_hpath_ratio, 
            path_length, 
            n_mol_layers, 
            n_heads, 
            n_ffn_dense_layers, 
            feat_drop, 
            attn_drop, 
            activation)
        # Predictor
        self.node_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, n_node_types)
        )
        self.fp_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, d_fp_feats)
        )
        self.md_predictor = nn.Sequential(
            nn.Linear(d_g_feats, d_g_feats),
            activation,
            nn.Linear(d_g_feats, d_md_feats)
        )
        self.apply(lambda module: init_params(module))

    def forward(self, g, fp, md):
        indicators = (
            g.tripletsNode_ifvan_label
        )  # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes

        # Input: 使用了node_emb/edge_emb/triplet_emb，构造得到triplet_h
        node_h = self.node_emb(
            g.tripletsNode_features_part1_inside_atom_pair, indicators
        )
        edge_h = self.edge_emb(g.tripletsNode_features_part2_inside_bond, indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)

        triplet_h[g.mask == 1] = self.mask_emb.weight

        # LiGhT
        triplet_h = self.model(g, triplet_h)  # 生成transformer的预测结果
        # Predictor
        return (
            self.node_predictor(triplet_h[g.mask >= 1]),
            self.fp_predictor(triplet_h[indicators == 1]),
            self.md_predictor(triplet_h[indicators == 2]),
        )
