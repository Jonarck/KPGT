import torch
from torch import nn
import dgl
from dgl import function as fn
from dgl.nn.functional import edge_softmax
import numpy as np

import networkx as nx


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

class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) with configurable number of layers and activation function.
    
    Args:
        d_in_feats (int): Number of input features.
        d_out_feats (int): Number of output features.
        n_dense_layers (int): Number of dense layers in the MLP.
        activation (nn.Module): Activation function to use.
        d_hidden_feats (int, optional): Number of features in hidden layers. Defaults to d_out_feats.
    """
    def __init__(self, d_in_feats, d_out_feats, n_dense_layers, activation, d_hidden_feats=None):
        super(MLP, self).__init__()
        self.n_dense_layers = n_dense_layers
        self.d_hidden_feats = d_out_feats if d_hidden_feats is None else d_hidden_feats
        self.dense_layer_list = nn.ModuleList()
        self.in_proj = nn.Linear(d_in_feats, self.d_hidden_feats)
        for _ in range(self.n_dense_layers-2):
            self.dense_layer_list.append(nn.Linear(self.d_hidden_feats, self.d_hidden_feats))
        self.out_proj = nn.Linear(self.d_hidden_feats, d_out_feats)
        self.act = activation
    
    def forward(self, feats):
        feats = self.act(self.in_proj(feats))
        for i in range(self.n_dense_layers-2):
            feats = self.act(self.dense_layer_list[i](feats))
        feats = self.out_proj(feats)
        return feats

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
    def __init__(self, d_in_feats, d_out_feats, n_ffn_dense_layers, feat_drop, activation):
        super(Residual, self).__init__()
        self.norm = nn.LayerNorm(d_in_feats)
        self.in_proj = nn.Linear(d_in_feats, d_out_feats)
        self.ffn = MLP(d_out_feats, d_out_feats, n_ffn_dense_layers, activation,d_hidden_feats=d_out_feats*4)
        self.feat_dropout = nn.Dropout(feat_drop)

    def forward(self, triplet_h, attn_message):
        module_input = triplet_h + self.feat_dropout(self.in_proj(attn_message))
        residual = module_input
        module_output = self.feat_dropout(self.ffn(self.norm(module_input)))
        module_output_with_residual = module_output + residual
        return module_output_with_residual

class AtomEmbedding(nn.Module):
    def __init__(
        self,
        d_atom_feats,
        d_g_feats,
        input_drop):
        super(AtomEmbedding, self).__init__()
        self.in_proj = nn.Linear(d_atom_feats, d_g_feats)
        self.virtual_atom_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)
    def forward(self, pair_node_feats, indicators):
        pair_node_h = self.in_proj(pair_node_feats)
        pair_node_h[indicators==VIRTUAL_ATOM_FEATURE_PLACEHOLDER, 1, :] = self.virtual_atom_emb.weight#.half()
        return torch.sum(self.input_dropout(pair_node_h), dim=-2)

class BondEmbedding(nn.Module):
    def __init__(
        self,
        d_bond_feats,
        d_g_feats,
        input_drop):
        super(BondEmbedding, self).__init__()
        self.in_proj = nn.Linear(d_bond_feats, d_g_feats)
        self.virutal_bond_emb = nn.Embedding(1, d_g_feats)
        self.input_dropout = nn.Dropout(input_drop)
    def forward(self, edge_feats, indicators):
        edge_h = self.in_proj(edge_feats)
        edge_h[indicators==VIRTUAL_BOND_FEATURE_PLACEHOLDER] = self.virutal_bond_emb.weight#.half()
        return self.input_dropout(edge_h)

class TripletEmbedding(nn.Module):
    def __init__(
        self,
        d_g_feats,
        d_fp_feats,
        d_md_feats,
        activation=nn.GELU()):
        super(TripletEmbedding, self).__init__()
        self.in_proj = MLP(d_g_feats*2, d_g_feats, 2, activation)
        self.fp_proj = MLP(d_fp_feats, d_g_feats, 2, activation)
        self.md_proj = MLP(d_md_feats, d_g_feats, 2, activation)
    def forward(self, node_h, edge_h, fp, md, indicators):
        triplet_h = torch.cat([node_h, edge_h], dim=-1)
        triplet_h = self.in_proj(triplet_h)
        triplet_h[indicators==1] = self.fp_proj(fp)
        triplet_h[indicators==2] = self.md_proj(md)
        return triplet_h

class TripletTransformer(nn.Module):
    def __init__(self,
                d_feats,
                d_hpath_ratio,
                path_length,
                n_heads,
                n_ffn_dense_layers,
                feat_drop=0.,
                attn_drop=0.,
                activation=nn.GELU()):
        super(TripletTransformer, self).__init__()
        self.d_feats = d_feats
        self.d_trip_path = d_feats//d_hpath_ratio
        self.path_length = path_length
        self.n_heads = n_heads
        self.scale = d_feats**(-0.5)

        self.attention_norm = nn.LayerNorm(d_feats)
        self.qkv = nn.Linear(d_feats, d_feats*3)
        self.node_out_layer = Residual(d_feats, d_feats, n_ffn_dense_layers, feat_drop,  activation)
        
        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.act = activation

    def pretrans_edges(self, edges):
        edge_h = edges.src['hv']
        return {"he": edge_h}

    def forward(self, g, triplet_h, dist_attn, path_attn):
        # 输入：图的整体DGL Graph对象；LiGhTPredictor前序步骤处理的triplet_h特征；LiGhTPredictor前序步骤处理的距离编码注意力dist_attn和路径编码注意力path_attn
        g = g.local_var()

        # 1. 注意力要素计算：
        # 1）triplet_h特征归一化
        new_triplet_h = self.attention_norm(triplet_h)
        # 2）qkv调用一个Linear模型。计算new_triplet_h的qkv
        qkv = self.qkv(new_triplet_h).reshape(-1, 3, self.n_heads, self.d_feats // self.n_heads).permute(1, 0, 2, 3)
        # 3）计算得到注意力要素——分别获取q, k, v
        q, k, v = qkv[0]*self.scale, qkv[1], qkv[2]
        
        # 2. 将注意力信息赋予到图上
        # 1）赋予节点注意力信息：为目标节点赋予K属性，为源节点赋予Q属性
        g.srcdata.update({'Q': q}) # 所有 源节点 赋予q
        g.dstdata.update({'K': k}) # 所有 目标节点 赋予k
        # 2）赋予边的注意力信息：
        # ① 逐边赋予边的node_attn属性
        g.apply_edges(fn.u_dot_v('Q', 'K', 'node_attn'))  # 边上的注意力系数
        # ② 为边的注意力信息引入关联性编码，为边赋予合成注意力信息：节点特征注意力+距离编码注意力+路径编码注意力
        g.edata['a'] = g.edata['node_attn'] + dist_attn.reshape(len(g.edata['node_attn']),-1,1) + path_attn.reshape(len(g.edata['node_attn']),-1,1)
        # ③ 利用合成注意力信息为边赋予softmax注意力
        g.edata['sa'] = self.attn_dropout(edge_softmax(g, g.edata['a']))
        
        # 3. 在图上将注意力信息构造为节点和边的特征信息
        # 1）定义目标节点的特征hv：
        g.ndata['hv'] = v.view(-1, self.d_feats)
        # 2）逐边赋予h特征：利用边的源节点的hv特征（源节点作为目标节点的特征）赋予边he特征→得到最终的边特征：
        g.apply_edges(self.pretrans_edges)
        g.edata['he'] = ((g.edata['he'].view(-1, self.n_heads, self.d_feats//self.n_heads)) * g.edata['sa']).view(-1, self.d_feats)
        
        # 4. 消息传递的最终实现：massage_func——fn.copy_e ; reduce_func——sum
        # 1）消息构建：将g.edata['he']作为消息
        # 2）消息聚合：将消息加到节点的【agg_h】特征上
        g.update_all(fn.copy_e('he', 'm'), fn.sum('m', 'agg_h'))

        # 将triplet_h和聚合的消息输入残差块
        return self.node_out_layer(triplet_h, g.ndata['agg_h'])

    def _device(self):
        return next(self.parameters()).device

class LiGhT(nn.Module):
    def __init__(self,
                d_g_feats,
                d_hpath_ratio,
                path_length,
                n_mol_layers=2,
                n_heads=4,
                n_ffn_dense_layers=4,
                feat_drop=0.,
                attn_drop=0.,
                activation=nn.GELU()):
        super(LiGhT, self).__init__()
        self.n_mol_layers = n_mol_layers
        self.n_heads = n_heads
        self.path_length = path_length
        self.d_g_feats = d_g_feats
        self.d_trip_path = d_g_feats//d_hpath_ratio

        self.mask_emb = nn.Embedding(1, d_g_feats)
        # Distance Attention
        self.path_len_emb = nn.Embedding(path_length+1, d_g_feats)
        self.virtual_path_emb = nn.Embedding(1, d_g_feats)
        self.self_loop_emb = nn.Embedding(1, d_g_feats)
        self.dist_attn_layer = nn.Sequential(
            nn.Linear(self.d_g_feats, self.d_g_feats),
            activation,
            nn.Linear(self.d_g_feats, n_heads)
        )
        # Path Attention  
        self.trip_fortrans = nn.ModuleList([
            MLP(d_g_feats, self.d_trip_path, 2, activation) for _ in range(self.path_length)
        ])
        self.path_attn_layer = nn.Sequential(
            nn.Linear(self.d_trip_path, self.d_trip_path),
            activation,
            nn.Linear(self.d_trip_path, n_heads)
        )
        # Molecule Transformer Layers
        self.mol_T_layers = nn.ModuleList([
            # TripletTransformer模块定义构造：__init__ 方法中接收静态的超参数
            TripletTransformer(d_g_feats, d_hpath_ratio, path_length, n_heads, n_ffn_dense_layers, feat_drop, attn_drop, activation) for _ in range(n_mol_layers)
        ])

        self.feat_dropout = nn.Dropout(p=feat_drop)
        self.attn_dropout = nn.Dropout(p=attn_drop)
        self.act = activation
    
    def _featurize_path(self, g, path_indices):
        mask = (path_indices[:,:]>=0).to(torch.int32)
        path_feats = torch.sum(mask, dim=-1)
        path_feats = self.path_len_emb(path_feats)
        path_feats[g.edata['vp']==1] = self.virtual_path_emb.weight # virtual path
        path_feats[g.edata['sl']==1] = self.self_loop_emb.weight # self loop
        return path_feats    
    
    def _init_path(self, g, triplet_h, path_indices):
        g = g.local_var()
        path_indices[path_indices<-99] = -1
        path_h = []
        for i in range(self.path_length):
            path_h.append(torch.cat([self.trip_fortrans[i](triplet_h),torch.zeros(size=(1,self.d_trip_path)).to(self._device())], dim=0)[path_indices[:, i]])
        path_h = torch.stack(path_h, dim=-1)
        mask = (path_indices>=0).to(torch.int32)
        path_size = torch.sum(mask, dim=-1,keepdim=True)
        path_h = torch.sum(path_h, dim=-1)/path_size
        return path_h

    def forward(self, g, triplet_h):
        path_indices = g.edata['path'] # 使用【路径列表】作为【模型输入】
        dist_h = self._featurize_path(g, path_indices) # 利用路径——生成距离嵌入编码
        path_h = self._init_path(g, triplet_h, path_indices) # 利用路径——生成路径嵌入编码

        dist_attn, path_attn = self.dist_attn_layer(dist_h), self.path_attn_layer(path_h) # 生成距离和路径的两类注意力

        for i in range(self.n_mol_layers):
            # TripletTransformer模块调用传参： 调用forward方法传入的是动态的输入数据         
            # # triplet_h = self.mol_T_layers[i](g, triplet_h, dist_attn, path_attn) 输入：g, triplet_h, dist_attn, path_attn
            triplet_h = self.mol_T_layers[i](g, triplet_h, dist_attn, path_attn)
        return triplet_h

    def _device(self):
        return next(self.parameters()).device

class LiGhTPredictor(nn.Module):
    def __init__(self,
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
                readout_mode='mean'
    ):
        super(LiGhTPredictor, self).__init__()
        self.d_g_feats = d_g_feats
        self.readout_mode=readout_mode
        # Input
        self.node_emb = AtomEmbedding(d_node_feats, d_g_feats, input_drop)
        self.edge_emb = BondEmbedding(d_edge_feats, d_g_feats, input_drop)
        self.triplet_emb  = TripletEmbedding(d_g_feats, d_fp_feats, d_md_feats, activation)
        self.mask_emb = nn.Embedding(1, d_g_feats)
        
        # Model Structure: LiGhT + Predictor
        # LiGhT
        self.model = LiGhT(
            d_g_feats,d_hpath_ratio, path_length, n_mol_layers, n_heads, n_ffn_dense_layers, feat_drop, attn_drop, activation
        )
        # Predictor
        # self.node_predictor = nn.Linear(d_g_feats, n_node_types)
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
        indicators = g.ndata['vavn'] # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes 
        
        # Input: 使用了node_emb/edge_emb/triplet_emb，构造得到triplet_h
        node_h = self.node_emb(g.ndata['begin_end'], indicators)          
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        triplet_h[g.ndata['mask']==1] = self.mask_emb.weight

        # LiGhT
        triplet_h = self.model(g, triplet_h) # 生成transformer的预测结果
        # Predictor
        return self.node_predictor(triplet_h[g.ndata['mask']>=1]), self.fp_predictor(triplet_h[indicators==1]), self.md_predictor(triplet_h[indicators==2])

    def forward_tune(self, g, fp, md):
        indicators = g.ndata['vavn'] # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes 
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)          
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        g.ndata['ht'] = triplet_h
        # Readout
        fp_vn = triplet_h[indicators==1]
        md_vn = triplet_h[indicators==2]
        g.remove_nodes(np.where(indicators.detach().cpu().numpy()>=1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout],dim=-1)
        #Predict
        return self.predictor(g_feats)

    def generate_fps(self, g, fp, md):
        indicators = g.ndata['vavn'] # 0 indicates normal atoms and nodes (triplets); -1 indicates virutal atoms; >=1 indicate virtual nodes 
        # Input
        node_h = self.node_emb(g.ndata['begin_end'], indicators)          
        edge_h = self.edge_emb(g.ndata['edge'], indicators)
        triplet_h = self.triplet_emb(node_h, edge_h, fp, md, indicators)
        # Model
        triplet_h = self.model(g, triplet_h)
        # Readout
        fp_vn = triplet_h[indicators==1]
        md_vn = triplet_h[indicators==2]
        g.ndata['ht'] = triplet_h
        g.remove_nodes(np.where(indicators.detach().cpu().numpy()>=1)[0])
        readout = dgl.readout_nodes(g, 'ht', op=self.readout_mode)
        g_feats = torch.cat([fp_vn, md_vn, readout],dim=-1)
        return g_feats
