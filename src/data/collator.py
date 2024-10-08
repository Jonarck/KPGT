import dgl
import torch
import numpy as np
from copy import deepcopy
from .featurizer import smiles_to_graph

def preprocess_batch_light(batch_num, batch_num_target, tensor_data):
    batch_num = np.concatenate([[0],batch_num],axis=-1)
    cs_num = np.cumsum(batch_num)
    add_factors = np.concatenate([[cs_num[i]]*batch_num_target[i] for i in range(len(cs_num)-1)], axis=-1)
    return tensor_data + torch.from_numpy(add_factors).reshape(-1,1)

class Collator_pretrain(object):
    def __init__(
        self, 
        vocab, 
        max_length, n_virtual_nodes, add_self_loop=True,
        candi_rate=0.15, mask_rate=0.8, replace_rate=0.1, keep_rate=0.1,
        fp_disturb_rate=0.15, md_disturb_rate=0.15
        ):
        self.vocab = vocab
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop

        self.candi_rate = candi_rate # 候选节点的比例，用于生成掩蔽或替换节点的候选集。
        self.mask_rate = mask_rate # 掩蔽节点的比例，指从候选节点中选择掩蔽节点的比例
        self.replace_rate = replace_rate # 替换节点的比例，指从候选节点中选择替换节点的比例。
        self.keep_rate = keep_rate # 保留节点的比例，指从候选节点中选择保留节点的比例。

        self.fp_disturb_rate = fp_disturb_rate # 分子指纹扰动率，用于扰动分子指纹。
        self.md_disturb_rate = md_disturb_rate # 分子描述符扰动率，用于扰动分子描述符。

    # 自监督训练任务1的数据生成: 掩藏或替换节点,用于预测
    def bert_mask_nodes(self, g):
        n_nodes = g.number_of_nodes()
        all_ids = np.arange(0, n_nodes, 1, dtype=np.int64) # 生成一个包含所有节点索引的数组。

        valid_ids = torch.where(g.ndata['vavn']<=0)[0].numpy()  # 【实际三元组节点】在DGL对象的节点数据里的id
        valid_labels = g.ndata['label'][valid_ids].numpy() # 【实际三元组节点】的vocab索引
        # 设定【节点选择概率】：与某节点vocab类型相同的节点数量越多，选择该节点的概率越小
        probs = np.ones(len(valid_labels))/len(valid_labels) # 初始化选择【实际三元组节点】的概率
        unique_labels = np.unique(np.sort(valid_labels))
        for label in unique_labels:
            label_pos = (valid_labels==label)
            probs[label_pos] = probs[label_pos]/np.sum(label_pos) # 更新选择【实际三元组节点】的概率：！！与该节点三元组类型相同的节点数量越多，选择该节点的概率越小！！
        probs = probs/np.sum(probs)
        
        # 在全体【实节点】中选择【候选节点】：数量为【实际三元组节点】总数的 candi_rate，根据【节点选择概率】 probs 选择。
        candi_ids = np.random.choice(valid_ids, size=int(len(valid_ids)*self.candi_rate),replace=False, p=probs)
        
        # 在【候选节点】中选择【遮挡节点】：
        mask_ids = np.random.choice(candi_ids, size=int(len(candi_ids)*self.mask_rate),replace=False)
        
        # 从【候选节点】中移除【遮挡节点】
        candi_ids = np.setdiff1d(candi_ids, mask_ids)

        # 在【剩余候选节点】中选择【替换节点】：
        replace_ids = np.random.choice(candi_ids, size=int(len(candi_ids)*(self.replace_rate/(1-self.keep_rate))),replace=False)
        
        # 从【候选节点】中移除【替换节点】。剩下的【候选节点】即为保留节点
        keep_ids = np.setdiff1d(candi_ids, replace_ids)
        
        # 设定：
        g.ndata['mask'] = torch.zeros(n_nodes,dtype=torch.long)
        g.ndata['mask'][mask_ids] = 1
        g.ndata['mask'][replace_ids] = 2
        g.ndata['mask'][keep_ids] = 3

        # 自监督训练任务1的【标签】
        sl_labels = g.ndata['label'][g.ndata['mask']>=1].clone()

        # Pre-replace
        new_ids = np.random.choice(valid_ids, size=len(replace_ids),replace=True, p=probs)
        replace_labels = g.ndata['label'][replace_ids].numpy()
        new_labels = g.ndata['label'][new_ids].numpy()
        is_equal = (replace_labels == new_labels)
        while(np.sum(is_equal)):
            new_ids[is_equal] = np.random.choice(valid_ids, size=np.sum(is_equal),replace=True, p=probs)
            new_labels = g.ndata['label'][new_ids].numpy()
            is_equal = (replace_labels == new_labels)
        g.ndata['begin_end'][replace_ids] = g.ndata['begin_end'][new_ids].clone()
        g.ndata['edge'][replace_ids] = g.ndata['edge'][new_ids].clone()
        g.ndata['vavn'][replace_ids] = g.ndata['vavn'][new_ids].clone()
        
        return sl_labels
    
    # 自监督训练任务2的数据生成：知识掩码，两类知识预测任务
    # 1. 分子指纹的知识掩码数据增益：
    def disturb_fp(self, fp):
        fp = deepcopy(fp)
        b, d = fp.shape
        fp = fp.reshape(-1)
        disturb_ids = np.random.choice(b*d, int(b*d*self.fp_disturb_rate), replace=False)
        fp[disturb_ids] = 1 - fp[disturb_ids]
        return fp.reshape(b,d)
    # 2. 分子描述符的知识掩码数据增益：
    def disturb_md(self, md):
        md = deepcopy(md)
        b, d = md.shape
        md = md.reshape(-1)
        sampled_ids = np.random.choice(b*d, int(b*d*self.md_disturb_rate), replace=False)
        a = torch.empty(len(sampled_ids)).uniform_(0, 1)
        sampled_md = a
        md[sampled_ids] = sampled_md
        return md.reshape(b,d)
    

    # 具体的训练数据构建过程：
    def __call__(self, samples):
        # 引入原始数据
        smiles_list, fps, mds = map(list, zip(*samples))
        graphs = []

        # 预处理数据
        # 1. 调用smiles_to_graph将每个分子构建为一个含有各种特征信息的DGL线对象，并成组化
        for smiles in smiles_list:
            graphs.append(smiles_to_graph(smiles, self.vocab, max_length=self.max_length, n_virtual_nodes=self.n_virtual_nodes, add_self_loop=self.add_self_loop))
        batched_graph = dgl.batch(graphs)

        # 2. 数据形式重整：
        # 2.1.重整两类知识的数据形式
        mds = torch.stack(mds, dim=0).reshape(len(smiles_list),-1)
        fps = torch.stack(fps, dim=0).reshape(len(smiles_list),-1)
        # 2.2.重整dgl分子线图的路径
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(), batched_graph.batch_num_edges(), batched_graph.edata['path'][:, :])
        '''
        def preprocess_batch_light(batch_num, batch_num_target, tensor_data):
            batch_num = np.concatenate([[0],batch_num],axis=-1)
            cs_num = np.cumsum(batch_num)
            add_factors = np.concatenate([[cs_num[i]]*batch_num_target[i] for i in range(len(cs_num)-1)], axis=-1)
            return tensor_data + torch.from_numpy(add_factors).reshape(-1,1)
        '''

        # 训练数据
        sl_labels = self.bert_mask_nodes(batched_graph) # 被掩蔽或替换节点的原始标签
        disturbed_fps = self.disturb_fp(fps) # 修改后的分子指纹
        disturbed_mds = self.disturb_md(mds) # 修改后的分子描述符

        '''
        ---------------------------------------------
        smiles_list
        ----------------------------------------------
        batched_graph：一批【分子线图】
        fps：一批【分子指纹】
        mds：一批【分子描述符】
        ----------------------------------------------
        sl_labels：一批【被掩蔽或替换节点的原始标签】
        disturbed_fps：一批【修改后的分子指纹】
        disturbed_mds：一批【修改后的分子描述符】
        ---------------------------------------------
        '''
        return smiles_list, batched_graph, fps, mds, sl_labels, disturbed_fps, disturbed_mds

class Collator_tune(object):
    def __init__(self, max_length=5, n_virtual_nodes=2, add_self_loop=True):
        self.max_length = max_length
        self.n_virtual_nodes = n_virtual_nodes
        self.add_self_loop = add_self_loop
    def __call__(self, samples):
        smiles_list, graphs, fps, mds, labels = map(list, zip(*samples))

        batched_graph = dgl.batch(graphs)
        fps = torch.stack(fps, dim=0).reshape(len(smiles_list),-1)
        mds = torch.stack(mds, dim=0).reshape(len(smiles_list),-1)
        labels = torch.stack(labels, dim=0).reshape(len(smiles_list),-1)
        batched_graph.edata['path'][:, :] = preprocess_batch_light(batched_graph.batch_num_nodes(), batched_graph.batch_num_edges(), batched_graph.edata['path'][:, :])
        return smiles_list, batched_graph, fps, mds, labels
