import torch
import torch.nn as nn
import torch.nn.init as init
# import numpy as np
# from utils import cal_cos_similarity
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils



def cal_cos_similarity(x, y): # the 2nd dimension of x and y are the same
    xy = x.mm(torch.t(y))
    x_norm = torch.sqrt(torch.sum(x*x, dim =1)).reshape(-1, 1)
    y_norm = torch.sqrt(torch.sum(y*y, dim =1)).reshape(-1, 1)
    cos_similarity = xy/x_norm.mm(torch.t(y_norm))
    cos_similarity[cos_similarity != cos_similarity] = 0
    return cos_similarity
    
    
def cal_batch_cos_similarity(x, y): 
    # x - bs, len, dim
    # y - bs, dim
    xy = torch.bmm(x, y.unsqueeze(2)) # bs, len, 1
    x_norm = torch.sqrt(torch.sum(x*x, dim =2)) # bs, len
    y_norm = torch.sqrt(torch.sum(y*y, dim =1)).unsqueeze(1) # bs, 1
    xy_norm = (x_norm*y_norm).unsqueeze(2) #bs, len, 1
    cos_similarity = xy
    # cos_similarity = xy/xy_norm
    cos_similarity[cos_similarity != cos_similarity] = 0
    return cos_similarity # bs, len, 1



# the HIST module for prediction
class GRAPHPREPROCESS2(nn.Module):
    def __init__(self, input_size=936, hidden_size=64, dropout=0.0, d_model=128, K=3, n_nodes = 12):
        super().__init__()

        self.K = K
        self.n_nodes = n_nodes
        self.hidden_size = hidden_size
        self.d_model = hidden_size
        self.fc_ps = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps.weight)
        self.fc_hs = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs.weight)


        self.fc_ps_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps_back.weight)
        self.fc_hs_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs_back.weight)
        self.fc_indi = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi.weight)

        self.leaky_relu = nn.LeakyReLU()
        # self.softmax_s2t = torch.nn.Softmax(dim = 0)
        self.softmax_t2s = torch.nn.Softmax(dim = 1)
        
        
        self.embedding = nn.Linear(input_size, self.d_model)
        torch.nn.init.xavier_uniform_(self.embedding.weight)
        self.init_encoder = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=True)
        
        
        
        self.mapping_node = nn.Linear(self.d_model,16)
        self.mapping_graph = nn.Linear(self.d_model,self.d_model)
        self.adj = nn.Linear(self.d_model+(n_nodes+K)*16,(n_nodes+K)**2)
        torch.nn.init.xavier_uniform_(self.mapping_node.weight)
        torch.nn.init.xavier_uniform_(self.mapping_graph.weight)
        torch.nn.init.xavier_uniform_(self.adj.weight)

    # def forward(self, x_hidden, concept_matrix, market_value):
    # def forward(self, node_feats, length_annos, K=2):
    def forward(self, graph_feats, scores):
        '''
        graph_feats  - input graph features, (batch_size, dim)
        scores       - scores of graph nodes, (batch_size, n_nodes)
        K            - the number of hidden nodes in each graph
        
        node_feats   - input node embeddings, (len, dim) len denotes the number of nodes of all the graphs in the batch
        length_annos - the graph index that each node belongs (len,) e.g., [0,0,1,1,1,2,2,...]
        adj_matrices - adj matrices (len, n_nodes, n_nodes)
        '''
        K = self.K
        K_local = 5
        device = torch.device(torch.get_device(graph_feats))
        # node_length = torch.bincount(length_annos, minlength=length_annos.max())  # bs,
        # node_length = node_length[node_length != 0]
        # print(node_length)
        # bs = len(node_length)
        node_length = scores.size()[1]
        if node_length != self.n_nodes:
          print('node_length != self.n_nodes')
        
        bs = scores.size()[0]
        
        
        graph_feats = self.embedding(graph_feats)
        scores[scores==float("-inf")] = -1e10
        norm_score = self.softmax_t2s(scores.t()) # (node_length, bs)
        node_feats = norm_score.mm(graph_feats.to(torch.float64)) # (node_length, dim)
        
        # print('!!!'*10)
        # print(scores)
        # aa = torch.rand(bs, node_length, self.d_model).to(device)
        #predefined node representation
        alpha_p = cal_cos_similarity(node_feats,graph_feats.to(torch.float64)) # node_length,bs
        alpha_p = self.softmax_t2s(alpha_p)
        node_feats = alpha_p.mm(graph_feats.to(torch.float64)) # node_length,d_model
        
        beta_p = cal_cos_similarity(graph_feats.to(torch.float64), node_feats) #bs, node_length
        beta_p = self.softmax_t2s(beta_p) #bs, node_length
        graph_feats_back = beta_p.mm(node_feats) #bs, d_model
        graph_feats_back = self.fc_ps(graph_feats_back.to(torch.float32))
        graph_feats_back = self.fc_ps_back(graph_feats_back)
        
        all_node_feats = node_feats.expand(bs, node_length, self.d_model)
        
        all_node_feats = beta_p.unsqueeze(2)*all_node_feats #bs, node_length, d_model
        # aa = all_node_feats
        
        # hidden nodes discovery
        graph_feats = graph_feats-graph_feats_back
        hidden_feats = graph_feats
        alpha_h = cal_cos_similarity(graph_feats, hidden_feats)
        
        dim = alpha_h.shape[0]
        diag = alpha_h.diagonal(0)
        alpha_h = alpha_h * (torch.ones(dim, dim) - torch.eye(dim)).to(device)
        if K_local>dim: K_local=dim
        
        row = torch.linspace(0, dim-1, dim).reshape([-1, 1]).repeat(1, K_local).reshape(1, -1).long().to(device)
        column = torch.topk(alpha_h, K_local, dim = 1)[1].reshape(1, -1)
        mask = torch.zeros([alpha_h.shape[0], alpha_h.shape[1]], device = alpha_h.device)
        mask[row, column] = 1
        alpha_h = alpha_h * mask
        alpha_h = alpha_h + torch.diag_embed((alpha_h.sum(0)!=0).float()*diag)
        hidden_feats = torch.t(graph_feats).mm(alpha_h).t()
        hidden_feats = hidden_feats[hidden_feats.sum(1)!=0] # n_hidden, d_model
        
        
        beta_h = cal_cos_similarity(graph_feats, hidden_feats) # bs, n_hidden(<=bs)
        # row = torch.linspace(0, dim-1, dim).reshape([-1, 1]).repeat(1, K_local).reshape(1, -1).long().to(device)
        # column = torch.topk(beta_h, K_local, dim = 1)[1].reshape(1, -1) # select top K_local hidden feats
        # mask = torch.zeros([beta_h.shape[0], beta_h.shape[1]], device = beta_h.device)
        # mask[row, column] = 1
        # beta_h = beta_h * mask
        beta_h = self.softmax_t2s(beta_h)
        graph_feats_back_2 = beta_h.mm(hidden_feats)
        graph_feats_back_2 = self.fc_hs(graph_feats_back_2)
        graph_feats_back_2 = self.fc_hs_back(graph_feats_back_2)
        
        
        n_hidden = hidden_feats.size()[0]
        all_hidden_feats = hidden_feats.expand(bs, n_hidden, self.d_model)
        all_hidden_feats = beta_h.unsqueeze(2)*all_hidden_feats #bs, n_hidden, d_model
        
        
        # select topK nodes for each graph - bs, K, d_model
        topk_idx = torch.topk(beta_h, K, dim = 1)[1] # bs, K
        all_hidden_feats =  torch.gather(all_hidden_feats, 1, topk_idx.unsqueeze(-1).expand(-1, -1, self.d_model)) # bs, K, d_model
        
        
        all_node_feats = torch.cat([all_node_feats,all_hidden_feats],1)# bs, node_length+K, d_model
        
        map_node_feats = self.mapping_node(all_node_feats.to(torch.float32)).view(bs, -1) # bs, (node_length+K)*16
        # print(map_node_feats.size())
        
        all_graph_feats = self.mapping_graph(graph_feats_back_2+graph_feats_back) # bs, d_model
        # print(all_graph_feats.size())
        adj = self.adj(torch.cat([all_graph_feats,map_node_feats],1))
        
        adj = torch.sigmoid(adj).view(bs,node_length+K, node_length+K)
        
        
        # adj = torch.sigmoid(torch.rand(bs,node_length+K, node_length+K)).to(device)
        adj = 0.5 * (adj + adj.transpose(1,2))
        adj = adj * (1 - torch.eye(node_length+K).to(device))
        # adj = adj * (1 - torch.eye(node_length).to(device))
        
        
        output_indi  = graph_feats - graph_feats_back_2
        output_indi = self.fc_indi(output_indi)
        output_indi = self.leaky_relu(output_indi) # bs, d_model
        
        
        length_annos = torch.tensor(range(bs)).to(device)
        length_annos = length_annos.expand(node_length+K,bs)
        # length_annos = length_annos.expand(node_length,bs)
        
        length_annos = length_annos.t().reshape(-1)
        
        xx = torch.rand(output_indi.size()).to(device)
        # aa = torch.cat([aa,torch.rand(all_hidden_feats.size()).to(device)],1)

        # return all_node_feats, length_annos, adj, xx # add output_indi to the final graph feature before the predictor
        return all_node_feats, length_annos, adj, output_indi # add output_indi to the final graph feature before the predictor

if __name__ == '__main__':
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = GRAPHPREPROCESS2()
    model.to(device)
    graph_feats = torch.rand(10,256).to(device)
    scores = torch.rand(10,12).to(device)
    a,b,c,d = model(graph_feats, scores)
