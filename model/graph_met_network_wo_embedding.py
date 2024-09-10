import torch
import torch_geometric
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import EdgeConv
from torch_cluster import knn_graph

class GraphMETNetwork(nn.Module):
    def __init__(self, continuous_dim, cat_dim, norm, output_dim=1, hidden_dim=32, conv_depth=1):
        super(GraphMETNetwork, self).__init__()
       
        self.datanorm = norm

        self.embed_charge = nn.Embedding(3, hidden_dim//4) # 3 -> 8
        self.embed_pdgid = nn.Embedding(7, hidden_dim//4) # 7 -> 8
        
        # Removed embed_continuous since it is not used

        self.embed_categorical = nn.Sequential(nn.Linear(2 * hidden_dim // 4, hidden_dim // 2),
                                               nn.ELU()) # 16 -> 16

        self.encode_all = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                        nn.ELU()) # 32 -> 32
        self.bn_all = nn.BatchNorm1d(hidden_dim) # 32
 
        self.conv_continuous = nn.ModuleList()        
        for i in range(conv_depth):
            mesg = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim))
            self.conv_continuous.append(nn.ModuleList())
            self.conv_continuous[-1].append(EdgeConv(nn=mesg).jittable())
            self.conv_continuous[-1].append(nn.BatchNorm1d(hidden_dim))

        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2),
                                    nn.ELU(), # 32 -> 16
                                    nn.Linear(hidden_dim // 2, output_dim)) # 16 -> 1

        self.pdgs = [1, 2, 11, 13, 22, 130, 211]

    def forward(self, x_cont, x_cat, edge_index, batch):
        x_cont *= self.datanorm

        # Embedding categorical features
        emb_chrg = self.embed_charge(x_cat[:, 1] + 1)

        pdg_remap = torch.abs(x_cat[:, 0])
        for i, pdgval in enumerate(self.pdgs):
            pdg_remap = torch.where(pdg_remap == pdgval, torch.full_like(pdg_remap, i), pdg_remap)
        emb_pdg = self.embed_pdgid(pdg_remap)

        emb_cat = self.embed_categorical(torch.cat([emb_chrg, emb_pdg], dim=1))
        
        combined_emb = torch.cat([emb_cat, x_cont], dim=1)
        
        emb = self.bn_all(self.encode_all(combined_emb))

                
        # Graph convolution for categorical variables
        for co_conv in self.conv_continuous:
            emb = emb + co_conv[1](co_conv[0](emb, edge_index))
                
        out = self.output(emb)
        
        return out.squeeze(-1)
