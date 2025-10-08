import torch
import torch_geometric

from torch import nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GraphConv, EdgeConv, GCNConv

from torch_cluster import radius_graph, knn_graph

class GraphMETNetwork(nn.Module):
    def __init__ (self, continuous_dim, cat_dim, norm, output_dim=1, hidden_dim=32, conv_depth=1):
    #def __init__ (self, continuous_dim, cat_dim, output_dim=1, hidden_dim=32, conv_depth=1):
        super(GraphMETNetwork, self).__init__()
       
        self.datanorm = norm

        self.embed_charge = nn.Embedding(3, hidden_dim//4)
        self.embed_pdgid = nn.Embedding(7, hidden_dim//4)
        #self.embed_pv = nn.Embedding(8, hidden_dim//4)
        
        self.embed_continuous = nn.Sequential(nn.Linear(continuous_dim,hidden_dim//2),
                                              nn.ELU(),
                                              #nn.BatchNorm1d(hidden_dim//2) # uncomment if it starts overtraining
                                             )

        self.embed_categorical = nn.Sequential(nn.Linear(2*hidden_dim//4,hidden_dim//2),
                                               nn.ELU(),                                               
                                               #nn.BatchNorm1d(hidden_dim//2)
                                              )

        self.encode_all = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                        nn.ELU()
                                       )
        self.bn_all = nn.BatchNorm1d(hidden_dim)
 
        self.conv_continuous = nn.ModuleList()        
        for i in range(conv_depth):
            mesg = nn.Sequential(nn.Linear(2*hidden_dim, hidden_dim))
            self.conv_continuous.append(nn.ModuleList())
            self.conv_continuous[-1].append(EdgeConv(nn=mesg).jittable())
            self.conv_continuous[-1].append(nn.BatchNorm1d(hidden_dim))

        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim//2),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim//2, output_dim)
                                   )
        self.pdgs = [1, 2, 11, 13, 22, 130, 211]

    def forward(self, x_cont, x_cat, edge_index, edge_attr, batch):
        # Apply normalization and embeddings
        x_cont *= self.datanorm

        emb_cont = self.embed_continuous(x_cont)        
        emb_chrg = self.embed_charge(x_cat[:, 1] + 1)

        pdg_remap = torch.abs(x_cat[:, 0])
        for i, pdgval in enumerate(self.pdgs):
            pdg_remap = torch.where(pdg_remap == pdgval, torch.full_like(pdg_remap, i), pdg_remap)
        emb_pdg = self.embed_pdgid(pdg_remap)

        emb_cat = self.embed_categorical(torch.cat([emb_chrg, emb_pdg], dim=1))
        emb = self.bn_all(self.encode_all(torch.cat([emb_cat, emb_cont], dim=1)))

        # --- new: edge-aware message passing ---
        for co_conv in self.conv_continuous:
            conv, bn = co_conv
            # EdgeConv ignores edge_attr, so we can inject it manually:
            row, col = edge_index
            edge_input = torch.cat([
                emb[row],          # source
                emb[col],          # target
                edge_attr          # edge features
            ], dim=1)

            # Replace co_conv[0](emb, edge_index) with explicit message MLP call
            message = conv.nn(edge_input)  # pass through MLP manually
            # Aggregate by mean (EdgeConv’s default aggregation)
            out = torch_scatter.scatter_mean(message, col, dim=0, dim_size=emb.size(0))

            emb = emb + bn(out)

        out = self.output(emb)
        return out.squeeze(-1)

