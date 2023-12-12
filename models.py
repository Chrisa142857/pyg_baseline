import torch
from torch import nn
from torch_geometric.nn import GCNConv, GCN2Conv, GATConv, SAGEConv, TransformerConv


class ToyMPNN(nn.Module):
    def __init__(self, CONV_OP, nlayer, inch, outch, hidch, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.CONV_OP = CONV_OP
        self.enc = CONV_OP(inch, hidch)
        self.dec = CONV_OP(hidch, outch)
        self.net = nn.ModuleList([CONV_OP(hidch, hidch) for i in range(nlayer)])
        self.nlayer = nlayer

    def forward(self, x, edge_index0):
        if self.CONV_OP == GCN2Conv:
            x = self.enc(x, x, edge_index0)
        else:
            x = self.enc(x, edge_index0)
        x0 = x
        for i, net in enumerate(self.net):
            edge_index = edge_index0
            if self.CONV_OP == GCN2Conv:
                x = net(x, x0, edge_index)
            else:
                x = net(x, edge_index) 
            x = torch.relu(x)
        if self.CONV_OP == GCN2Conv:
            x = self.dec(x, x0, edge_index0)
        else:
            x = self.dec(x, edge_index0)
        return x
    
