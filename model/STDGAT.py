import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attentionlayer(nn.Module):
    def __init__(self, args, num_nodes, in_features, out_features):
        super(Attentionlayer, self).__init__()
        self.args = args
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes

        # (in, out)
        self.W = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        nn.init.kaiming_normal_(self.W)

        self.a = nn.Parameter(torch.Tensor(self.out_features*2, 1))
        nn.init.kaiming_normal_(self.a)

    # x (seq_len, batch_size, num_node, in_feature)
    def forward(self, x, adj):
        h = torch.matmul(x, self.W)
        left = torch.unsqueeze(h, dim=3).repeat(1, 1, 1, self.num_nodes, 1)
        right = torch.unsqueeze(h, dim=2).repeat(1, 1, self.num_nodes, 1, 1)
        e = torch.cat((left, right), dim=4)
        e = F.leaky_relu(torch.matmul(e, self.a))
        e = e.view(e.shape[0], e.shape[1], self.num_nodes, -1)
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj>0, e, zero_vec)
        attention = F.softmax(attention, dim=3)
        res = torch.sum(torch.unsqueeze(h, dim=3)*torch.unsqueeze(attention, dim=4), dim=3)
        res = F.leaky_relu(res)
        return res


class STDGAT(nn.Module):
    def __init__(self, args):
        super(STDGAT, self).__init__()
        self.args = args

        self.attlayer1 = Attentionlayer(args=self.args, num_nodes=121, in_features=1, out_features=32)
        self.attlayer2 = Attentionlayer(args=self.args, num_nodes=121, in_features=32, out_features=32)
        self.attlayer3 = Attentionlayer(args=self.args, num_nodes=121, in_features=32, out_features=32)

        self.rnn = nn.LSTM(input_size=32*121, hidden_size=512)

        self.fc = nn.Linear(in_features=512, out_features=121)


    def forward(self, x, features):
        x = torch.transpose(x, 0, 1)
        att_out = self.attlayer1.forward(x=x, adj=features)
        att_out = self.attlayer2.forward(x=att_out, adj=features)
        att_out = self.attlayer3.forward(x=att_out, adj=features)
        
        att_out = att_out.view(att_out.shape[0], att_out.shape[1], -1)
        rnn_out, hn = self.rnn(att_out)
        rnn_out = rnn_out[-1]
        out = self.fc(rnn_out)
        out = F.relu(out)
        out = out.unsqueeze(dim=2)
        return out

class GAT(nn.Module):
    def __init__(self, args):
        super(GAT, self).__init__()
        self.args = args

        self.attlayer1 = Attentionlayer(args=self.args, num_nodes=121, in_features=1, out_features=32)
        self.attlayer2 = Attentionlayer(args=self.args, num_nodes=121, in_features=32, out_features=32)
        self.attlayer3 = Attentionlayer(args=self.args, num_nodes=121, in_features=32, out_features=32)

        self.fc1 = nn.Linear(in_features=32*121, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=121)


    def forward(self, x, features):
        att_out = self.attlayer1.forward(x=x, adj=features)
        att_out = self.attlayer2.forward(x=att_out, adj=features)
        att_out = self.attlayer3.forward(x=att_out, adj=features)

        att_out = att_out.view(att_out.shape[0], att_out.shape[1], -1)

        out = self.fc1(att_out)
        out = F.relu(out)
        out = torch.mean(out, dim=0)
        out = self.fc2(out)
        out = F.relu(out)
        out = out.unsqueeze(dim=2)

        return out
