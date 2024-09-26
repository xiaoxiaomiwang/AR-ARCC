from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F

class HeterogeneousSelfAttention(nn.Module):
    def __init__(self, input_dim, heads=8):
        super(HeterogeneousSelfAttention, self).__init__()
        self.heads = heads
        self.input_dim = input_dim

        self.query1 = nn.Linear(input_dim, input_dim * heads, bias=False)
        self.key1 = nn.Linear(input_dim, input_dim * heads, bias=False)
        self.value1 = nn.Linear(input_dim, input_dim * heads, bias=False)

        self.query2 = nn.Linear(input_dim, input_dim * heads, bias=False)
        self.key2 = nn.Linear(input_dim, input_dim * heads, bias=False)
        self.value2 = nn.Linear(input_dim, input_dim * heads, bias=False)

        self.fc_out = nn.Linear(2*input_dim * heads, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        N, in_dim = x.shape


        Q1 = self.query1(x).view(N, in_dim, self.heads)
        K1 = self.key1(x).view(N, in_dim, self.heads)
        V1 = self.value1(x).view(N, in_dim, self.heads)


        Q2 = self.query2(x).view(N, in_dim, self.heads)
        K2 = self.key2(x).view(N, in_dim, self.heads)
        V2 = self.value2(x).view(N, in_dim, self.heads)


        Q1_swapped = Q1+K2
        Q2_swapped = Q2+K1
        K1_swapped = K2
        K2_swapped = K1

        attention1 = torch.einsum("nqk,nkh->nqk", [Q1_swapped, K1_swapped.transpose(1, 2)])
        attention1 = F.softmax(attention1 / (in_dim ** 0.5), dim=2)
        OUT1 = torch.einsum("nqk,nkh->nqk", [attention1, V1.transpose(1, 2)]).view(N, self.heads * in_dim)

        attention2 = torch.einsum("nqk,nkh->nqk", [Q2_swapped, K2_swapped.transpose(1, 2)])
        attention2 = F.softmax(attention2 / (self.input_dim ** 0.5), dim=-1)
        OUT2 = torch.einsum("nqk,nkh->nqk", [attention2, V2.transpose(1, 2)]).view(N, self.heads * in_dim)

        # 拼接输出
        final_output = torch.cat([OUT1, OUT2], dim=-1)
        out = self.fc_out(final_output)
        out = self.dropout(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            HeterogeneousSelfAttention(in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, in_features)
        )
    def forward(self, x):
        return x + self.block(x)

class NN(nn.Module, ABC):


    def __init__(self, input_shape=500, output_size=1):
        super(NN, self).__init__()


        self.hidden_layer1 = nn.Sequential(nn.Linear(in_features=input_shape, out_features=100))
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.1)
        self.residual = ResidualBlock(100)
        self.hidden_layer2 = nn.Sequential(nn.Linear(in_features=100, out_features=100), nn.Tanh())
        self.drop2 = nn.Dropout(p=0.1)
        self.out = nn.Linear(100, output_size)

    def forward(self, x):


        x = self.hidden_layer1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.residual(x)
        x = self.hidden_layer2(x)
        x = self.drop2(x)
        return self.out(x)

