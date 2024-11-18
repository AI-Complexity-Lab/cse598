import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1,2))/math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)
    
class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)

class SetEncoder(pl.LightningModule):
    def __init__(self,cfg):
        super(SetEncoder, self).__init__()
        self.linear = cfg.linear
        self.bit16 = cfg.bit16
        self.norm = cfg.norm
        assert cfg.linear != cfg.bit16, "one and only one between linear and bit16 must be true at the same time" 
        if cfg.norm:
            self.register_buffer("mean", torch.tensor(cfg.mean))
            self.register_buffer("std", torch.tensor(cfg.std))
            
        self.activation = cfg.activation
        self.input_normalization = cfg.input_normalization
        if cfg.linear:
            self.linearl = nn.Linear(cfg.dim_input,16*cfg.dim_input)
        self.selfatt = nn.ModuleList()
        #dim_input = 16*dim_input
        self.selfatt1 = ISAB(16*cfg.dim_input, cfg.dim_hidden, cfg.num_heads, cfg.num_inds, ln=cfg.ln)
        for i in range(cfg.n_l_enc):
            self.selfatt.append(ISAB(cfg.dim_hidden, cfg.dim_hidden, cfg.num_heads, cfg.num_inds, ln=cfg.ln))
        self.outatt = PMA(cfg.dim_hidden, cfg.num_heads, cfg.num_features, ln=cfg.ln)


    def float2bit(self, f, num_e_bits=5, num_m_bits=10, bias=127., dtype=torch.float32):
        ## SIGN BIT
        s = (torch.sign(f+0.001)*-1 + 1)*0.5 #Swap plus and minus => 0 is plus and 1 is minus
        s = s.unsqueeze(-1)
        f1 = torch.abs(f)
        ## EXPONENT BIT
        e_scientific = torch.floor(torch.log2(f1))
        e_scientific[e_scientific == float("-inf")] = -(2**(num_e_bits-1)-1)
        e_decimal = e_scientific + (2**(num_e_bits-1)-1)
        e = self.integer2bit(e_decimal, num_bits=num_e_bits)
        ## MANTISSA
        f2 = f1/2**(e_scientific)
        m2 = self.remainder2bit(f2 % 1, num_bits=bias)
        fin_m = m2[:,:,:,:num_m_bits] #[:,:,:,8:num_m_bits+8]
        return torch.cat([s, e, fin_m], dim=-1).type(dtype)

    def remainder2bit(self, remainder, num_bits=127):
        dtype = remainder.type()
        exponent_bits = torch.arange(num_bits, device = self.device).type(dtype)
        exponent_bits = exponent_bits.repeat(remainder.shape + (1,))
        out = (remainder.unsqueeze(-1) * 2 ** exponent_bits) % 1
        return torch.floor(2 * out)

    def integer2bit(self,integer, num_bits=8):
        dtype = integer.type()
        exponent_bits = -torch.arange(-(num_bits - 1), 1, device = self.device).type(dtype)
        exponent_bits = exponent_bits.repeat(integer.shape + (1,))
        out = integer.unsqueeze(-1) / 2 ** exponent_bits
        return (out - (out % 1)) % 2
    
    def forward(self, x):
        if self.bit16:
            x = self.float2bit(x)
            x = x.view(x.shape[0],x.shape[1],-1)
            if self.norm:
                x = (x-0.5)*2
        if self.input_normalization:
            means = x[:,:,-1].mean(axis=1).reshape(-1,1)
            std = x[:,:,-1].std(axis=1).reshape(-1,1)
            std[std==0] = 1
            x[:,:,-1] = (x[:,:,-1] - means)/std
            
        if self.linear:
            if self.activation == 'relu':
                x = torch.relu(self.linearl(x))
            elif self.activation == 'sine':
                x = torch.sin(self.linearl(x))
            else:
                x = (self.linearl(x))
        x = self.selfatt1(x)
        for layer in self.selfatt:
            x = layer(x)
        x = self.outatt(x)
        return x