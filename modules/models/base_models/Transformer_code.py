import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def a_norm(Q, K):
    # m = torch.matmul(Q, K.transpose(2,1).float())
    m = torch.matmul(Q, K.transpose(3, 2).float())
    m /= torch.sqrt(torch.tensor(Q.shape[-1]).float())

    return torch.softmax(m, -1)


def attention(Q, K, V):
    # Attention(Q, K, V) = norm(QK)V
    a = a_norm(Q, K)  # (batch_size, dim_attn, seq_length)

    return torch.matmul(a, V)  # (batch_size, seq_length, seq_length)


class AttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn):
        super(AttentionBlock, self).__init__()
        self.value = Value(dim_val, dim_val)  # [10,10]
        self.key = Key(dim_val, dim_attn)  # [10,5]
        self.query = Query(dim_val, dim_attn)

    def forward(self, x, kv=None):
        if (kv is None):
            # Attention with x connected to Q,K and V (For encoder)
            return attention(self.query(x), self.key(x), self.value(x))

        # Attention with x as Q, external vector kv as K an V (For decoder)
        return attention(self.query(x), self.key(kv), self.value(kv))


class MultiHeadAttentionBlock(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads):
        super(MultiHeadAttentionBlock, self).__init__()
        self.heads = []
        for i in range(n_heads):
            self.heads.append(AttentionBlock(dim_val, dim_attn))

        self.heads = nn.ModuleList(self.heads)

        self.fc = nn.Linear(n_heads * dim_val, dim_val, bias=False)

    def forward(self, x, kv=None):
        a = []
        for h in self.heads:
            c = h(x, kv=kv)
            a.append(c)

        a = torch.stack(a, dim=-1)  # combine heads
        # a = a.flatten(start_dim = 2) #flatten all head outputs
        a = a.flatten(start_dim=3)  # flatten all head outputs

        x = self.fc(a)

        return x


class Value(torch.nn.Module):
    def __init__(self, dim_input, dim_val):
        super(Value, self).__init__()
        self.dim_val = dim_val

        self.fc1 = nn.Linear(dim_input, dim_val, bias=False)
        # self.fc2 = nn.Linear(5, dim_val)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)

        return x


class Key(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Key, self).__init__()
        self.dim_attn = dim_attn

        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)
        # self.fc2 = nn.Linear(5, dim_attn)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)

        return x


class Query(torch.nn.Module):
    def __init__(self, dim_input, dim_attn):
        super(Query, self).__init__()
        self.dim_attn = dim_attn

        self.fc1 = nn.Linear(dim_input, dim_attn, bias=False)
        # self.fc2 = nn.Linear(5, dim_attn)

    def forward(self, x):
        x = self.fc1(x)
        # print(x.shape)
        # x = self.fc2(x)

        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)  # 5000*10
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        pp = self.pe[:x.size(2), :].squeeze(1)  # [6,10]
        x = x + pp  # [15，6,10]
        return x


class EncoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)

    def forward(self, x):
        a = self.attn(x)
        x = self.norm1(x + a)

        a = self.fc1(F.elu(self.fc2(x)))
        x = self.norm2(x + a)

        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, n_heads=1):
        super(DecoderLayer, self).__init__()
        self.attn1 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.attn2 = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)

        self.norm1 = nn.LayerNorm(dim_val)
        self.norm2 = nn.LayerNorm(dim_val)
        self.norm3 = nn.LayerNorm(dim_val)

    def forward(self, x, enc):
        a = self.attn1(x)
        x = self.norm1(a + x)

        a = self.attn2(x, kv=enc)
        x = self.norm2(a + x)

        a = self.fc1(F.elu(self.fc2(x)))

        x = self.norm3(x + a)  # [15,2,10]
        return x


class Transformer(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, input_size, dec_seq_len, out_seq_len, n_decoder_layers=1, n_encoder_layers=1,
                 n_heads=1):
        super(Transformer, self).__init__()
        self.dec_seq_len = dec_seq_len

        # Initiate encoder and Decoder layers
        self.encs = nn.ModuleList()
        for i in range(n_encoder_layers):
            self.encs.append(EncoderLayer(dim_val, dim_attn, n_heads))

        self.decs = nn.ModuleList()
        for i in range(n_decoder_layers):
            self.decs.append(DecoderLayer(dim_val, dim_attn, n_heads))

        self.pos = PositionalEncoding(dim_val)  # dim_val=10

        # Dense layers for managing network inputs and outputs
        self.enc_input_fc = nn.Linear(input_size, dim_val)
        self.dec_input_fc = nn.Linear(input_size, dim_val)
        self.out_fc = nn.Linear(dec_seq_len * dim_val, out_seq_len)

    def forward(self, x):
        # encoder
        a = self.enc_input_fc(x)
        b = self.pos(a)
        e = self.encs[0](b)  # 输入第一个编码器 [15，6，10]

        # e = self.encs[0](self.pos(self.enc_input_fc(x)))
        for enc in self.encs[1:]:
            e = enc(e)

        # decoder
        dd = x[:, :, -self.dec_seq_len:, :]  # [15,2,1]
        ddd = self.dec_input_fc(dd)  # [15,2,10]
        # d = self.decs[0](self.dec_input_fc(x[:,-self.dec_seq_len:]), e)  # [15,2,10]、[15,6,10]
        d = self.decs[0](ddd, e)
        for dec in self.decs[1:]:
            d = dec(d, e)  # [15,2,10]、[15,6,10]

        # output
        cccc = d.flatten(start_dim=2)  # [15，20]
        x = self.out_fc(cccc)

        return x
