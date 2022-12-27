import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

'''
Input: (bs, seg_len, Nverts, 64)
Encoder output: (bs, seg_len, Nverts, 64) -->  (atten_head, 64, d_k) x 2 [K, V]
                                          -->  (bs, z_dim)
Decoder output: (bs, seg_len, Nverts, 43)
'''

def get_sinusoid_pos_encoding(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_idx) for hid_idx in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)


def get_mask(seg_batch):
    ''' For masking out the subsequent info. '''
    bs, seg_len, n_verts, _ = seg_batch.size()
    subsequent_mask = torch.triu(
        torch.ones((seg_len, seg_len), device=seg_batch.device, dtype=torch.bool), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(bs, -1, -1)

    return subsequent_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_in, d_k, d_v):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_in, n_head * d_k)
        self.w_k = nn.Linear(d_in, n_head * d_k)
        self.w_v = nn.Linear(d_in, n_head * d_v)

        self.temperature = np.power(d_k, 0.5)
        self.attn_dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(n_head * d_v, d_in)

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters(d_in, d_k, d_v)

    def reset_parameters(self, d_in, d_k, d_v):
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0 / (d_in + d_k)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0 / (d_in + d_k)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0 / (d_in + d_v)))
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x, mask=None):
        # input: (bs, seg_len, Nverts, d_in), mask: (bs, seg_len, seg_len)
        bs, seg_len, n_verts, hidden_dim = x.shape

        residual = x

        q = self.w_q(x).view(bs, seg_len, n_verts, self.n_head, self.d_k).permute(3, 2, 0, 1, 4).contiguous().view(-1, seg_len, self.d_k)  # (n_head*Nverts*bs, seg_len, d_k)
        k = self.w_k(x).view(bs, seg_len, n_verts, self.n_head, self.d_k).permute(3, 2, 0, 1, 4).contiguous().view(-1, seg_len, self.d_k)
        v = self.w_v(x).view(bs, seg_len, n_verts, self.n_head, self.d_v).permute(3, 2, 0, 1, 4).contiguous().view(-1, seg_len, self.d_v)   # (n_head*Nverts*bs, seg_len, d_v)

        attn = torch.bmm(q, k.transpose(1, 2))  # (n_head*Nverts*bs, seg_len, seg_len)
        attn = attn / self.temperature

        if mask is not None:
            mask = mask.repeat(self.n_head * n_verts, 1, 1)  # (n_head*Nverts*bs, seg_len, seg_len)
            attn = attn.masked_fill(mask == 0, -np.inf)

        if (mask is not None and mask.sum() == 0):
            attn[:, :, :] = 0
        else:
            attn = F.softmax(attn, dim=2)  # (n_head*Nverts*bs, seg_len, seg_len)

        attn = self.attn_dropout(attn)
        output = torch.bmm(attn, v)  # (n_head*Nverts*bs, seg_len, d_v)

        output = output.view(self.n_head, n_verts, bs, seg_len, self.d_v)
        output = output.permute(2, 3, 1, 0, 4).contiguous().view(bs, seg_len, n_verts, -1)  # (bs, seg_len, Nverts, n_head*d_v)

        output = self.dropout(self.fc(output))  # (bs, seg_len, Nverts, d_in)
        output = self.layer_norm(output + residual)  # (bs, seg_len, n_verts, d_in)

        return output

class MultiHeadEncDecAttention(nn.Module):
    def __init__(self, n_head, d_in, d_q):
        super(MultiHeadEncDecAttention, self).__init__()

        self.n_head = n_head
        self.d_in = d_in
        self.d_q = d_q

        self.w_q = nn.Linear(d_in, n_head * d_q)
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0 / (d_in + d_q)))

        self.temperature = np.power(d_q, 0.5)
        self.attn_dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(n_head * d_q, d_in)
        nn.init.xavier_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(d_in)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, z, mask=None):
        # input: (bs, seg_len, Nverts, d_in), z: (bs * n_head * n_verts, seg_len, d_in), mask: (bs, seg_len, seg_len)
        bs, seg_len, n_verts, hidden_dim = x.shape

        residual = x

        q = self.w_q(x).view(bs, seg_len, n_verts, self.n_head, self.d_q).permute(3, 2, 0, 1, 4).contiguous().view(-1, seg_len, self.d_q)  # (n_head*Nverts*bs, seg_len, d_q)

        attn = torch.bmm(q, z.transpose(1, 2))  # (n_head*Nverts*bs, seg_len, seg_len)
        attn = attn / self.temperature

        if mask is not None:
            mask = mask.repeat(self.n_head * n_verts, 1, 1)  # (n_head*Nverts*bs, seg_len, seg_len)
            attn = attn.masked_fill(mask, -np.inf)

        attn = F.softmax(attn, dim=2)  # (n_head*Nverts*bs, seg_len, seg_len)
        attn = self.attn_dropout(attn)
        output = torch.bmm(attn, z)  # (n_head*Nverts*bs, seg_len, d_v)

        output = output.view(self.n_head, n_verts, bs, seg_len, -1)
        output = output.permute(2, 3, 1, 0, 4).contiguous().view(bs, seg_len, n_verts, -1)  # (bs, seg_len, Nverts, n_head*d_v)

        output = self.dropout(self.fc(output))  # (bs, seg_len, Nverts, d_in)
        output = self.layer_norm(output + residual)  # (bs, seg_len, Nverts, d_in)

        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(0.1)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.w_1.weight)
        torch.nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x):
        # x: (bs, seg_len, n_verts, d_in)
        bs, seg_len, n_verts, d_in = x.shape
        residual = x
        output = x.view(bs, -1, d_in).transpose(1, 2) # (bs, d_in, seg_len*n_verts)
        output = self.w_2(F.relu(self.w_1(output))) # (bs, d_in, seg_len*n_verts)
        output = output.transpose(1, 2).view(bs, seg_len, n_verts, -1)  # (bs, seg_len, n_verts, d_in)
        output = self.dropout(output)
        output = self.layer_norm(output + residual) # (bs, seg_len, n_verts, d_in)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, n_head, d_in, d_k, d_v):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_in, d_k, d_v)
        self.pos_wise_ffnn = PositionwiseFeedForward(d_in, d_in)

    def forward(self, x, mask=None):
        x = self.self_attn(x, mask)
        x = self.pos_wise_ffnn(x)
        # if mask is not None:
        #     output_mask = mask[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        #     x *= output_mask
        return x


class DecoderLayer(nn.Module):
    def __init__(self, n_head, d_in, d_k, d_v):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(n_head, d_in, d_k, d_v)
        self.pos_wise_ffnn = PositionwiseFeedForward(d_in, d_in)

    def forward(self, x, mask=None):
        x = self.self_attn(x, mask)
        x = self.pos_wise_ffnn(x)
        # if mask is not None:
        #     output_mask = mask[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        #     x *= output_mask
        return x


class DecoderLayer_1(nn.Module):
    def __init__(self, n_head, d_in, d_k, d_v):
        super(DecoderLayer_1, self).__init__()
        # self.self_attn = MultiHeadAttention(n_head, d_in, d_k, d_v)
        self.encdec_attn = MultiHeadEncDecAttention(n_head, d_in, d_in)
        self.pos_wise_ffnn = PositionwiseFeedForward(d_in, d_in)

    def forward(self, x, z, mask=None):
        # z: (bs, seg_len, n_verts, d_z * n_head)
        # x = self.self_attn(x, mask)
        x = self.encdec_attn(x, z, mask)
        x = self.pos_wise_ffnn(x)
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_layers, n_head, d_in, d_k, d_v, d_z, seg_len, n_verts, upsampling_matrices):
        super(TransformerEncoder, self).__init__()
        self.n_head = n_head
        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v
        self.upsampling_matrices = upsampling_matrices
        self.pos_codebook = nn.Embedding.from_pretrained(get_sinusoid_pos_encoding(seg_len, d_in), freeze=True) # (seg_len, d_in)
        self.encoder_layers = nn.ModuleList([EncoderLayer(n_head, d_in, d_k, d_v) for _ in range(n_layers)])

        self.latent_mapping = nn.Linear(seg_len * d_in, d_z)
        self.latent_norm = nn.LayerNorm(d_z)
        self.output_w_k = nn.Linear(d_in, n_head * d_k)
        self.output_w_v = nn.Linear(d_in, n_head * d_v)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.latent_mapping.weight)
        nn.init.normal_(self.output_w_k.weight, mean=0, std=np.sqrt(2.0 / (self.d_in + self.d_k)))
        nn.init.normal_(self.output_w_v.weight, mean=0, std=np.sqrt(2.0 / (self.d_in + self.d_v)))

    def forward(self, x):
        bs, seg_len, n_verts, d_in = x.shape
        pos_vec = torch.arange(seg_len).unsqueeze(0).repeat(bs, 1).to(x.device)  # (bs, seg_len)
        pos_emb = self.pos_codebook(pos_vec)    # (bs, seg_len, d_in)
        pos_emb = pos_emb.unsqueeze(2).repeat(1, 1, n_verts, 1) # (bs, seg_len, n_verts, d_in)
        x = pos_emb + x
        for enc_layer in self.encoder_layers:
            x = enc_layer(x)
        z = torch.max(x, dim=2)[0].view(bs, -1)    # (bs, seg_len * d_in)
        z = self.latent_mapping(z)  # (bs, d_z)
        z = self.latent_norm(z)
        z = F.relu(z)

        # output_k = self.output_w_k(x).view(bs, seg_len, n_verts, self.n_head, self.d_k).permute(0, 1, 3, 4, 2)
        # output_k = torch.matmul(output_k, self.upsampling_matrices[2].T)
        # output_k = torch.matmul(output_k, self.upsampling_matrices[1].T)    # (bs, seg_len, n_head, d_k, n_verts)
        # output_k = output_k.permute(2, 4, 0, 1, 3).contiguous().view(-1, seg_len, self.d_k)  # (n_head*Nverts*bs, seg_len, d_k)
        #
        # output_v = self.output_w_v(x).view(bs, seg_len, n_verts, self.n_head, self.d_v).permute(0, 1, 3, 4, 2)
        # output_v = torch.matmul(output_v, self.upsampling_matrices[2].T)
        # output_v = torch.matmul(output_v, self.upsampling_matrices[1].T)    # (bs, seg_len, n_head, d_k, n_verts)
        # output_v = output_v.permute(2, 4, 0, 1, 3).contiguous().view(-1, seg_len, self.d_v)  # (n_head*Nverts*bs, seg_len, d_b)
        return z


class TransformerDecoder(nn.Module):
    def __init__(self, n_layers, n_head, d_in, d_out, d_k, d_v, seg_len):
        super(TransformerDecoder, self).__init__()
        self.pos_codebook = nn.Embedding.from_pretrained(get_sinusoid_pos_encoding(seg_len, d_in),
                                                         freeze=True)  # (seg_len, d_in)
        self.decoder_layers = nn.ModuleList([DecoderLayer(n_head, d_in, d_k, d_v) for _ in range(n_layers)])
        self.final_lin = nn.Linear(d_in, d_out)
        self.reset_paremeters()

    def reset_paremeters(self):
        torch.nn.init.xavier_uniform_(self.final_lin.weight)

    def forward(self, x, mask=None):
        bs, seg_len, n_verts, d_in = x.shape
        pos_vec = torch.arange(seg_len).unsqueeze(0).repeat(bs, 1).to(x.device)  # (bs, seg_len)
        pos_emb = self.pos_codebook(pos_vec)  # (bs, seg_len, d_in)
        pos_emb = pos_emb.unsqueeze(2).repeat(1, 1, n_verts, 1)  # (bs, seg_len, n_verts, d_in)
        x = pos_emb + x
        for dec_layer in self.decoder_layers:
            x = dec_layer(x, mask)
        x = self.final_lin(x)
        # if mask is not None:
        #     output_mask = mask[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
        #     x *= output_mask
        return x

class TransformerEncoder_1(nn.Module):
    def __init__(self, n_layers, n_head, d_in, d_k, d_v, d_z, seg_len):
        super(TransformerEncoder_1, self).__init__()
        self.n_head = n_head
        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v
        self.pos_codebook = nn.Embedding.from_pretrained(get_sinusoid_pos_encoding(seg_len + 2, d_in), freeze=True) # (seg_len, d_in)
        self.encoder_layers = nn.ModuleList([EncoderLayer(n_head, d_in, d_k, d_v) for _ in range(n_layers)])
        self.latent_tokens = nn.Embedding(2, d_in)
        self.enc_mu = nn.Linear(d_in, d_z)
        self.enc_logvar = nn.Linear(d_in, d_z)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.latent_mapping.weight)
        pass

    def forward(self, x, mask=None):
        bs, seg_len, n_verts, d_in = x.shape
        pos_vec = torch.arange(seg_len + 2).unsqueeze(0).repeat(bs, 1).to(x.device)  # (bs, seg_len + 2)
        pos_emb = self.pos_codebook(pos_vec)    # (bs, seg_len + 2, d_in)
        pos_emb = pos_emb.unsqueeze(2).repeat(1, 1, n_verts, 1) # (bs, seg_len + 2, n_verts, d_in)
        extra_tokens = self.latent_tokens(torch.LongTensor([0, 1]).to(x.device))
        extra_tokens = extra_tokens.unsqueeze(0).unsqueeze(2).expand(bs, -1, n_verts, -1)
        x = torch.cat([extra_tokens, x], dim=1)
        x = pos_emb + x
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, mask)                    # (bs, seg_len + 2, n_verts, d_in)

        mu = torch.max(x[:, 0, :, :], dim=1)[0] # (bs, d_in)
        mu = self.enc_mu(mu)    # (bs, d_z)
        logvar = torch.max(x[:, 1, :, :], dim=1)[0]
        logvar = self.enc_logvar(logvar)
        return mu, logvar


class TransformerDecoder_1(nn.Module):
    def __init__(self, n_layers, n_head, d_in, d_out, d_k, d_v, seg_len):
        super(TransformerDecoder_1, self).__init__()
        self.pos_codebook = nn.Embedding.from_pretrained(get_sinusoid_pos_encoding(seg_len, d_in),
                                                         freeze=True)  # (seg_len, d_in)
        self.decoder_layers = nn.ModuleList([DecoderLayer_1(n_head, d_in, d_k, d_v) for _ in range(n_layers)])
        self.final_lin = nn.Linear(d_in, d_out)
        self.reset_paremeters()

    def reset_paremeters(self):
        torch.nn.init.xavier_uniform_(self.final_lin.weight)

    def forward(self, x, z, mask=None):
        bs, seg_len, n_verts, d_in = x.shape
        pos_vec = torch.arange(seg_len).unsqueeze(0).repeat(bs, 1).to(x.device)  # (bs, seg_len)
        pos_emb = self.pos_codebook(pos_vec)  # (bs, seg_len, d_in)
        pos_emb = pos_emb.unsqueeze(2).repeat(1, 1, n_verts, 1)  # (bs, seg_len, n_verts, d_in)
        x = pos_emb + x
        for dec_layer in self.decoder_layers:
            x = dec_layer(x, z, mask)
        x = self.final_lin(x)
        return x


class TransformerEncoder_2(nn.Module):
    def __init__(self, n_layers, n_head, d_in, d_k, d_v, d_z, d_prev, seg_len):
        super(TransformerEncoder_2, self).__init__()
        self.n_head = n_head
        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v
        self.pos_codebook = nn.Embedding.from_pretrained(get_sinusoid_pos_encoding(seg_len + 2, d_in), freeze=True) # (seg_len, d_in)
        self.encoder_layers = nn.ModuleList([EncoderLayer(n_head, d_in, d_k, d_v) for _ in range(n_layers)])
        self.latent_tokens = nn.Embedding(2, d_in)
        self.enc_mu = nn.Linear(d_in + d_prev, d_z)
        self.enc_logvar = nn.Linear(d_in + d_prev, d_z)
        self.reset_parameters()

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.latent_mapping.weight)
        pass

    def forward(self, x, prev_feat, mask=None):
        bs, seg_len, n_verts, d_in = x.shape
        pos_vec = torch.arange(seg_len + 2).unsqueeze(0).repeat(bs, 1).to(x.device)  # (bs, seg_len + 2)
        pos_emb = self.pos_codebook(pos_vec)    # (bs, seg_len + 2, d_in)
        pos_emb = pos_emb.unsqueeze(2).repeat(1, 1, n_verts, 1) # (bs, seg_len + 2, n_verts, d_in)
        extra_tokens = self.latent_tokens(torch.LongTensor([0, 1]).to(x.device))
        extra_tokens = extra_tokens.unsqueeze(0).unsqueeze(2).expand(bs, -1, n_verts, -1)
        x = torch.cat([extra_tokens, x], dim=1)
        x = pos_emb + x
        for enc_layer in self.encoder_layers:
            x = enc_layer(x, mask)                    # (bs, seg_len + 2, n_verts, d_in)

        mu = torch.max(x[:, 0, :, :], dim=1)[0] # (bs, d_in)
        mu = self.enc_mu(torch.cat((mu, prev_feat), dim=-1))    # (bs, d_z)
        logvar = torch.max(x[:, 1, :, :], dim=1)[0]
        logvar = self.enc_logvar(torch.cat((logvar, prev_feat), dim=-1))
        return mu, logvar


class TransformerDecoder_2(nn.Module):
    def __init__(self, n_layers, n_head, d_in, d_out, d_k, d_v, d_prev, seg_len):
        super(TransformerDecoder_2, self).__init__()
        self.pos_codebook = nn.Embedding.from_pretrained(get_sinusoid_pos_encoding(seg_len, d_in),
                                                         freeze=True)  # (seg_len, d_in)
        self.decoder_layers = nn.ModuleList([DecoderLayer(n_head, d_in, d_k, d_v) for _ in range(n_layers)])
        self.final_lin = nn.ModuleList()
        self.final_lin.append(nn.Linear(d_in + d_prev, d_in))    # I'm hard coding this
        self.final_lin.append(nn.Linear(d_in, d_out))
        self.final_lin = nn.Sequential(*self.final_lin)
        self.reset_paremeters()

    def reset_paremeters(self):
        pass

    def forward(self, x, prev_feat, mask=None):
        # prev_feat: (bs, f_prev)
        bs, seg_len, n_verts, d_in = x.shape
        pos_vec = torch.arange(seg_len).unsqueeze(0).repeat(bs, 1).to(x.device)  # (bs, seg_len)
        pos_emb = self.pos_codebook(pos_vec)  # (bs, seg_len, d_in)
        pos_emb = pos_emb.unsqueeze(2).repeat(1, 1, n_verts, 1)  # (bs, seg_len, n_verts, d_in)
        x = pos_emb + x
        for dec_layer in self.decoder_layers:
            x = dec_layer(x, mask)
        prev_feat = prev_feat.unsqueeze(1).unsqueeze(1).expand(-1, seg_len, n_verts, -1)
        x = self.final_lin(torch.cat((x, prev_feat), dim=-1))
        return x



