import os
import torch
import torch.utils.data
from torch import nn
import numpy as np
from models_common import *
import transformer
from posa_models import POSA

# tf_1_256_top_jump8

"""
###############################################################################
The ContactFormer and its variants presented in the paper
###############################################################################
"""
class ContactFormer(nn.Module):
    def __init__(self, seg_len, encoder_mode, decoder_mode, n_layer=6, n_head=8, f_vert=64, dim_ff=512,
                 d_hid=512, mesh_ds_dir="../data/mesh_ds", posa_path=None, **kwargs):
        super(ContactFormer, self).__init__()
        self.seg_len = seg_len
        self.encoder_mode = encoder_mode
        self.decoder_mode = decoder_mode
        self.posa = POSA(ds_us_dir=mesh_ds_dir, use_semantics=True, channels=f_vert)
        if posa_path is not None:
            checkpoint = torch.load(posa_path)
            self.posa.load_state_dict(checkpoint['model_state_dict'])
            for p in self.posa.parameters():
                p.requires_grad = False

        if self.decoder_mode == 0:
            for p in self.posa.parameters():
                p.requires_grad = True
        elif self.decoder_mode == 1:
            self.decoder = TransformerDecoder(seg_len=seg_len, n_layer=n_layer, n_head=n_head, f_vert=f_vert, dim_ff=dim_ff,
                                              d_hid=d_hid)
        elif self.decoder_mode == 2:
            self.decoder = TransformerDecoder2(seg_len=seg_len, n_layer=n_layer, n_head=n_head, f_vert=f_vert, dim_ff=dim_ff,
                                               d_hid=d_hid)
        elif self.decoder_mode == 3:
            self.decoder = MLPDecoder3(seg_len=seg_len, d_hid=d_hid)
        elif self.decoder_mode == 4:
            self.decoder = LSTMDecoder4(seg_len=seg_len, n_layer=n_layer, dim_ff=dim_ff, d_hid=d_hid)

    def forward(self, cf, vertices, mask):
        # vertices: (bs, seg_len, Nverts, 3), mask: (bs, seg_len)
        vertices = vertices.squeeze()
        cf = cf.squeeze()
        posa_out, mu, logvar = self.posa(cf, vertices)  # (seg_len, nv, 8)

        if self.decoder_mode == 0:
            out = posa_out.unsqueeze(0)
        else:
            out = self.decoder(posa_out, mask)
        return out, mu.unsqueeze(0), logvar.unsqueeze(0)


class TransformerDecoder(nn.Module):
    def __init__(self, seg_len, n_layer=6, n_head=8, f_vert=64, dim_ff=512,
                 d_hid=512, add_virtual_node=False, **kwargs):
        super(TransformerDecoder, self).__init__()
        self.seg_len = seg_len
        self.frame_emb_linear = nn.Linear(655 * 8, d_hid)
        self.relu = nn.ReLU()
        self.pos_codebook = nn.Embedding.from_pretrained(
            transformer.get_sinusoid_pos_encoding(seg_len, d_hid),
            freeze=True)
        self.tf_decoder = nn.Transformer(d_model=d_hid, nhead=n_head, num_encoder_layers=n_layer,
                                      num_decoder_layers=n_layer, dim_feedforward=dim_ff)
        self.out_linear = nn.ModuleList()
        self.out_linear.append(nn.Linear(8 + d_hid, d_hid // 2))
        self.out_linear.append(self.relu)
        self.out_linear.append(nn.Linear(d_hid // 2, 8))
        self.out_linear = nn.Sequential(*self.out_linear)

    def forward(self, posa_out, mask):
        tf_in = posa_out.reshape(posa_out.shape[0], -1)  # (seg_len, nv * 8)
        tf_in = self.relu(self.frame_emb_linear(tf_in)).unsqueeze(0)  # (1, seg_len, d_hid)
        pos_vec = torch.arange(self.seg_len).unsqueeze(0).repeat(1, 1).to(tf_in.device)  # (bs, seg_len)
        pos_emb = self.pos_codebook(pos_vec)  # (bs, seg_len, d_hid)
        tf_in = tf_in + pos_emb
        tf_in = tf_in.permute(1, 0, 2)  # (seg_len, bs, d_hid)
        mask = mask > 0
        tf_out = self.tf_decoder(tf_in, tf_in, src_key_padding_mask=~mask, tgt_key_padding_mask=~mask,
                                memory_key_padding_mask=~mask)  # (seg_len, bs, d_hid)
        tf_out = tf_out.permute(1, 0, 2)  # (bs, seg_len, d_hid)
        tf_out = tf_out.unsqueeze(2).expand(-1, -1, posa_out.shape[1], -1)  # (bs, seg_len, nv, d_hid)
        tf_out = torch.cat((posa_out.unsqueeze(0), tf_out), dim=-1)
        out = self.out_linear(tf_out)
        return out


class TransformerDecoder2(nn.Module):
    def __init__(self, seg_len, n_layer=6, n_head=8, f_vert=64, dim_ff=512,
                 d_hid=512, add_virtual_node=False, **kwargs):
        super(TransformerDecoder2, self).__init__()
        self.seg_len = seg_len
        self.frame_emb_linear = nn.Linear(655 * 8, d_hid)
        self.relu = nn.ReLU()
        self.pos_codebook = nn.Embedding.from_pretrained(
            transformer.get_sinusoid_pos_encoding(seg_len, d_hid),
            freeze=True)
        self.transformerLayer = nn.TransformerEncoderLayer(d_model=d_hid, nhead=n_head,
                                                            dim_feedforward=dim_ff)
        self.tf_decoder = nn.TransformerEncoder(self.transformerLayer, num_layers=n_layer)
        self.out_linear = nn.ModuleList()
        self.out_linear.append(nn.Linear(8 + d_hid, d_hid // 2))
        self.out_linear.append(nn.ReLU())
        self.out_linear.append(nn.Linear(d_hid // 2, 8))
        self.out_linear = nn.Sequential(*self.out_linear)

    def forward(self, posa_out, mask):
        tf_in = posa_out.reshape(posa_out.shape[0], -1)  # (seg_len, nv * 8)
        tf_in = self.frame_emb_linear(tf_in)
        tf_in = self.relu(tf_in)  # (1, seg_len, d_hid)
        tf_in = tf_in.unsqueeze(0)
        pos_vec = torch.arange(self.seg_len).unsqueeze(0).repeat(1, 1).to(tf_in.device)  # (1, seg_len)
        pos_emb = self.pos_codebook(pos_vec)  # (1, seg_len, d_hid)
        tf_in = tf_in + pos_emb
        tf_in = tf_in.permute(1, 0, 2)  # (seg_len, bs, d_hid)
        mask = mask > 0
        tf_out = self.tf_decoder(tf_in, src_key_padding_mask=~mask)  # (seg_len, bs, d_hid)
        tf_out = tf_out.permute(1, 0, 2)  # (bs, seg_len, d_hid)
        tf_out = tf_out.unsqueeze(2).expand(-1, -1, posa_out.shape[1], -1)  # (bs, seg_len, nv, d_hid)
        tf_out = torch.cat((posa_out.unsqueeze(0), tf_out), dim=-1)
        out = self.out_linear(tf_out)
        return out

class MLPDecoder3(nn.Module):
    def __init__(self, seg_len, d_hid=512, add_virtual_node=False, **kwargs):
        super(MLPDecoder3, self).__init__()
        self.seg_len = seg_len
        self.frame_emb_linear = nn.Linear(655 * 8, d_hid)
        self.relu = nn.ReLU()

        self.mlp_block = nn.ModuleList()
        self.mlp_block.append(nn.Linear(d_hid, d_hid * 2))
        self.mlp_block.append(nn.ReLU())
        self.mlp_block.append(nn.Linear(d_hid * 2, d_hid))
        self.mlp_block.append(nn.ReLU())
        self.mlp_block = nn.Sequential(*self.mlp_block)

        self.out_linear = nn.ModuleList()
        self.out_linear.append(nn.Linear(8 + d_hid, d_hid // 2))
        self.out_linear.append(nn.ReLU())
        self.out_linear.append(nn.Linear(d_hid // 2, 8))
        self.out_linear = nn.Sequential(*self.out_linear)

    def forward(self, posa_out, mask=None):
        mlp_in = posa_out.reshape(posa_out.shape[0], -1)  # (seg_len, nv * 8)
        mlp_in = self.frame_emb_linear(mlp_in)
        mlp_in = self.relu(mlp_in)  # (seg_len, d_hid)

        mlp_out = self.mlp_block(mlp_in)    # (seg_len, d_hid)

        mlp_out = mlp_out.unsqueeze(1).expand(-1, posa_out.shape[1], -1)  # (seg_len, nv, d_hid)
        out = torch.cat((posa_out, mlp_out), dim=-1)
        out = self.out_linear(out)
        return out.unsqueeze(0)

class LSTMDecoder4(nn.Module):
    def __init__(self, seg_len, n_layer=1, dim_ff=512,
                 d_hid=512, add_virtual_node=False, **kwargs):
        super(LSTMDecoder4, self).__init__()
        self.seg_len = seg_len
        self.frame_emb_linear = nn.Linear(655 * 8, d_hid)
        self.relu = nn.ReLU()

        self.lstm = nn.LSTM(d_hid, dim_ff, num_layers=n_layer, batch_first=True, bidirectional=True)
        self.h0 = nn.Parameter(torch.randn(n_layer * 2, 1, dim_ff))
        self.c0 = nn.Parameter(torch.randn(n_layer * 2, 1, dim_ff))
        self.bidir = nn.Linear(dim_ff * 2, d_hid)

        self.out_linear = nn.ModuleList()
        self.out_linear.append(nn.Linear(8 + d_hid, d_hid // 2))
        self.out_linear.append(nn.ReLU())
        self.out_linear.append(nn.Linear(d_hid // 2, 8))
        self.out_linear = nn.Sequential(*self.out_linear)

    def forward(self, posa_out, mask):
        lstm_in = posa_out.reshape(posa_out.shape[0], -1)  # (seg_len, nv * 8)
        lstm_in = self.frame_emb_linear(lstm_in)
        lstm_in = self.relu(lstm_in)
        lstm_in = lstm_in.unsqueeze(0)  # (1, seg_len, d_hid)

        mask = mask.unsqueeze(-1).expand(-1, -1, lstm_in.shape[-1])
        lstm_in = lstm_in * mask
        lstm_out, (ho, co) = self.lstm(lstm_in, (self.h0, self.c0))  # (bs, seg_len, d_hid * 2)
        lstm_out = self.bidir(lstm_out)
        lstm_out = self.relu(lstm_out)

        lstm_out = lstm_out.unsqueeze(2).expand(-1, -1, posa_out.shape[1], -1)  # (bs, seg_len, nv, d_hid)
        out = torch.cat((posa_out.unsqueeze(0), lstm_out), dim=-1)
        out = self.out_linear(out)
        return out

"""
###############################################################################
The legacy versions for ContactFormer. Code kept here for your reference. Feel
Free to play around and experiment with them.
###############################################################################
"""
'''
Modified POSA + Customizable Tranformer. This version does not skip any frames in motion
sequences. It assumes a motion sequence is split into several non-overlapping segments. 
Each time it predicts contact labels for some segment. Then it simply concatenates
predictions for those segments.
'''
class POSA_temp_transformer(nn.Module):
    def __init__(self, seg_len, encoder_mode, decoder_mode, n_layer=6, n_head=8, add_virtual_node=False, **kwargs):
        super(POSA_temp_transformer, self).__init__()
        self.seg_len = seg_len
        self.encoder = GNNEncoder(self.seg_len, encoder_mode=encoder_mode, n_layer=n_layer, n_head=n_head,
                                  add_virtual_node=add_virtual_node, **kwargs)
        self.decoder = GNNDecoder(self.seg_len, decoder_mode=decoder_mode, n_layer=n_layer, n_head=n_head,
                                  add_virtual_node=add_virtual_node, **kwargs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, cf, vertices):
        # For temp_posa, vertices: (bs, seg_len, Nverts, 3)
        mu, logvar = self.encoder(cf, vertices)  # mu, logvar = (bs, 256)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z, vertices)
        return out, mu, logvar

class GNNEncoder(nn.Module):
    def __init__(self, seg_len, h_dim=512, z_dim=256, channels=64, ds_us_dir='../data/mesh_ds',
                 normalization_mode='group_norm', num_groups=8, seq_length=9, no_obj_classes=8, use_cuda=True,
                 encoder_mode=0, n_layer=6, n_head=8, add_virtual_node=False, **kwargs):
        super(GNNEncoder, self).__init__()

        # Here seg_len means the number of frames concatenated as the input of the temporal POSA,
        # seq_length is the number of neighbor nodes considered in GCN.
        # num_groups for GroupNorm layer
        self.encoder_mode = encoder_mode
        self.seg_len = seg_len
        self.f_dim = no_obj_classes
        self.spiral_indices = []
        self.nv = []
        self.D = []
        self.U = []
        levels = [0, 1, 2]
        for level in levels:
            nv, spiral_indices, _, U, D, _ = load_ds_us_param(ds_us_dir, level, seq_length, use_cuda)
            self.nv.append(nv)  # nv: [655, 164, 41]
            self.D.append(D)    # D: [(655, 2619),(164, 655),(41, 164)]
            self.U.append(U)
            self.spiral_indices.append(spiral_indices) # spiral_indices: (655->164->41, 9)
        self.channels = (channels * np.ones(4)).astype(np.int).tolist() # [64, 64, 64, 64]

        self.en_spiral = nn.ModuleList()
        self.en_spiral.append(
            Spiral_block(3 + self.f_dim, self.channels[0], self.spiral_indices[0], normalization_mode, num_groups,
                         add_virtual_node=add_virtual_node))
        for i in levels:
            self.en_spiral.append(
                Spiral_block(self.channels[i], self.channels[i + 1], self.spiral_indices[i], normalization_mode,
                             num_groups, add_virtual_node=add_virtual_node))
            if i != len(levels) - 1:
                self.en_spiral.append(ds_us_fn(self.D[i + 1]))

        self.en_spiral = nn.Sequential(*self.en_spiral)
        self.en_mu = nn.Linear(h_dim, z_dim)
        self.en_log_var = nn.Linear(h_dim, z_dim)
        if encoder_mode == 0:
            self.en_transformer = transformer.TransformerEncoder(n_layers=n_layer, n_head=n_head, d_in=channels, d_k=channels,
                                                                d_v=channels, d_z=h_dim, seg_len=seg_len,
                                                                n_verts=self.nv[-1], upsampling_matrices=self.U)
        else:
            self.en_transformer = transformer.TransformerEncoder_1(n_layers=n_layer, n_head=n_head, d_in=channels,
                                                                 d_k=channels, d_v=channels, d_z=z_dim, seg_len=seg_len)


    def forward(self, x, vertices):
        x = torch.cat((vertices, x), dim=-1)    # x: (bs, seg_len, Nv, 46)
        x = self.en_spiral(x)   # x: (bs, seg_len, Nv, 64)
        if self.encoder_mode == 0:
            z = self.en_transformer(x)
            return self.en_mu(z), self.en_log_var(z)
        else:
            mu, logvar = self.en_transformer(x)
            return mu, logvar


class GNNDecoder(nn.Module):
    def __init__(self, seg_len, z_dim=256, num_hidden_layers=3, channels=64, ds_us_dir='../data/mesh_ds',
                 normalization_mode='group_norm', num_groups=8, seq_length=9, no_obj_classes=8, use_cuda=True,
                 decoder_mode=0, n_layer=6, n_head=8, add_virtual_node=False, **kwargs):
        super(GNNDecoder, self).__init__()
        self.decoder_mode = decoder_mode
        self.num_hidden_layers = num_hidden_layers
        self.f_dim = no_obj_classes
        self.spiral_indices = []
        self.nv = []
        levels = [0, 1, 2]
        for level in levels:
            nv, spiral_indices, _, _, _, _ = load_ds_us_param(ds_us_dir, level, seq_length, use_cuda)
            self.nv.append(nv)
            self.spiral_indices.append(spiral_indices)
        self.channels = (channels * np.ones(4)).astype(np.int).tolist()

        self.de_spiral = nn.ModuleList()
        self.de_spiral.append(GraphLin_block(3 + z_dim, z_dim // 2, normalization_mode, num_groups))
        self.de_spiral.append(GraphLin_block(z_dim // 2, self.channels[0], normalization_mode, num_groups))
        for _ in range(self.num_hidden_layers):
            self.de_spiral.append(
                Spiral_block(self.channels[0], self.channels[0], self.spiral_indices[0], normalization_mode,
                             num_groups, add_virtual_node=add_virtual_node))
        # self.de_spiral.append(SpiralConv(self.channels[0], self.channels[0], self.spiral_indices[0],
        #                                  add_virtual_node=add_virtual_node))
        self.de_spiral = nn.Sequential(*self.de_spiral)

        self.de_transformer = transformer.TransformerDecoder(n_layers=n_layer, n_head=n_head, d_in=channels, d_out=self.f_dim,
                                                             d_k=channels, d_v=channels, seg_len=seg_len)

    def forward(self, x, vertices):
        x = x.unsqueeze(1)
        x = x.unsqueeze(1).expand((-1, vertices.shape[1], self.nv[0], -1))
        x = torch.cat((vertices, x), dim=-1)
        x = self.de_spiral(x)

        x = self.de_transformer(x)
        return x

'''
Modified POSA + Customizable Tranformer. This version does not skip any frames in motion
sequences. It assumes a motion sequence is split into several non-overlapping segments. 
The model sequentially takes these segments as input. For some intermediate segment, the model
takes both the body pose info for the current segment and the prediction from previous segment
as the input.
'''
class POSA_temp_transformer_seq(nn.Module):
    def __init__(self, seg_len, num_seg, encoder_mode, decoder_mode, n_layer=6, n_head=8, add_virtual_node=False,
                 prev_frame_feat_dim=64, no_obj_classes=8, **kwargs):
        super(POSA_temp_transformer_seq, self).__init__()
        self.seg_len = seg_len
        self.num_seg = num_seg
        self.encoder_mode = encoder_mode
        self.decoder_mode = decoder_mode
        self.encoder = Encoder_seq(self.seg_len, prev_feat_dim=prev_frame_feat_dim, encoder_mode=encoder_mode, n_layer=n_layer, n_head=n_head,
                               add_virtual_node=add_virtual_node, **kwargs)
        self.decoder = Decoder_seq(self.seg_len, prev_feat_dim=prev_frame_feat_dim, decoder_mode=decoder_mode, n_layer=n_layer, n_head=n_head,
                               add_virtual_node=add_virtual_node, **kwargs)
        self.prev_frame_feat_dim = prev_frame_feat_dim
        if decoder_mode == 2:
            self.out_linear = nn.Linear(seg_len * no_obj_classes, prev_frame_feat_dim)
        else:
            self.out_linear = nn.Linear(no_obj_classes, prev_frame_feat_dim)
        self.relu = nn.ReLU()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, cf, vertices):
        # vertices: (bs, num_seg, seg_len, Nverts, 3)
        init_cf = cf[:, 0]
        init_vertices = vertices[:, 0]
        if self.encoder_mode == 2:
            init_feat = torch.ones(cf.shape[0], self.prev_frame_feat_dim, device=cf.device)  # (bs, seg_len, Nverts, f_dim)
        else:
            init_feat = torch.ones(cf.shape[0], cf.shape[2], cf.shape[3], self.prev_frame_feat_dim, device=cf.device)   # (bs, seg_len, Nverts, f_dim)
        init_feat *= 0.1

        init_mu, init_logvar = self.encoder(init_cf, init_vertices, init_feat)  # mu, logvar = (bs, 256)
        init_z = self.reparameterize(init_mu, init_logvar)

        init_out = self.decoder(init_z, init_vertices, init_feat)

        out = init_out.unsqueeze(1)
        mu = init_mu.unsqueeze(1)
        logvar = init_logvar.unsqueeze(1)
        if self.encoder_mode == 2:
            prev_out = torch.max(init_out, dim=2)[0].reshape(cf.shape[0], -1)
        else:
            prev_out = init_out

        for i in range(1, cf.shape[1]):
            prev_out_feat = self.relu(self.out_linear(prev_out))

            cur_cf = cf[:, i]
            cur_vertices = vertices[:, i]
            cur_mu, cur_logvar = self.encoder(cur_cf, cur_vertices, prev_out_feat)
            cur_z = self.reparameterize(cur_mu, cur_logvar)

            cur_out = self.decoder(cur_z, cur_vertices, prev_out_feat)
            if self.encoder_mode == 2:
                prev_out = torch.max(cur_out, dim=2)[0].reshape(cf.shape[0], -1)
            else:
                prev_out = cur_out

            out = torch.cat((out, cur_out.unsqueeze(1)), dim=1)
            mu = torch.cat((mu, cur_mu.unsqueeze(1)), dim=1)
            logvar = torch.cat((logvar, cur_logvar.unsqueeze(1)), dim=1)

        return out, mu, logvar


class Encoder_seq(nn.Module):
    def __init__(self, seg_len, prev_feat_dim=64, h_dim=512, z_dim=256, channels=64, ds_us_dir='../data/mesh_ds',
                 normalization_mode='group_norm', num_groups=8, seq_length=9, no_obj_classes=8, use_cuda=True,
                 encoder_mode=0, n_layer=6, n_head=8, add_virtual_node=False, **kwargs):
        super(Encoder_seq, self).__init__()

        # Here seg_len means the number of frames concatenated as the input of the temporal POSA,
        # seq_length is the number of neighbor nodes considered in GCN.
        # num_groups for GroupNorm layer
        self.encoder_mode = encoder_mode
        self.seg_len = seg_len
        self.prev_feat_dim = prev_feat_dim
        self.f_dim = no_obj_classes
        self.spiral_indices = []
        self.nv = []
        self.D = []
        self.U = []
        levels = [0, 1, 2]
        for level in levels:
            nv, spiral_indices, _, U, D, _ = load_ds_us_param(ds_us_dir, level, seq_length, use_cuda)
            self.nv.append(nv)  # nv: [655, 164, 41]
            self.D.append(D)    # D: [(655, 2619),(164, 655),(41, 164)]
            self.U.append(U)
            self.spiral_indices.append(spiral_indices) # spiral_indices: (655->164->41, 9)
        self.channels = (channels * np.ones(4)).astype(np.int).tolist() # [64, 64, 64, 64]

        self.en_spiral = nn.ModuleList()
        if encoder_mode == 2:
            self.en_spiral.append(Spiral_block(3 + self.f_dim, self.channels[0],
                                               self.spiral_indices[0], normalization_mode, num_groups,
                                               add_virtual_node=add_virtual_node))
        else:
            self.en_spiral.append(Spiral_block(3 + self.f_dim + prev_feat_dim, self.channels[0],
                                               self.spiral_indices[0], normalization_mode, num_groups,
                                               add_virtual_node=add_virtual_node))
        for i in levels:
            self.en_spiral.append(
                Spiral_block(self.channels[i], self.channels[i + 1], self.spiral_indices[i], normalization_mode,
                             num_groups, add_virtual_node=add_virtual_node))
            if i != len(levels) - 1:
                self.en_spiral.append(ds_us_fn(self.D[i + 1]))

        self.en_spiral = nn.Sequential(*self.en_spiral)
        self.en_mu = nn.Linear(h_dim, z_dim)
        self.en_log_var = nn.Linear(h_dim, z_dim)
        if encoder_mode == 0:
            self.en_transformer = transformer.TransformerEncoder(n_layers=n_layer, n_head=n_head, d_in=channels, d_k=channels,
                                                                d_v=channels, d_z=h_dim, seg_len=seg_len,
                                                                n_verts=self.nv[-1], upsampling_matrices=self.U)
        elif encoder_mode == 1:
            self.en_transformer = transformer.TransformerEncoder_1(n_layers=n_layer, n_head=n_head, d_in=channels,
                                                                 d_k=channels, d_v=channels, d_z=z_dim, seg_len=seg_len)
        elif encoder_mode == 2:
            self.en_transformer = transformer.TransformerEncoder_2(n_layers=n_layer, n_head=n_head, d_in=channels,
                                                                   d_k=channels, d_v=channels, d_z=z_dim, d_prev = prev_feat_dim,
                                                                   seg_len=seg_len)

    def forward(self, x, vertices, prev_seg_feat):
        if self.encoder_mode == 0:
            x = torch.cat((vertices, x, prev_seg_feat), dim=-1)    # x: (bs, seg_len, Nv, 3 + 8 + feat_dim)
            x = self.en_spiral(x)   # x: (bs, seg_len, Nv, 64)
            z = self.en_transformer(x)
            return self.en_mu(z), self.en_log_var(z)
        elif self.encoder_mode == 1:
            x = torch.cat((vertices, x, prev_seg_feat), dim=-1)  # x: (bs, seg_len, Nv, 3 + 8 + feat_dim)
            x = self.en_spiral(x)  # x: (bs, seg_len, Nv, 64)
            mu, logvar = self.en_transformer(x)
            return mu, logvar
        elif self.encoder_mode == 2:
            # prev_seg_feat: (bs, d_feat)
            x = torch.cat((vertices, x), dim=-1)
            x = self.en_spiral(x)
            mu, logvar = self.en_transformer(x, prev_seg_feat)
            return mu, logvar


class Decoder_seq(nn.Module):
    def __init__(self, seg_len, z_dim=256, prev_feat_dim=64, num_hidden_layers=3, channels=64, ds_us_dir='../data/mesh_ds',
                 normalization_mode='group_norm', num_groups=8, seq_length=9, no_obj_classes=8, use_cuda=True,
                 decoder_mode=0, n_layer=6, n_head=8, add_virtual_node=False, **kwargs):
        super(Decoder_seq, self).__init__()
        self.decoder_mode = decoder_mode
        self.num_hidden_layers = num_hidden_layers
        self.f_dim = no_obj_classes
        self.spiral_indices = []
        self.nv = []
        levels = [0, 1, 2]
        for level in levels:
            nv, spiral_indices, _, _, _, _ = load_ds_us_param(ds_us_dir, level, seq_length, use_cuda)
            self.nv.append(nv)
            self.spiral_indices.append(spiral_indices)
        self.channels = (channels * np.ones(4)).astype(np.int).tolist()

        self.de_spiral = nn.ModuleList()
        if decoder_mode == 0:
            self.de_spiral.append(GraphLin_block(3 + z_dim + prev_feat_dim, z_dim // 2, normalization_mode, num_groups))
            self.de_spiral.append(GraphLin_block(z_dim // 2, self.channels[0], normalization_mode, num_groups))
        elif decoder_mode == 1:
            self.de_spiral.append(GraphLin_block(3 + prev_feat_dim, self.channels[0], normalization_mode, num_groups))
        elif decoder_mode == 2:
            self.de_spiral.append(GraphLin_block(3 + z_dim, z_dim // 2, normalization_mode, num_groups))
            self.de_spiral.append(GraphLin_block(z_dim // 2, self.channels[0], normalization_mode, num_groups))
        for _ in range(self.num_hidden_layers):
            self.de_spiral.append(
                Spiral_block(self.channels[0], self.channels[0], self.spiral_indices[0], normalization_mode,
                             num_groups, add_virtual_node=add_virtual_node))
        # self.de_spiral.append(SpiralConv(self.channels[0], self.channels[0], self.spiral_indices[0],
        #                                  add_virtual_node=add_virtual_node))
        self.de_spiral = nn.Sequential(*self.de_spiral)
        if decoder_mode == 0:
            self.de_transformer = transformer.TransformerDecoder(n_layers=n_layer, n_head=n_head, d_in=channels, d_out=self.f_dim,
                                                                d_k=channels, d_v=channels, seg_len=seg_len)
        elif decoder_mode == 1:
            self.de_transformer = transformer.TransformerDecoder_1(n_layers=n_layer, n_head=n_head, d_in=channels, d_out=self.f_dim,
                                                                   d_k=channels, d_v=channels, seg_len=seg_len)
        elif decoder_mode == 2:
            self.de_transformer = transformer.TransformerDecoder_2(n_layers=n_layer, n_head=n_head, d_in=channels,
                                                                 d_out=self.f_dim, d_k=channels, d_v=channels,
                                                                 d_prev=prev_feat_dim, seg_len=seg_len)

    def forward(self, z, vertices, prev_out_feat):
        if self.decoder_mode == 0:
            z = z.unsqueeze(1).unsqueeze(1).expand((-1, vertices.shape[1], self.nv[0], -1))
            x = torch.cat((vertices, prev_out_feat, z), dim=-1)
            x = self.de_spiral(x)
            x = self.de_transformer(x)
        elif self.decoder_mode == 1:
            x = torch.cat((vertices, prev_out_feat), dim=-1)
            x = self.de_spiral(x)
            z = z.view(-1, 4, 64)   # I'm hard coding this, z: (bs, n_head, d_q)
            z = z.unsqueeze(2).unsqueeze(2).expand(-1, -1, x.shape[2], x.shape[1], -1)
            z = z.reshape(-1, x.shape[1], 64)  # (bs * n_head * n_verts, seg_len, d_q)
            x = self.de_transformer(x, z)
        elif self.decoder_mode == 2:
            z = z.unsqueeze(1).unsqueeze(1).expand((-1, vertices.shape[1], self.nv[0], -1))
            x = torch.cat((vertices, z), dim=-1)
            x = self.de_spiral(x)
            x = self.de_transformer(x, prev_out_feat)
        return x

'''
This model is similar to POSA_temp_transformer_seq. However, it accepts input motion segments
with variant lengths. For different purposes, there are many reasonable ways split a motion sequence into several
segments with different lengths. E.g. Split segments based on human's positions.
'''
class POSA_temp_transformer_var(nn.Module):
    def __init__(self, max_frame, encoder_mode, decoder_mode, n_layer=6, n_head=8, add_virtual_node=False,
                 prev_feat_dim=64, no_obj_classes=8, **kwargs):
        super(POSA_temp_transformer_var, self).__init__()
        self.max_frame = max_frame
        self.encoder = Encoder_var(self.max_frame, prev_feat_dim=prev_feat_dim, encoder_mode=encoder_mode,  n_layer=n_layer, n_head=n_head,
                               add_virtual_node=add_virtual_node, no_obj_classes=no_obj_classes, **kwargs)
        self.decoder = Decoder_var(self.max_frame, prev_feat_dim=prev_feat_dim, decoder_mode=decoder_mode,  n_layer=n_layer, n_head=n_head,
                               add_virtual_node=add_virtual_node, no_obj_classes=no_obj_classes, **kwargs)
        self.prev_feat_dim = prev_feat_dim
        self.out_linear = nn.Linear(no_obj_classes, prev_feat_dim)
        self.relu = nn.ReLU()


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, cf, vertices, mask):
        # cf: (bs, num_seg, seg_len, n_verts, 8), mask: (bs, num_seg, seg_len)
        init_cf = cf[:, 0]
        init_vertices = vertices[:, 0]
        init_feat = torch.ones(cf.shape[0], cf.shape[2], cf.shape[3], self.prev_feat_dim,
                               device=cf.device)  # (bs, seg_len, n_verts, f_dim)
        init_feat *= 0.1

        # Prepend mask with 2 extra tokens for latent Gaussian space, and expand mask to the correct dimension.
        init_mask = mask[:, 0]  # (bs, seg_len)
        init_encoder_mask = torch.cat((torch.ones(init_mask.shape[0], 2, device=init_mask.device), init_mask), dim=1)  # (bs, seg_len + 2)
        init_encoder_mask = init_encoder_mask.unsqueeze(1).expand(-1, init_encoder_mask.shape[1], -1)  # (bs, seg_len + 2, seg_len + 2)
        init_decoder_mask = init_mask.unsqueeze(1).expand(-1, init_mask.shape[1], -1)  # (bs, seg_len, seg_len)

        init_mu, init_logvar = self.encoder(init_cf, init_vertices, init_feat, init_encoder_mask)  # mu, logvar = (bs, 256)
        init_z = self.reparameterize(init_mu, init_logvar)

        init_out = self.decoder(init_z, init_vertices, init_feat, init_decoder_mask)
        prev_out = init_out

        out = init_out.unsqueeze(1)
        mu = init_mu.unsqueeze(1)
        logvar = init_logvar.unsqueeze(1)

        for i in range(1, cf.shape[1]):
            prev_seg_feat = self.relu(self.out_linear(prev_out))

            cur_cf = cf[:, i]
            cur_vertices = vertices[:, i]
            cur_mask = mask[:, i]
            cur_encoder_mask = torch.cat((torch.ones(cur_mask.shape[0], 2, device=cur_mask.device), cur_mask), dim=1)  # (bs, seg_len + 2)
            cur_encoder_mask = cur_encoder_mask.unsqueeze(1).expand(-1, cur_encoder_mask.shape[1], -1)  # (bs, seg_len + 2, seg_len + 2)
            cur_decoder_mask = cur_mask.unsqueeze(1).expand(-1, cur_mask.shape[1], -1)  # (bs, seg_len, seg_len)
            cur_mu, cur_logvar = self.encoder(cur_cf, cur_vertices, prev_seg_feat, cur_encoder_mask)
            cur_z = self.reparameterize(cur_mu, cur_logvar)

            cur_out = self.decoder(cur_z, cur_vertices, prev_seg_feat, cur_decoder_mask)
            prev_out = cur_out

            out = torch.cat((out, cur_out.unsqueeze(1)), dim=1)
            mu = torch.cat((mu, cur_mu.unsqueeze(1)), dim=1)
            logvar = torch.cat((logvar, cur_logvar.unsqueeze(1)), dim=1)

        return out, mu, logvar


class Encoder_var(nn.Module):
    def __init__(self, max_frame, prev_feat_dim=64, h_dim=512, z_dim=256, channels=64, ds_us_dir='../data/mesh_ds',
                 normalization_mode='group_norm', num_groups=8, seq_length=9, no_obj_classes=8, use_cuda=True,
                 encoder_mode=0, n_layer=6, n_head=8, add_virtual_node=False, **kwargs):
        super(Encoder_var, self).__init__()

        # Here seg_len means the number of frames concatenated as the input of the temporal POSA,
        # seq_length is the number of neighbor nodes considered in GCN.
        # num_groups for GroupNorm layer
        self.encoder_mode = encoder_mode
        self.max_frame = max_frame
        self.f_dim = no_obj_classes
        self.spiral_indices = []
        self.nv = []
        self.D = []
        self.U = []
        levels = [0, 1, 2]
        for level in levels:
            nv, spiral_indices, _, U, D, _ = load_ds_us_param(ds_us_dir, level, seq_length, use_cuda)
            self.nv.append(nv)  # nv: [655, 164, 41]
            self.D.append(D)    # D: [(655, 2619),(164, 655),(41, 164)]
            self.U.append(U)
            self.spiral_indices.append(spiral_indices) # spiral_indices: (655->164->41, 9)
        self.channels = (channels * np.ones(4)).astype(np.int).tolist() # [64, 64, 64, 64]

        self.en_spiral = nn.ModuleList()
        self.en_spiral.append(
            Spiral_block(3 + self.f_dim + prev_feat_dim, self.channels[0], self.spiral_indices[0], normalization_mode, num_groups,
                         add_virtual_node=add_virtual_node))
        for i in levels:
            self.en_spiral.append(
                Spiral_block(self.channels[i], self.channels[i + 1], self.spiral_indices[i], normalization_mode,
                             num_groups, add_virtual_node=add_virtual_node))
            if i != len(levels) - 1:
                self.en_spiral.append(ds_us_fn(self.D[i + 1]))

        self.en_spiral = nn.Sequential(*self.en_spiral)
        self.en_mu = nn.Linear(h_dim, z_dim)
        self.en_log_var = nn.Linear(h_dim, z_dim)

        assert encoder_mode == 1, "encoder_mode in Encoder_var must be 1"
        self.en_transformer = transformer.TransformerEncoder_1(n_layers=n_layer, n_head=n_head, d_in=channels,
                                                               d_k=channels, d_v=channels, d_z=z_dim, seg_len=max_frame)


    def forward(self, x, vertices, prev_seg_feat, mask):
        x = torch.cat((vertices, x, prev_seg_feat), dim=-1)    # x: (bs, seg_len, Nv, 3+8+prev_feat_dim)
        x = self.en_spiral(x)   # x: (bs, seg_len, Nv, 64)
        mu, logvar = self.en_transformer(x, mask)
        return mu, logvar


class Decoder_var(nn.Module):
    def __init__(self, max_frame, z_dim=256, prev_feat_dim=64, num_hidden_layers=2, channels=64, ds_us_dir='../data/mesh_ds',
                 normalization_mode='group_norm', num_groups=8, seq_length=9, no_obj_classes=8, use_cuda=True,
                 decoder_mode=0, n_layer=6, n_head=8, add_virtual_node=False, **kwargs):
        super(Decoder_var, self).__init__()
        self.decoder_mode = decoder_mode
        self.num_hidden_layers = num_hidden_layers
        self.f_dim = no_obj_classes
        self.spiral_indices = []
        self.nv = []
        levels = [0, 1, 2]
        for level in levels:
            nv, spiral_indices, _, _, _, _ = load_ds_us_param(ds_us_dir, level, seq_length, use_cuda)
            self.nv.append(nv)
            self.spiral_indices.append(spiral_indices)
        self.channels = (channels * np.ones(4)).astype(np.int).tolist()

        self.de_spiral = nn.ModuleList()
        self.de_spiral.append(GraphLin_block(3 + z_dim + prev_feat_dim, z_dim // 2, normalization_mode, num_groups))
        self.de_spiral.append(GraphLin_block(z_dim // 2, self.channels[0], normalization_mode, num_groups))
        for _ in range(self.num_hidden_layers):
            self.de_spiral.append(
                Spiral_block(self.channels[0], self.channels[0], self.spiral_indices[0], normalization_mode,
                             num_groups, add_virtual_node=add_virtual_node))
        # self.de_spiral.append(SpiralConv(self.channels[0], self.channels[0], self.spiral_indices[0],
        #                                  add_virtual_node=add_virtual_node))
        self.de_spiral = nn.Sequential(*self.de_spiral)

        self.de_transformer = transformer.TransformerDecoder(n_layers=n_layer, n_head=n_head, d_in=channels, d_out=self.f_dim,
                                                             d_k=channels, d_v=channels, seg_len=max_frame)

    def forward(self, x, vertices, prev_seg_feat, mask):
        x = x.unsqueeze(1)
        x = x.unsqueeze(1).expand((-1, vertices.shape[1], self.nv[0], -1))
        x = torch.cat((vertices, x, prev_seg_feat), dim=-1)
        x = self.de_spiral(x)   # (bs, seg_len, n_verts, 64)

        x = self.de_transformer(x, mask)
        return x

'''
This model is similar to POSA_temp_transformer_seq. In addition to it, the model
considers predictions from both previous segments and next segments.
'''
class POSA_temp_transformer_bidir(nn.Module):
    def __init__(self, seg_len, encoder_mode, decoder_mode, n_layer=6, n_head=8, f_vert=64, dim_ff=256,
                 add_virtual_node=False, h_dim=512, **kwargs):
        super(POSA_temp_transformer_bidir, self).__init__()
        self.seg_len = seg_len
        self.encoder_mode = encoder_mode
        self.decoder_mode = decoder_mode
        self.encoder = Encoder_bidir(self.seg_len, encoder_mode=encoder_mode, n_layer=n_layer, n_head=n_head,
                                     channels=f_vert, dim_ff=dim_ff, add_virtual_node=add_virtual_node, h_dim=h_dim, **kwargs)
        self.decoder = Decoder_bidir(self.seg_len, decoder_mode=decoder_mode, n_layer=n_layer, n_head=n_head,
                                     channels=f_vert, dim_ff=dim_ff, add_virtual_node=add_virtual_node, h_dim=h_dim, **kwargs)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, cf, vertices, mask):
        # vertices: (bs, seg_len, Nverts, 3), mask: (bs, seg_len)
        # Prepend mask with 2 extra tokens for latent Gaussian space, and expand mask to the correct dimension.
        encoder_mask = torch.cat((torch.ones(mask.shape[0], 2, device=mask.device), mask), dim=1)  # (bs, seg_len + 2)
        if self.encoder_mode == 1:
            encoder_mask = encoder_mask.unsqueeze(1).expand(-1, encoder_mask.shape[1], -1)  # (bs, seg_len + 2, seg_len + 2)
            decoder_mask = mask.unsqueeze(1).expand(-1, mask.shape[1], -1)  # (bs, seg_len, seg_len)
        elif self.encoder_mode == 0 or self.encoder_mode == 2:
            encoder_mask = (1 - encoder_mask) > 0
            decoder_mask = (1 - mask) > 0
        mu, logvar = self.encoder(cf, vertices, encoder_mask)
        z = self.reparameterize(mu, logvar)
        out = self.decoder(z, vertices, decoder_mask)
        return out, mu, logvar


class Encoder_bidir(nn.Module):
    def __init__(self, seg_len, h_dim=512, z_dim=256, channels=64, ds_us_dir='../data/mesh_ds',
                 normalization_mode='group_norm', num_groups=8, seq_length=9, no_obj_classes=8, use_cuda=True,
                 encoder_mode=1, n_layer=3, n_head=4, dim_ff=256, add_virtual_node=False, **kwargs):
        super(Encoder_bidir, self).__init__()

        # Here seg_len means the number of frames concatenated as the input of the temporal POSA,
        # seq_length is the number of neighbor nodes considered in GCN.
        # num_groups for GroupNorm layer
        self.encoder_mode = encoder_mode
        self.seg_len = seg_len
        self.f_dim = no_obj_classes
        self.spiral_indices = []
        self.nv = []
        self.D = []
        self.U = []
        levels = [0, 1, 2]
        for level in levels:
            nv, spiral_indices, _, U, D, _ = load_ds_us_param(ds_us_dir, level, seq_length, use_cuda)
            self.nv.append(nv)  # nv: [655, 164, 41]
            self.D.append(D)    # D: [(655, 2619),(164, 655),(41, 164)]
            self.U.append(U)
            self.spiral_indices.append(spiral_indices) # spiral_indices: (655->164->41, 9)
        self.channels = (channels * np.ones(4)).astype(np.int).tolist() # [64, 64, 64, 64]

        self.en_spiral = nn.ModuleList()
        self.en_spiral.append(
            Spiral_block(3 + self.f_dim, self.channels[0], self.spiral_indices[0], normalization_mode, num_groups,
                         add_virtual_node=add_virtual_node))
        for i in levels:
            self.en_spiral.append(
                Spiral_block(self.channels[i], self.channels[i + 1], self.spiral_indices[i], normalization_mode,
                             num_groups, add_virtual_node=add_virtual_node))
            if i != len(levels) - 1:
                self.en_spiral.append(ds_us_fn(self.D[i + 1]))

        self.en_spiral = nn.Sequential(*self.en_spiral)
        if encoder_mode == 1:
            self.en_transformer = transformer.TransformerEncoder_1(n_layers=n_layer, n_head=n_head, d_in=channels,
                                                                d_k=channels, d_v=channels, d_z=z_dim, seg_len=seg_len)
        elif encoder_mode == 0:
            self.pos_codebook = nn.Embedding.from_pretrained(transformer.get_sinusoid_pos_encoding(seg_len + 2, channels),
                                                             freeze=True)   # (seg_len + 2, 64)
            self.latent_tokens = nn.Parameter(torch.randn(2, channels))
            self.en_transformerLayer = nn.TransformerEncoderLayer(d_model=channels, nhead=n_head,
                                                                  dim_feedforward=dim_ff)
            self.en_transformer = nn.TransformerEncoder(self.en_transformerLayer, num_layers=n_layer)
            self.en_vertsEmb = nn.Linear(self.nv[-1] * self.channels[-1], h_dim)
            self.en_mu = nn.Linear(h_dim + channels, z_dim)
            self.en_logvar = nn.Linear(h_dim + channels, z_dim)
            self.relu = nn.ReLU()
        elif encoder_mode == 2:
            self.pos_codebook = nn.Embedding.from_pretrained(transformer.get_sinusoid_pos_encoding(seg_len + 2, h_dim),
                                                             freeze=True)   # (seg_len + 2, h_dim)
            self.latent_tokens = nn.Parameter(torch.randn(2, h_dim))
            self.en_transformerLayer = nn.TransformerEncoderLayer(d_model=h_dim, nhead=n_head,
                                                                  dim_feedforward=dim_ff)
            self.en_transformer = nn.TransformerEncoder(self.en_transformerLayer, num_layers=n_layer)
            self.en_vertsEmb = nn.Linear(self.nv[-1] * self.channels[-1], h_dim)
            self.en_mu = nn.Linear(h_dim + h_dim, z_dim)
            self.en_logvar = nn.Linear(h_dim + h_dim, z_dim)
            self.relu = nn.ReLU()

    def forward(self, x, vertices, mask):
        x = torch.cat((vertices, x), dim=-1)    # x: (bs, seg_len, Nv, 3 + 8)
        x = self.en_spiral(x)   # x: (bs, seg_len, Nv, 64)
        bs, _, nv, f_vert = x.shape
        if self.encoder_mode == 1:
            mu, logvar = self.en_transformer(x, mask)
            return mu, logvar
        elif self.encoder_mode == 0:
            verts_feat = x
            x = torch.max(x, dim=2)[0]  # x: (bs, seg_len, 64)
            latent_tokens = self.latent_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)
            x = torch.cat((latent_tokens, x), dim=1)   # (bs, seg_len + 2, 64)
            pos_vec = torch.arange(self.seg_len + 2).unsqueeze(0).repeat(x.shape[0], 1).to(x.device)  # (bs, seg_len + 2)
            pos_emb = self.pos_codebook(pos_vec)  # (bs, seg_len + 2, 64)
            x = x + pos_emb
            x = x.permute(1, 0, 2)  # (seg_len + 2, bs, f_vert)
            x = self.en_transformer(x, src_key_padding_mask=mask)   # (seg_len + 2, bs, f_vert)

            mu = x[0]   # (bs, f_vert)
            mu = mu.unsqueeze(1).expand(-1, self.seg_len, -1)   # (bs, seg_len, f_vert)
            logvar = x[1]   # (bs, f_vert)
            logvar = logvar.unsqueeze(1).expand(-1, self.seg_len, -1)
            verts_feat = verts_feat.reshape(bs, self.seg_len, -1)   # (bs, seg_len, nv * f_vert)
            verts_feat = self.relu(self.en_vertsEmb(verts_feat))   # (bs, seg_len, h_dim)
            mu = self.en_mu(torch.cat((verts_feat, mu), dim=-1))  # (bs, seg_len, z_dim)
            logvar = self.en_logvar(torch.cat((verts_feat, logvar), dim=-1))  # (bs, seg_len, z_dim)
            return mu, logvar
        elif self.encoder_mode == 2:
            x = x.reshape(bs, self.seg_len, -1) # (bs, seg_len, nv * f_vert)
            x = self.relu(self.en_vertsEmb(x))  # (bs, seg_len, h_dim)
            frame_feat = x

            latent_tokens = self.latent_tokens.unsqueeze(0).expand(x.shape[0], -1, -1)  # (bs, 2, h_dim)
            x = torch.cat((latent_tokens, x), dim=1)  # (bs, seg_len + 2, h_dim)
            pos_vec = torch.arange(self.seg_len + 2).unsqueeze(0).repeat(x.shape[0], 1).to(x.device)  # (bs, seg_len + 2)
            pos_emb = self.pos_codebook(pos_vec)  # (bs, seg_len + 2, h_dim)
            x = x + pos_emb
            x = x.permute(1, 0, 2)  # (seg_len + 2, bs, h_dim)
            x = self.en_transformer(x, src_key_padding_mask=mask)  # (seg_len + 2, bs, h_dim)

            mu = x[0]  # (bs, h_dim)
            mu = mu.unsqueeze(1).expand(-1, self.seg_len, -1)  # (bs, seg_len, h_dim)
            logvar = x[1]
            logvar = logvar.unsqueeze(1).expand(-1, self.seg_len, -1)
            mu = self.en_mu(torch.cat((frame_feat, mu), dim=-1))  # (bs, seg_len, z_dim)
            logvar = self.en_logvar(torch.cat((frame_feat, logvar), dim=-1))  # (bs, seg_len, z_dim)
            return mu, logvar


class Decoder_bidir(nn.Module):
    def __init__(self, seg_len, z_dim=256, h_dim=512, num_hidden_layers=3, channels=64, ds_us_dir='../data/mesh_ds',
                 normalization_mode='group_norm', num_groups=8, seq_length=9, no_obj_classes=8, use_cuda=True,
                 decoder_mode=0, n_layer=6, n_head=8, dim_ff=256, add_virtual_node=False, **kwargs):
        super(Decoder_bidir, self).__init__()
        self.decoder_mode = decoder_mode
        self.seg_len = seg_len
        self.num_hidden_layers = num_hidden_layers
        self.f_dim = no_obj_classes
        self.spiral_indices = []
        self.nv = []
        levels = [0, 1, 2]
        for level in levels:
            nv, spiral_indices, _, _, _, _ = load_ds_us_param(ds_us_dir, level, seq_length, use_cuda)
            self.nv.append(nv)
            self.spiral_indices.append(spiral_indices)
        self.channels = (channels * np.ones(4)).astype(np.int).tolist()

        self.de_spiral = nn.ModuleList()
        self.de_spiral.append(GraphLin_block(3 + z_dim, z_dim // 2, normalization_mode, num_groups))
        self.de_spiral.append(GraphLin_block(z_dim // 2, self.channels[0], normalization_mode, num_groups))
        if decoder_mode == 0 or decoder_mode == 2:
            self.num_hidden_layers += 1
        for _ in range(self.num_hidden_layers):
            self.de_spiral.append(
                Spiral_block(self.channels[0], self.channels[0], self.spiral_indices[0], normalization_mode,
                             num_groups, add_virtual_node=add_virtual_node))
        # self.de_spiral.append(SpiralConv(self.channels[0], self.channels[0], self.spiral_indices[0],
        #                                  add_virtual_node=add_virtual_node))
        self.de_spiral = nn.Sequential(*self.de_spiral)

        if decoder_mode == 1:
            self.de_transformer = transformer.TransformerDecoder(n_layers=n_layer, n_head=n_head, d_in=channels, d_out=self.f_dim,
                                                             d_k=channels, d_v=channels, seg_len=seg_len)
        elif decoder_mode == 0:
            self.pos_codebook = nn.Embedding.from_pretrained(
                transformer.get_sinusoid_pos_encoding(seg_len, channels),
                freeze=True)
            self.de_transformerLayer = nn.TransformerDecoderLayer(d_model=channels, nhead=n_head,
                                                                  dim_feedforward=dim_ff)
            self.de_transformer = nn.TransformerDecoder(decoder_layer=self.de_transformerLayer, num_layers=n_layer)
            self.z_to_f_vert = nn.Linear(z_dim, channels)
            self.fin_linear = nn.Linear(channels * 2, self.f_dim)
        elif decoder_mode == 2:
            self.pos_codebook = nn.Embedding.from_pretrained(
                transformer.get_sinusoid_pos_encoding(seg_len, h_dim),
                freeze=True)
            self.de_transformerLayer = nn.TransformerEncoderLayer(d_model=h_dim, nhead=n_head,
                                                                  dim_feedforward=dim_ff)
            self.de_transformer = nn.TransformerEncoder(self.de_transformerLayer, num_layers=n_layer)
            self.de_vertsEmb = nn.Linear(self.nv[0] * self.channels[-1], h_dim)
            self.relu = nn.ReLU()
            self.fin_linear = nn.ModuleList()
            self.fin_linear.append(nn.Linear(h_dim + channels, channels))
            self.fin_linear.append(nn.ReLU())
            self.fin_linear.append(nn.Linear(channels, self.f_dim))
            self.fin_linear = nn.Sequential(*self.fin_linear)

    def forward(self, x, vertices, mask):
        bs, _, nv, _ = vertices.shape
        if self.decoder_mode == 1:
            x = x.unsqueeze(1).unsqueeze(1).expand((-1, vertices.shape[1], self.nv[0], -1))
            x = torch.cat((vertices, x), dim=-1)
            x = self.de_spiral(x)
            x = self.de_transformer(x, mask)
            return x
        elif self.decoder_mode == 0:
            z = x   # (bs, seg_len, z_dim)
            x = x.unsqueeze(2).expand((-1, -1, self.nv[0], -1))
            x = torch.cat((vertices, x), dim=-1)
            x = self.de_spiral(x)   # (bs, seg_len, nv, f_vert)
            verts_feat = x

            x = torch.max(x, dim=2)[0]  # (bs, seg_len, f_vert)
            pos_vec = torch.arange(self.seg_len).unsqueeze(0).repeat(x.shape[0], 1).to(x.device)  # (bs, seg_len)
            pos_emb = self.pos_codebook(pos_vec)  # (bs, seg_len, f_vert)
            x = x + pos_emb
            x = x.permute(1, 0, 2)  # (seg_len, bs, f_vert)
            z = self.z_to_f_vert(z).permute(1, 0, 2)    # (seg_len, bs, f_vert)
            x = self.de_transformer(x, memory=z, tgt_key_padding_mask=mask) # (seg_len, bs, f_vert)
            x = x.permute(1, 0, 2)  # (bs, seg_len, f_vert)
            x = x.unsqueeze(2).expand(-1, -1, nv, -1)   #(bs, seg_len, nv, f_vert)
            x = self.fin_linear(torch.cat((verts_feat, x), dim=-1))
            return x
        elif self.decoder_mode == 2:
            x = x.unsqueeze(2).expand((-1, -1, self.nv[0], -1))
            x = torch.cat((vertices, x), dim=-1)
            x = self.de_spiral(x)  # (bs, seg_len, nv, f_vert)

            verts_feat = x
            x = x.reshape(bs, self.seg_len, -1) # (bs, seg_len, nv * f_vert)
            x = self.relu(self.de_vertsEmb(x))  # (bs, seg_len, h_dim)

            pos_vec = torch.arange(self.seg_len).unsqueeze(0).repeat(x.shape[0], 1).to(x.device)  # (bs, seg_len)
            pos_emb = self.pos_codebook(pos_vec)  # (bs, seg_len, h_dim)
            x = x + pos_emb
            x = x.permute(1, 0, 2)  # (seg_len, bs, h_dim)
            x = self.de_transformer(x, src_key_padding_mask=mask)  # (seg_len, bs, h_dim)

            x = x.permute(1, 0, 2)  # (bs, seg_len, h_dim)
            x = x.unsqueeze(2).expand(-1, -1, nv, -1)  # (bs, seg_len, nv, h_dim)
            x = self.fin_linear(torch.cat((verts_feat, x), dim=-1))
            return x