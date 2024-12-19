import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import copy
import numpy as np
import math
import tqdm
import pickle
from eval import read_file
from torch.nn.functional import scaled_dot_product_attention
from eval import evaluate

from eval import segment_bars_with_confidence
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, MeanShift, AffinityPropagation, SpectralClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

def class_aware_balancing_loss(target, pred, gamma=0.05, num_classes=11):
    log_probs = F.log_softmax(pred, dim=1)
    probs = torch.exp(log_probs)
    

    target_one_hot = torch.zeros_like(pred).scatter(1, target.unsqueeze(1), 1)
    pt = (probs * target_one_hot).sum(dim=1)
    total_length = target.size(0)
    alpha = []

    for i in range(num_classes):
        class_count = (target == i).sum().item()
        alpha_i = total_length / (total_length + class_count * num_classes)
        alpha.append(alpha_i)

    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=pred.device).gather(0, target)    
    epsilon = 1e-8
    balancing_loss = -alpha_t * (1 - pt + epsilon) ** gamma * log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    return balancing_loss.mean()         

class ProbabilityProgressFusionModel(nn.Module):
    def __init__(self, num_classes):
        super(ProbabilityProgressFusionModel, self).__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv1d(num_classes*2, num_classes, 1)

    def forward(self, in_cls, in_prg):
        ### in_cls: batch_size x num_classes x T
        ### in_prg: batch_size x num_classes x T
        # Concatenate classification and progress inputs
        input_concat = torch.cat((in_cls, in_prg), dim=1)
        out = self.conv(input_concat)
        return out
    
class TaskGraphLearner(nn.Module):
    def __init__(self, init_graph_path, learnable=False, reg_weight=0.01, eta=0.01):
        super(TaskGraphLearner, self).__init__()
        with open(init_graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        matrix_pre, matrix_suc = self.graph['matrix_pre'], self.graph['matrix_suc']
        self.matrix_pre = nn.Parameter(torch.from_numpy(matrix_pre).float(), requires_grad=learnable)
        self.matrix_suc = nn.Parameter(torch.from_numpy(matrix_suc).float(), requires_grad=learnable)
        self.learnable = learnable
        if learnable:
            self.matrix_pre_original = nn.Parameter(self.matrix_pre, requires_grad=False)
            self.matrix_suc_original = nn.Parameter(self.matrix_suc, requires_grad=False)
        self.reg_weight = reg_weight
        self.eta = eta

    def forward(self, cls, prg):
        action_prob = F.softmax(cls, dim=1)
        prg = torch.clamp(prg, min=0, max=1)
        completion_status, _ = torch.cummax(prg, dim=-1)
        alpha_pre = torch.einsum('bkt,kK->bKt', 1 - completion_status, self.matrix_pre)
        alpha_suc = torch.einsum('bkt,kK->bKt', completion_status, self.matrix_suc)
        graph_loss = ((alpha_pre + alpha_suc) * action_prob).mean()
        if self.learnable:
            regularization = torch.mean((self.matrix_pre - self.matrix_pre_original) ** 2)
            return graph_loss + self.reg_weight * regularization
        return graph_loss

    def inference(self, cls, prg):
        action_prob = F.softmax(cls, dim=1)
        prg = torch.clamp(prg, min=0, max=1)
        completion_status, _ = torch.cummax(prg, dim=-1)
        alpha_pre = torch.einsum('bkt,kK->bKt', 1 - completion_status, self.matrix_pre)
        alpha_suc = torch.einsum('bkt,kK->bKt', completion_status, self.matrix_suc)
        logits = cls - self.eta * (alpha_pre + alpha_suc)
        return logits
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask, short = False, short_window_scale = 0.2):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(
            d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        if short:
            seq_len = attn_mask.size(-1)
            for i in range(seq_len):
                attn_mask[:, i, :int(i * (1-short_window_scale))] = True
                
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is True.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn
    
class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)


    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask,casual=False):
        '''
        scalar dot attention.
        :param proj_query: shape of (B, C, L) => (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        '''
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape
        
        assert c1 == c2
        
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)
        attn_mask = torch.tril(torch.ones(l1, l2)).expand(m, -1, -1).to(device)
        attention = attention + torch.log(padding_mask + 1e-9) # mask the zero paddings. log(1e-6) for zero paddings
        
        attention = self.softmax(attention) 
        attention = attention * padding_mask
        
        attention = attention.permute(0,2,1)
        out = torch.bmm(proj_val, attention)
        return out

class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type): # r1 = r2
        super(AttLayer, self).__init__()
        
        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        
        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder','decoder']
        
        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask()
        
    
    def construct_window_mask(self):
        '''
            construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        '''
        window_mask = torch.zeros((1, self.bl, self.bl + 2* (self.bl //2)))
        for i in range(self.bl):
            window_mask[:, :, i:i+self.bl] = 1
        return window_mask.to(device)
    
    def forward(self, x1, x2, mask, causal = False, short = False, short_window_scale = 0.2):
        # x1 from the encoder
        # x2 from the decoder
        
        query = self.query_conv(x1)
        key = self.key_conv(x1)
        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)
            
        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value,mask, causal, short, short_window_scale)
        elif self.att_type == 'block_att':
            return self._block_wise_self_att(query, key, value,mask, causal, short, short_window_scale)
        elif self.att_type == 'sliding_att':
            return self._sliding_window_self_att(query, key, value,mask, causal, short, short_window_scale)

    
    def _normal_self_att(self,q,k,v, mask, causal = False, short = False, short_window_scale = 0.2):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:]
        
        attn_mask = torch.tril(torch.ones(L, L)).expand(m_batchsize, -1, -1).to(device).bool()
        attn_mask = attn_mask.logical_not()
        output, attn = ScaledDotProductAttention()(q.permute(0, 2, 1), k.permute(0, 2, 1), v.permute(0, 2, 1), attn_mask)
        output = output.transpose(-2, -1)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]  
        
    def _block_wise_self_att(self, q,k,v, mask,causal = False, short = False,short_window_scale = 0.2):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1

        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        k = k.reshape(m_batchsize, c2, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c2, self.bl)
        v = v.reshape(m_batchsize, c3, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c3, self.bl)
        attn_mask = torch.tril(torch.ones(self.bl, self.bl)).expand(m_batchsize * nb, -1, -1).to(device).bool()
        attn_mask = attn_mask.logical_not()
        output, attn = ScaledDotProductAttention()(q.permute(0, 2, 1), k.permute(0, 2, 1), v.permute(0, 2, 1), attn_mask,short,short_window_scale)
        output = output.transpose(-2, -1)
        output = self.conv_out(F.relu(output))
        
        output = output.reshape(m_batchsize, nb, -1, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, -1, nb * self.bl)

        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]  
    
    def _sliding_window_self_att(self, q,k,v, mask,causal = False, short = False, short_window_scale = 0.2):
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()
        
        
        assert m_batchsize == 1  # currently, we only accept input with batch size 1
        # padding zeros for the last segment
        nb = L // self.bl 
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1
        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:], torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)],dim=-1)
        
        # sliding window approach, by splitting query_proj and key_proj into shape (c1, l) x (c1, 2l)
        # sliding window for query_proj: reshape
        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        
        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = torch.cat([torch.zeros(m_batchsize, c2, self.bl // 2).to(device), k, torch.zeros(m_batchsize, c2, self.bl // 2).to(device)], dim=-1)
        v = torch.cat([torch.zeros(m_batchsize, c3, self.bl // 2).to(device), v, torch.zeros(m_batchsize, c3, self.bl // 2).to(device)], dim=-1)
        padding_mask = torch.cat([torch.zeros(m_batchsize, 1, self.bl // 2).to(device), padding_mask, torch.zeros(m_batchsize, 1, self.bl // 2).to(device)], dim=-1)
        
        # 2. reshape key_proj of shape (m_batchsize*nb, c1, 2*self.bl)
        k = torch.cat([k[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # special case when self.bl = 1
        v = torch.cat([v[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) 
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat([padding_mask[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # of shape (m*nb, 1, 2l)
        final_mask = self.window_mask.repeat(m_batchsize * nb, 1, 1) * padding_mask 
        output = self.att_helper.scalar_dot_att(q, k, v, final_mask,causal)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, -1, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, -1, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]


class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
#         assert v_dim % num_head == 0
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type)) for i in range(num_head)])
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        out = self.conv_out(self.dropout(out))
        return out
            

class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation * 2, padding_mode='replicate', dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            nn.Dropout(),
            nn.Conv1d(out_channels, out_channels, 1)
        )
        
    def forward(self, x):
        return self.layer(x)
    

class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type, stage=stage) # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()
        self.alpha = alpha
        self.dilation = dilation
        
    def forward(self, x, f, mask, causal=False, short = False, short_window_scale = 0.2):
        out = self.feed_forward(x)
        out = out = out[..., :-self.dilation*2]
        out = self.alpha * self.att_layer(out, f, mask, causal, short, short_window_scale) + out
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0,2,1) # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return x + self.pe[:, :, 0:x.shape[2]]

class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha,causal=True):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha) for i in # 2**i
             range(num_layers)])
        
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate
        self.causal = causal
        self.gru_app = nn.GRU(num_f_maps, num_f_maps, num_layers=1, batch_first=True, bidirectional=not causal)
        self.conv_app = nn.Conv1d(num_f_maps, num_classes, 1)
        self.prob_fusion = ProbabilityProgressFusionModel(num_classes)

    def forward(self, x, mask, causal=False, short = False, short_window_scale = 0.2):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            x = self.dropout(x)
            x = x.squeeze(2)

        feature = self.conv_1x1(x)

        for layer in self.layers:
            feature = layer(feature, None, mask, causal, short, short_window_scale)

        progress_out, _ = self.gru_app(feature.permute(0, 2, 1))
        progress_out = progress_out.permute(0, 2, 1)
        progress_out = self.conv_app(progress_out) * mask[:, 0:1, :]
        prob_out = self.conv_out(feature) * mask[:, 0:1, :]
        out = self.prob_fusion(prob_out, progress_out)
        out = out * mask[:, 0:1, :]

        return out, feature, progress_out


class BED(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha,causal=True):
        super(BED, self).__init__()       
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha) for i in # 2 ** i
             range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        self.gru_app = nn.GRU(num_f_maps, num_f_maps, num_layers=1, batch_first=True, bidirectional=not causal)
        self.conv_app = nn.Conv1d(num_f_maps, num_classes, 1)
        self.prob_fusion = ProbabilityProgressFusionModel(num_classes)

    def forward(self, x, fencoder, mask, causal=False, short = False, short_window_scale = 0.2):
        feature = self.conv_1x1(x)
        
        for layer in self.layers:
            feature = layer(feature, fencoder, mask,causal,short, short_window_scale)
        progress_out, _ = self.gru_app(feature.permute(0, 2, 1))
        progress_out = progress_out.permute(0, 2, 1)
        progress_out = self.conv_app(progress_out) * mask[:, 0:1, :]

        prob_out = self.conv_out(feature) * mask[:, 0:1, :]
        out = self.prob_fusion(prob_out, progress_out)
        out = out * mask[:, 0:1, :]
        return out, feature, progress_out

class FeatureFusionModel(nn.Module):
    def __init__(self, num_f_maps):
        super(FeatureFusionModel, self).__init__()
        self.num_f_maps = num_f_maps
        self.conv = nn.Conv1d(num_f_maps*2, num_f_maps, 1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(num_f_maps, num_f_maps, 1)
    def forward(self, in_cls, in_prg):
        input_concat = torch.cat((in_cls, in_prg), dim=1)
        out = self.conv(input_concat)
        out = self.relu(out)
        out = self.conv2(out)

        return out


class PBE(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, causal=True, use_graph=True, **graph_args):
        super(PBE, self).__init__()
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type='block_att', alpha=1,causal=causal)
        self.decoders = nn.ModuleList([copy.deepcopy(BED(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='block_att', alpha=exponential_descrease(s),causal=causal)) for s in range(num_decoders)]) # num_decoders
        self.prototype_fusion = FeatureFusionModel(num_f_maps)
        self.causal = causal
        self.use_graph = use_graph
        if use_graph:
            self.graph_learner = TaskGraphLearner(**graph_args)
    def forward(self, x, mask, prototypes, causal=False, short = False, short_window_scale = 0.2):
        out, feature, progress_out = self.encoder(x, mask, causal = self.causal)
        _, predicted = torch.max(out.data, 1)
        predicted = predicted.squeeze()
        if predicted.dim() == 0:
            predicted = predicted.unsqueeze(0)
        prototype_feature = []
        for i in range(predicted.shape[0]):
            class_idx = str(predicted[i].item())
            class_prototypes = prototypes[class_idx]
            
            current_feature = feature[:,:,i]
            if class_prototypes is None:
                prototype_feature.append(current_feature)
                continue
            similarity = F.cosine_similarity(class_prototypes.unsqueeze(0), current_feature.unsqueeze(0), dim=-1)
            max_sim_idx = similarity.argmax(dim=-1)  
            best_prototype = class_prototypes[max_sim_idx]
            prototype_feature.append(best_prototype)

        prototype_feature = torch.stack(prototype_feature, dim=0)
        prototype_feature = prototype_feature.permute(1,2,0)  
        feature = self.prototype_fusion(feature, prototype_feature)

        prob_outputs = out.unsqueeze(0)
        progress_outputs = progress_out.unsqueeze(0)
        for decoder in self.decoders:
            out, feature,progress_out = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature* mask[:, 0:1, :], mask, causal = self.causal, short = short, short_window_scale = short_window_scale)
            prob_outputs = torch.cat((prob_outputs, out.unsqueeze(0)), dim=0)
            progress_outputs = torch.cat((progress_outputs, progress_out.unsqueeze(0)), dim=0)
 
        return prob_outputs,progress_outputs


class PLT(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, causal=True, use_graph=True,cluster = 'kmeans', **graph_args):
        super(PLT, self).__init__()
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type='block_att', alpha=1,causal=causal)
        self.decoders = nn.ModuleList([copy.deepcopy(BED(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='block_att', alpha=exponential_descrease(s),causal=causal)) for s in range(num_decoders)]) # num_decoders
        self.causal = causal
        self.num_classes = num_classes
        self.use_graph = use_graph
        self.clustering_method = cluster
        if use_graph:
            self.graph_learner = TaskGraphLearner(**graph_args)
        self.num_clusters = 8
        self.init_prototypes()
        
    def forward(self, x, mask, causal=False):
        out, feature, progress_out = self.encoder(x, mask, causal = self.causal)
        prob_outputs = out.unsqueeze(0)
        progress_outputs = progress_out.unsqueeze(0)
        feature_out = feature
        for decoder in self.decoders:
            out, feature,progress_out = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature* mask[:, 0:1, :], mask, causal = self.causal)
            prob_outputs = torch.cat((prob_outputs, out.unsqueeze(0)), dim=0)
            progress_outputs = torch.cat((progress_outputs, progress_out.unsqueeze(0)), dim=0)
 
        return prob_outputs,progress_outputs,feature_out
    
    def init_prototypes(self):
        self.prototypes = {}
        self.threshold_dict = {}
        for cls_idx in range(self.num_classes):
            self.prototypes[str(cls_idx)] = None
        if self.num_clusters > 1:
            self.cluster_models = {}
            for i in range(self.num_classes):
                if self.clustering_method == 'kmeans':
                    self.cluster_models[str(i)] = KMeans(n_clusters=self.num_clusters, verbose=False)
                elif self.clustering_method == 'gmm':
                    self.cluster_models[str(i)] = GaussianMixture(n_components=self.num_clusters, verbose=0)
                elif self.clustering_method == 'agglomerative':
                    self.cluster_models[str(i)] = AgglomerativeClustering(n_clusters=self.num_clusters)
                elif self.clustering_method == 'birch':
                    self.cluster_models[str(i)] = Birch(n_clusters=self.num_clusters)
                elif self.clustering_method == 'mean_shift':
                    self.cluster_models[str(i)] = MeanShift()
    def generate_prototypes(self, features, labels, num_clusters=8):
        batch_size, num_features, num_frames = features.shape
        for cls_idx in range(self.num_classes):
            mask = (labels == cls_idx).unsqueeze(1).expand(batch_size, num_features, num_frames)
            class_features = features[mask].view(num_features, -1)
            if class_features.size(1) == 0:
                continue
            class_features = class_features.permute(1, 0).contiguous()
            if self.prototypes[str(cls_idx)] is None:
                self.prototypes[str(cls_idx)] = class_features.detach()
            else:
                self.prototypes[str(cls_idx)] = torch.cat((self.prototypes[str(cls_idx)], class_features.detach()), dim=0)
    def flush_prototypes(self):
        for key, value in self.prototypes.items():
            if value is None:
                continue
            if self.num_clusters > 1:
                cluster_labels = self.cluster_models[key].fit_predict(value.detach().cpu().numpy())
                cluster_centers = []
                for i in range(self.num_clusters):
                    cluster_centers.append(value[cluster_labels == i].mean(dim=0, keepdim=True))
                cluster_centers = torch.cat(cluster_centers, dim=0)
                cluster_centers = cluster_centers.to(value.device)
                self.prototypes[key] = cluster_centers
            else:
                self.prototypes[key] = value.mean(dim=1, keepdim=True)  
class Trainer:
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate,causal ,logger, progress_lw=1.0, graph_lw=0.01, use_graph=True, init_graph_path='', learnable=True, gamma=0.05, balancing_lw=0.6, be_lw = 0.2, short_window_scale = 0.2,cluster = 'kmeans'):
        self.model = PBE(3, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, causal=causal, use_graph=use_graph, init_graph_path=init_graph_path, learnable=learnable)
        self.prototype_model = PLT(3, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, causal=causal, use_graph=use_graph, init_graph_path=init_graph_path, learnable=learnable, cluster=cluster)
        self.prototypes = {}
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.num_classes = num_classes

        self.progress_lw = progress_lw
        self.use_graph = use_graph
        self.graph_lw = graph_lw
        self.logger = logger
        self.gamma = gamma
        self.balancing_lw = balancing_lw
        self.be_lw = be_lw
        self.short_window_scale = short_window_scale
        self.cluster = cluster
        
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate, feature_transpose=False, map_delimiter=' ', dataset = 'gtea'):
        self.model.eval()
        self.prototype_model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            self.prototype_model.to(device)
            self.prototype_model.load_state_dict(torch.load(model_dir + "/prototype_epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                if dataset in ['egoprocel']:
                    features = np.load(features_path + vid + '.npy')
                else:
                    features = np.load(features_path + vid.split('.')[0] + '.npy')
                if feature_transpose:
                    features = features.T
                features = features[:, ::sample_rate]

                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                prototypes = self.prototype_model.prototypes
                predictions, progress_predictions = self.model(input_x, torch.ones(input_x.size(), device=device), prototypes)
                final_predictions = self.model.graph_learner.inference(predictions[-1], progress_predictions[-1])

                final_predictions_seq = final_predictions.permute(2, 0, 1).view(-1, self.num_classes)
                final_predictions_prob = F.softmax(final_predictions_seq, dim=1)
                _, predicted = torch.max(final_predictions_prob.data, 1)                
                predicted = predicted.squeeze()

                recognition = []

                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                if dataset in ['egoprocel']:
                    f_name = vid.split('/')[-1]
                else:
                    f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(map_delimiter.join(recognition))
                f_ptr.close()

    def predict_online(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate, feature_transpose=False, map_delimiter=' ', dataset = 'gtea'):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            self.prototype_model.to(device)
            self.prototype_model.load_state_dict(torch.load(model_dir + "/prototype_epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                if feature_transpose:
                    features = features.T
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                n_frames = input_x.shape[-1]
                recognition = []
                prototypes = self.prototype_model.prototypes
                for frame_i in tqdm.tqdm(range(n_frames)):
                    curr_input_x = input_x[:, :, :frame_i+1]
                    predictions, progress_predictions = self.model(curr_input_x, torch.ones(curr_input_x.size(), device=device), prototypes)
                    final_predictions = self.model.graph_learner.inference(predictions[-1], progress_predictions[-1])
                    final_predictions_seq = final_predictions.permute(2, 0, 1).view(-1, self.num_classes)
                    _, predicted = torch.max(final_predictions_seq.data, 1)
                    predicted = predicted.squeeze()
                    if predicted.dim() == 0:
                        predicted = predicted.unsqueeze(0)

                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[-1].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(map_delimiter.join(recognition))
                f_ptr.close()



if __name__ == '__main__':
    pass
