import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module, MultiheadAttention

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE


class AttentionBlock(Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.atn = MultiheadAttention(emb_dim, num_heads=num_heads)
        self.mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                 nn.GELU(),
                                 nn.Linear(emb_dim,emb_dim))
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        
    def forward(self, x):
        out = x + self.atn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        out = out + self.mlp(self.norm2(out))
        return out

    
class SeqAdapterBlock(Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.attn = AttentionBlock(emb_dim, 4)
        self.attn_cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))
        self.out_attn = Linear(emb_dim, emb_dim)

    def forward(self, x):
        kqv = torch.vstack((self.attn_cls_token.repeat((1, x.shape[1],1)), x))
        res = self.attn(kqv)
        res = self.out_attn(res[0])
        return res
    
    
class AdapterBlock(Module):
    def __init__(self, in_channel, num_module):
        super().__init__()
        self.num_module = num_module
        self.adapter = Sequential(BatchNorm2d(in_channel),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(in_channel * 7 * 7, 512 * num_module))
        

    def forward(self, x , w):
        out = self.adapter(x).view(-1, self.num_module, 512)
        return w + out

class BackboneEncoderFirstStage(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderFirstStage, self).__init__()
        # print('Using BackboneEncoderFirstStage')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(opts.input_nc, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))

        self.output_layer_3 = Sequential(BatchNorm2d(256),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(256 * 7 * 7, 512 * 9))
        
        self.adapter_layer_3 = AdapterBlock(256,9)
        
        self.output_layer_4 = Sequential(BatchNorm2d(128),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(128 * 7 * 7, 512 * 5))
        
        self.adapter_layer_4 = AdapterBlock(128,5)
        
        self.output_layer_5 = Sequential(BatchNorm2d(64),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(64 * 7 * 7, 512 * 4))
        
        self.adapter_layer_5 = AdapterBlock(64,4)
        
        
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)
        
        self.seq_adapters = nn.ModuleList([SeqAdapterBlock(512) for _ in range(18)])

    def calc_w(self, x):
        x = self.input_layer(x)
        for l in self.modulelist[:3]:
          x = l(x)
        lc_part_4 = self.output_layer_5(x).view(-1, 4, 512)
        lc_part_4_frontal = self.adapter_layer_5(x, lc_part_4)
        for l in self.modulelist[3:7]:
          x = l(x)
        lc_part_3 = self.output_layer_4(x).view(-1, 5, 512)
        lc_part_3_frontal = self.adapter_layer_4(x, lc_part_3)
        for l in self.modulelist[7:21]:
          x = l(x)
        lc_part_2 = self.output_layer_3(x).view(-1, 9, 512)
        lc_part_2_frontal = self.adapter_layer_3(x, lc_part_2)

        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        x_frontal = torch.cat((lc_part_2_frontal, lc_part_3_frontal, lc_part_4_frontal), dim=1)
        return x, x_frontal
    
    def forward(self, x):
        ws = list()
        ws_frontal = list()
        for i in range(x.shape[1]):
            w, w_frontal = self.calc_w(x[:,i,...])
            ws.append(w)
            ws_frontal.append(w_frontal)  
        ws = torch.stack(ws)
        ws_frontal = torch.stack(ws_frontal)
        
        vectors = list()
        for i in range(18):
            vector = ws[:,:,i,...]
            frontal_vector = ws_frontal[:,:,i,...]
            vector = self.seq_adapters[i](vector)
            frontal_vector = torch.mean(frontal_vector, dim=0)
            vectors.append(vector + frontal_vector)
        return torch.stack(vectors, dim=1)