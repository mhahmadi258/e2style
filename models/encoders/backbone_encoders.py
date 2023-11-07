import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE

class AdapterBlock(Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.adapter = Linear(emb_dim, emb_dim)
        

    def forward(self, x):
        out = self.adapter(x)
        res = x + out
        return res

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
        
        
        self.output_layer_4 = Sequential(BatchNorm2d(128),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(128 * 7 * 7, 512 * 5))
               
        self.output_layer_5 = Sequential(BatchNorm2d(64),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(64 * 7 * 7, 512 * 4))
        
        self.adapter_layers = nn.ModuleList([AdapterBlock(512) for _ in range(6)])
        
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)

    
    def calc_w(self, x):
        x = self.input_layer(x)
        for l in self.modulelist[:3]:
          x = l(x)
        lc_part_4 = self.output_layer_5(x).view(-1, 4, 512)
        for l in self.modulelist[3:7]:
          x = l(x)
        lc_part_3 = self.output_layer_4(x).view(-1, 5, 512)
        for l in self.modulelist[7:21]:
          x = l(x)
        lc_part_2 = self.output_layer_3(x).view(-1, 9, 512)

        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        return x
    
    def forward(self, x):
        w = self.calc_w(x)
        w_frontal = list()
        for i in range(18):
            vector = w[:,i,...]
            if i < len(self.adapter_layers):
                vector = self.adapter_layers[i](vector)
            w_frontal.append(vector.unsqueeze(1))
            
        return torch.cat(w_frontal, dim=1)          