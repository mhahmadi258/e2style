import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE

class SeqAdapterBlock(Module):
    def __init__(self, emb_dim, num_module):
        super().__init__()
        self.num_module = num_module
        self.adapters = nn.ModuleList([nn.Sequential(Linear(emb_dim, 256), nn.ReLU(), Linear(256, emb_dim)) for _ in range(num_module)])
        
    def forward(self, x):
        vectors = list()
        for i in range(self.num_module):
            vector = x[:,i,...]
            out = self.adapters[i](vector)
            vectors.append(out)
        return torch.stack(vectors,dim=1)
   
class ParallelAdapterBlock(Module):
    def __init__(self, in_channel, num_module):
        super().__init__()
        self.num_module = num_module
        self.adapter = Sequential(BatchNorm2d(in_channel),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(in_channel * 7 * 7, 512 * num_module))
        

    def forward(self, x):
        out = self.adapter(x).view(-1, self.num_module, 512)
        return out

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
        
        self.parallel_adapter_layer_3 = ParallelAdapterBlock(256, 9)
        self.seq_adapter_layer_3 = SeqAdapterBlock(512, 9)
        
        self.output_layer_4 = Sequential(BatchNorm2d(128),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(128 * 7 * 7, 512 * 5))
        
        self.parallel_adapter_layer_4 = ParallelAdapterBlock(128, 5)
        self.seq_adapter_layer_4 = SeqAdapterBlock(512, 5)
        
        self.output_layer_5 = Sequential(BatchNorm2d(64),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(64 * 7 * 7, 512 * 4))
        
        self.parallel_adapter_layer_5 = ParallelAdapterBlock(64, 4)
        self.seq_adapter_layer_5 = SeqAdapterBlock(512, 4)
        
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)

    def forward(self, x):
        x = self.input_layer(x)
        for l in self.modulelist[:3]:
          x = l(x)
        lc_part_4 = self.output_layer_5(x).view(-1, 4, 512)
        lc_part_4 = self.parallel_adapter_layer_5(x, lc_part_4) \
            + self.seq_adapter_layer_5(lc_part_4)
        for l in self.modulelist[3:7]:
          x = l(x)
        lc_part_3 = self.output_layer_4(x).view(-1, 5, 512)
        lc_part_3 = self.parallel_adapter_layer_4(x, lc_part_3) \
            + self.seq_adapter_layer_4(lc_part_3)
        for l in self.modulelist[7:21]:
          x = l(x)
        lc_part_2 = self.output_layer_3(x).view(-1, 9, 512)
        lc_part_2 = self.parallel_adapter_layer_3(x, lc_part_2) \
            + self.seq_adapter_layer_3(lc_part_2)

        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        return x