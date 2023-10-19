import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear, Conv2d, BatchNorm2d, PReLU, Sequential, Module, MultiheadAttention

from models.encoders.helpers import get_blocks, Flatten, bottleneck_IR, bottleneck_IR_SE

class AdapterBlock(Module):
    def __init__(self, in_d, out_d, num_module):
        super().__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.num_module = num_module
        self.adapters = nn.ModuleList([Linear(in_d, out_d, device='cuda:0') for _ in range(num_module)])
        self.attns = nn.ModuleList([MultiheadAttention(out_d, 4) for _ in range(num_module)])
        self.attns_cls_token = nn.Parameter(torch.rand(num_module, 1, out_d, device='cuda:0'))
        self.out_attns = nn.ModuleList([Linear(out_d, out_d, device='cuda:0') for _ in range(num_module)])
 

    def forward(self, x):
        vectors = list()
        for i in range(self.num_module):
            vector = x[:,i,...]
            out = self.adapters[i](vector)
            kqv = torch.stack((vector, out))
            kqv = torch.vstack((self.attns_cls_token[i].repeat((1,vector.shape[0],1)), kqv))
            res = self.attns[i](kqv, kqv, kqv)[0]
            res = self.out_attns[i](res[0])
            vectors.append(res)
        return torch.stack(vectors,dim=1)

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
        
        self.adapter_layer_3 = AdapterBlock(512,512,9)
        
        self.output_layer_4 = Sequential(BatchNorm2d(128),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(128 * 7 * 7, 512 * 5))
        
        self.adapter_layer_4 = AdapterBlock(512,512,5)
        
        self.output_layer_5 = Sequential(BatchNorm2d(64),
                                         torch.nn.AdaptiveAvgPool2d((7, 7)),
                                         Flatten(),
                                         Linear(64 * 7 * 7, 512 * 4))
        
        self.adapter_layer_5 = AdapterBlock(512,512,4)
        
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
        lc_part_4 = self.adapter_layer_5(lc_part_4)
        for l in self.modulelist[3:7]:
          x = l(x)
        lc_part_3 = self.output_layer_4(x).view(-1, 5, 512)
        lc_part_3 = self.adapter_layer_4(lc_part_3)
        for l in self.modulelist[7:21]:
          x = l(x)
        lc_part_2 = self.output_layer_3(x).view(-1, 9, 512)
        lc_part_2 = self.adapter_layer_3(lc_part_2)

        x = torch.cat((lc_part_2, lc_part_3, lc_part_4), dim=1)
        return x

class BackboneEncoderRefineStage(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(BackboneEncoderRefineStage, self).__init__()
        # print('Using BackboneEncoderRefineStage')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(6, 64, (3, 3), 1, 1, bias=False),
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
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        self.modulelist = list(self.body)

    def forward(self, x, first_stage_output_image):
        x = torch.cat((x,first_stage_output_image), dim=1)
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