import torch
from torch import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.cosloss = torch.nn.CosineEmbeddingLoss()
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x):
        n_samples = y.shape[0]
        cos_target = torch.ones(n_samples).float().cuda()
        loss = 0
        sim_improvement = 0 
        id_logs = list()
        count = 0
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        x_feates = self.extract_feats(x)
        # print(x_feates[4].shape)
        for i in range(5):
            y_feat_detached = y_feats[i].detach()
            loss += self.cosloss(y_feat_detached, y_hat_feats[i], cos_target)
            if i == 4:
                for j in range(n_samples):
                    diff_target = 1 - self.cosloss(y_hat_feats[i][j], y_feat_detached[j],torch.tensor(1))
                    diff_input = 1 - self.cosloss(y_hat_feats[i][j], x_feates[i][j],torch.tensor(1))
                    diff_views = 1 - self.cosloss(y_feat_detached[j], x_feates[i][j],torch.tensor(1))
                    id_logs.append({'diff_target': float(diff_target),
                                    'diff_input': float(diff_input),
                                    'diff_views': float(diff_views)})
                    id_diff = float(diff_target) - float(diff_views)
                    sim_improvement += id_diff
                    count += 1
        
        return loss, sim_improvement/ count, id_logs
