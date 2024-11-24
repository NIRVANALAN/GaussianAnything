import torch
from pdb import set_trace as st
from torch import nn
from .model_irse import Backbone
from .paths_config import model_paths


class IDLoss(nn.Module):

    def __init__(self, device):
        # super(IDLoss, self).__init__()
        super().__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112,
                                num_layers=50,
                                drop_ratio=0.6,
                                mode='ir_se').to(device)
        # self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        try:
            face_net_model = torch.load(model_paths['ir_se50'],
                                        map_location=device)
        except Exception as e:
            face_net_model = torch.load(model_paths['ir_se50_hwc'],
                                        map_location=device)

        self.facenet.load_state_dict(face_net_model)

        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, x):
        n_samples, _, H, W = x.shape
        assert H == W == 256, 'idloss needs 256*256 input images'

        x_feats = self.extract_feats(x)
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            diff_input = y_hat_feats[i].dot(x_feats[i])
            diff_views = y_feats[i].dot(x_feats[i])
            id_logs.append({
                'diff_target': float(diff_target),
                'diff_input': float(diff_input),
                'diff_views': float(diff_views)
            })
            loss += 1 - diff_target
            id_diff = float(diff_target) - float(diff_views)
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, id_logs
