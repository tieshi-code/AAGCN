import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

RIGHT_LEG_JOINTS = [12, 14, 16]
LEFT_LEG_JOINTS = [11, 13, 15]


class AmputeeLegAttention(nn.Module):
    
    def __init__(self, channels, amputee_joints, reduction=4):
        super(AmputeeLegAttention, self).__init__()
        self.amputee_joints = amputee_joints
        self.num_joints = 17 
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # (N, C, T, V) -> (N, C, 1, 1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
        
        self.joint_weights = nn.Parameter(torch.ones(self.num_joints))
        
    def forward(self, x, amputee_type='none', attention_weight=2.0):
        N, C, T, V = x.size()
        
        y = self.global_pool(x)  # (N, C, 1, 1)
        y = y.view(N, C)  # (N, C)
        
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)  # (N, C)
        y = y.view(N, C, 1, 1)  # (N, C, 1, 1)
        
        x_channel_att = x * y  # (N, C, T, V)
        
        joint_mask = torch.ones(N, 1, 1, V, device=x.device)
        
        if isinstance(amputee_type, (list, tuple)):
            for i, amp_type in enumerate(amputee_type):
                if i >= N:
                    break
                if amp_type == 'right_leg':
                    for j in RIGHT_LEG_JOINTS:
                        if j < V:
                            joint_mask[i, :, :, j] = attention_weight
                elif amp_type == 'left_leg':
                    for j in LEFT_LEG_JOINTS:
                        if j < V:
                            joint_mask[i, :, :, j] = attention_weight
                elif amp_type == 'both_legs':
                    for j in RIGHT_LEG_JOINTS + LEFT_LEG_JOINTS:
                        if j < V:
                            joint_mask[i, :, :, j] = attention_weight
        elif isinstance(amputee_type, torch.Tensor):
            pass
        else:
            if amputee_type == 'right_leg':
                for j in RIGHT_LEG_JOINTS:
                    if j < V:
                        joint_mask[:, :, :, j] = attention_weight
            elif amputee_type == 'left_leg':
                for j in LEFT_LEG_JOINTS:
                    if j < V:
                        joint_mask[:, :, :, j] = attention_weight
            elif amputee_type == 'both_legs':
                for j in RIGHT_LEG_JOINTS + LEFT_LEG_JOINTS:
                    if j < V:
                        joint_mask[:, :, :, j] = attention_weight
        
        joint_weights = self.joint_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, V)
        joint_mask = joint_mask * joint_weights
        
        mask_mean = joint_mask.mean()
        if mask_mean > 0:
            joint_mask = joint_mask / mask_mean
        
        x_att = x_channel_att * joint_mask  # (N, C, T, V)
        
        return x_att, joint_mask


class st_gcn_amputee(nn.Module):
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True,
                 use_amputee_attention=True):
        super(st_gcn_amputee, self).__init__()
        
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        
        self.use_amputee_attention = use_amputee_attention
        
        # GCN + TCN
        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        
        if self.use_amputee_attention:
            self.amputee_attention = AmputeeLegAttention(out_channels, [])
        
        # Residual
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x, A, amputee_type='none', attention_weight=2.0):
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x)
        
        if self.use_amputee_attention:
            x, _ = self.amputee_attention(x, amputee_type=amputee_type, attention_weight=attention_weight)
        
        x = x + res
        return self.relu(x), A


class Model_Amputee(nn.Module):
    
    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, use_amputee_attention=True, **kwargs):
        super().__init__()
        
        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        self.use_amputee_attention = use_amputee_attention
        
        if 'attention_weight' in kwargs:
            self.attention_weight = kwargs.pop('attention_weight')
        else:
            self.attention_weight = attention_weight  
        
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_amputee(in_channels, 64, kernel_size, 1, residual=False, 
                          use_amputee_attention=use_amputee_attention, **kwargs0),
            st_gcn_amputee(64, 64, kernel_size, 1, 
                          use_amputee_attention=use_amputee_attention, **kwargs),
            st_gcn_amputee(64, 64, kernel_size, 1, 
                          use_amputee_attention=use_amputee_attention, **kwargs),
            st_gcn_amputee(64, 64, kernel_size, 1, 
                          use_amputee_attention=use_amputee_attention, **kwargs),
            st_gcn_amputee(64, 128, kernel_size, 2, 
                          use_amputee_attention=use_amputee_attention, **kwargs),
            st_gcn_amputee(128, 128, kernel_size, 1, 
                          use_amputee_attention=use_amputee_attention, **kwargs),
            st_gcn_amputee(128, 128, kernel_size, 1, 
                          use_amputee_attention=use_amputee_attention, **kwargs),
            st_gcn_amputee(128, 256, kernel_size, 2, 
                          use_amputee_attention=use_amputee_attention, **kwargs),
            st_gcn_amputee(256, 256, kernel_size, 1, 
                          use_amputee_attention=use_amputee_attention, **kwargs),
            st_gcn_amputee(256, 256, kernel_size, 1, 
                          use_amputee_attention=use_amputee_attention, **kwargs),
        ))
        
        # initialize parameters for edge importance weighting
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)
        
        # fcn for prediction
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)
    
    def forward(self, x, amputee_type='none'):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        
        if isinstance(amputee_type, (list, tuple)):
            if len(amputee_type) == N:
                amputee_type_expanded = []
                for amp_type in amputee_type:
                    amputee_type_expanded.extend([amp_type] * M)
                current_type = amputee_type_expanded
            else:
                current_type = amputee_type[0] if len(amputee_type) > 0 else 'none'
        elif isinstance(amputee_type, torch.Tensor):
            current_type = 'none'
        elif isinstance(amputee_type, str):
            current_type = amputee_type
        else:
            current_type = 'none'
        
        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance, amputee_type=current_type, attention_weight=self.attention_weight)
        
        # global pooling
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)
        
        # prediction
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        
        return x
    
    def extract_feature(self, x, amputee_type='none'):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        
        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            if isinstance(amputee_type, str):
                current_type = amputee_type
            else:
                current_type = 'none'
            x, _ = gcn(x, self.A * importance, current_type)
        
        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)
        
        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)
        
        return output, feature
