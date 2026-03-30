import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph

# AlphaPose COCO 17点关节索引
# 右腿关节：RHip=12, RKnee=14, RAnkle=16
# 左腿关节：LHip=11, LKnee=13, LAnkle=15
RIGHT_LEG_JOINTS = [12, 14, 16]
LEFT_LEG_JOINTS = [11, 13, 15]


class AmputeeLegAttention(nn.Module):
    """截肢腿专用注意力模块（SE-like）"""
    
    def __init__(self, channels, amputee_joints, reduction=4):
        """
        Args:
            channels: 特征通道数
            amputee_joints: 截肢腿关节索引列表（如 [12,14,16] 表示右腿）
            reduction: 压缩比例
        """
        super(AmputeeLegAttention, self).__init__()
        self.amputee_joints = amputee_joints
        self.num_joints = 17  # AlphaPose 固定17个关节
        
        # SE-like 注意力：全局池化 -> FC -> ReLU -> FC -> Sigmoid
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # (N, C, T, V) -> (N, C, 1, 1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
        
        # 关节级别的注意力权重（可学习）
        self.joint_weights = nn.Parameter(torch.ones(self.num_joints))
        
    def forward(self, x, amputee_type='none', attention_weight=2.0):
        """
        Args:
            x: (N, C, T, V) 特征图
            amputee_type: 'right_leg', 'left_leg', 'both_legs', 'none' 或 list/tensor
            attention_weight: 截肢腿关节的注意力权重倍数（默认2.0）
        Returns:
            x_att: 加权后的特征图
            attention_map: 注意力权重图 (N, 1, 1, V)
        """
        N, C, T, V = x.size()
        
        # 1. 通道注意力（SE-like）
        # 全局池化
        y = self.global_pool(x)  # (N, C, 1, 1)
        y = y.view(N, C)  # (N, C)
        
        # FC -> ReLU -> FC -> Sigmoid
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)  # (N, C)
        y = y.view(N, C, 1, 1)  # (N, C, 1, 1)
        
        # 应用通道注意力
        x_channel_att = x * y  # (N, C, T, V)
        
        # 2. 关节级别注意力（对截肢腿关节加权）
        joint_mask = torch.ones(N, 1, 1, V, device=x.device)
        
        # 处理batch级别的截肢类型
        if isinstance(amputee_type, (list, tuple)):
            # 列表：每个样本一个类型
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
            # Tensor: 每个样本一个类型（需要转换为字符串）
            # 简化处理：暂时不支持tensor，使用none
            pass
        else:
            # 字符串：所有样本使用相同类型
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
        
        # 应用可学习的关节权重
        joint_weights = self.joint_weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, V)
        joint_mask = joint_mask * joint_weights
        
        # 归一化（保持能量，避免梯度爆炸）
        mask_mean = joint_mask.mean()
        if mask_mean > 0:
            joint_mask = joint_mask / mask_mean
        
        x_att = x_channel_att * joint_mask  # (N, C, T, V)
        
        return x_att, joint_mask


class st_gcn_amputee(nn.Module):
    """带截肢腿注意力的 ST-GCN 块"""
    
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
        
        # 截肢腿注意力模块
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
        
        # 应用截肢腿注意力
        if self.use_amputee_attention:
            x, _ = self.amputee_attention(x, amputee_type=amputee_type, attention_weight=attention_weight)
        
        x = x + res
        return self.relu(x), A


class Model_Amputee(nn.Module):
    """改进的 ST-GCN 模型，支持截肢腿注意力"""
    
    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting, use_amputee_attention=True, **kwargs):
        super().__init__()
        
        # load graph
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)
        
        self.use_amputee_attention = use_amputee_attention
        
        # 从kwargs中提取attention_weight（如果配置文件中传入）
        if 'attention_weight' in kwargs:
            self.attention_weight = kwargs.pop('attention_weight')
        else:
            self.attention_weight = attention_weight  # 使用默认值或参数传入的值
        
        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        
        # 使用改进的 st_gcn_amputee 块
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
        """
        Args:
            x: (N, C, T, V, M) 输入骨架数据
            amputee_type: str, list, or tensor, 截肢类型
                - 'none': 无截肢
                - 'right_leg': 右腿截肢
                - 'left_leg': 左腿截肢
                - 'both_legs': 双腿截肢
                或 list/tuple of length N，每个样本一个类型
        """
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        
        # 处理 amputee_type：扩展到 (N*M,) 维度
        if isinstance(amputee_type, (list, tuple)):
            # 列表：每个样本一个类型，扩展到 (N*M,)
            if len(amputee_type) == N:
                # 扩展到每个person
                amputee_type_expanded = []
                for amp_type in amputee_type:
                    amputee_type_expanded.extend([amp_type] * M)
                current_type = amputee_type_expanded
            else:
                current_type = amputee_type[0] if len(amputee_type) > 0 else 'none'
        elif isinstance(amputee_type, torch.Tensor):
            # Tensor暂不支持，使用none
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
        """提取特征（用于可视化或迁移学习）"""
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
