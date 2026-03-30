import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.tgcn import ConvTemporalGraphical
from net.utils.graph import Graph
from .st_gcn_amputee import AmputeeLegAttention  # 复用已有的截肢注意力模块


class st_gcn_as_amputee(nn.Module):
    """
    AS-GCN 风格的 ST-GCN 块：
    - 使用静态骨架图做结构卷积（structural branch）
    - 额外构造数据驱动的动作图（actional branch），做一次自注意力式的消息传播
    - 然后与截肢腿注意力模块结合
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
        use_amputee_attention: bool = True,
        lambda_action: float = 0.5,
    ):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.use_amputee_attention = use_amputee_attention
        self.lambda_action = lambda_action

        # 结构图卷积：与原 ST-GCN 一致
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        # 时间卷积
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

        # 动作图（actional graph）所需的线性映射：Q, K, V
        inter_channels = max(out_channels // 2, 1)
        self.theta = nn.Conv2d(out_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(out_channels, inter_channels, kernel_size=1)
        self.gcn_value = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # 截肢腿注意力
        if self.use_amputee_attention:
            self.amputee_attention = AmputeeLegAttention(out_channels, [])

        # 残差
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A, amputee_type="none", attention_weight: float = 2.0):
        # 结构图卷积 + 残差
        res = self.residual(x)
        x, A = self.gcn(x, A)  # (N, C, T, V)
        x_struct = x

        # ===== AS-GCN 风格的动作图分支 =====
        # 使用自注意力构造每一帧的动态邻接，并进行一次信息传播
        N, C, T, V = x.size()

        # 计算 Q, K, V' 特征
        q = self.theta(x)  # (N, C', T, V)
        k = self.phi(x)    # (N, C', T, V)
        v = self.gcn_value(x)  # (N, C, T, V)

        # 变形为 (N*T, V, C') / (N*T, C', V)
        q = q.permute(0, 2, 3, 1).contiguous().view(N * T, V, -1)
        k = k.permute(0, 2, 1, 3).contiguous().view(N * T, -1, V)
        v = v.permute(0, 2, 3, 1).contiguous().view(N * T, V, -1)

        # 相似度矩阵 (N*T, V, V)
        attn = torch.bmm(q, k) / math.sqrt(q.size(-1) + 1e-6)
        attn = F.softmax(attn, dim=-1)

        # 动作图信息传播
        y = torch.bmm(attn, v)  # (N*T, V, C)
        y = y.view(N, T, V, -1).permute(0, 3, 1, 2).contiguous()  # (N, C, T, V)

        # 结构 + 动作 融合
        x = x_struct + self.lambda_action * y

        # 时间卷积
        x = self.tcn(x)

        # 截肢腿注意力
        if self.use_amputee_attention:
            x, _ = self.amputee_attention(
                x, amputee_type=amputee_type, attention_weight=attention_weight
            )

        x = x + res
        return self.relu(x), A


class Model_ASGCN_Amputee(nn.Module):
    """
    基于 AS-GCN 思想的截肢注意力 ST-GCN 模型：
    - 与 Model_Amputee 结构基本一致
    - 每个 block 使用 st_gcn_as_amputee（结构 + 动作双分支）
    """

    def __init__(
        self,
        in_channels: int,
        num_class: int,
        graph_args,
        edge_importance_weighting: bool,
        use_amputee_attention: bool = True,
        attention_weight: float = 2.0,
        lambda_action: float = 0.5,
        **kwargs,
    ):
        super().__init__()

        # 图结构
        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        self.use_amputee_attention = use_amputee_attention
        self.attention_weight = attention_weight
        self.lambda_action = lambda_action

        # 网络结构
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))

        kwargs0 = {k: v for k, v in kwargs.items() if k != "dropout"}

        def make_block(cin, cout, s, first=False):
            base_kwargs = kwargs0 if first else kwargs
            return st_gcn_as_amputee(
                cin,
                cout,
                kernel_size,
                stride=s,
                residual=not first,
                use_amputee_attention=use_amputee_attention,
                lambda_action=self.lambda_action,
                **base_kwargs,
            )

        self.st_gcn_networks = nn.ModuleList(
            (
                make_block(in_channels, 64, 1, first=True),
                make_block(64, 64, 1),
                make_block(64, 64, 1),
                make_block(64, 64, 1),
                make_block(64, 128, 2),
                make_block(128, 128, 1),
                make_block(128, 128, 1),
                make_block(128, 256, 2),
                make_block(256, 256, 1),
                make_block(256, 256, 1),
            )
        )

        # 边重要性权重
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # 分类层
        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x, amputee_type="none"):
        """
        Args:
            x: (N, C, T, V, M)
            amputee_type: str 或 list[str]，长度为 N
        """
        N, C, T, V, M = x.size()

        # 与原 ST-GCN 一样的数据归一化流程
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # 扩展 amputee_type 到 N*M
        if isinstance(amputee_type, (list, tuple)):
            if len(amputee_type) == N:
                amp_expanded = []
                for t in amputee_type:
                    amp_expanded.extend([t] * M)
                current_type = amp_expanded
            else:
                current_type = list(amputee_type)
        else:
            # 单个字符串：所有样本相同类型
            current_type = [amputee_type] * (N * M)

        # 前向传播
        for (gcn, importance) in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(
                x,
                self.A * importance,
                amputee_type=current_type,
                attention_weight=self.attention_weight,
            )

        # 全局池化
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        # 分类
        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x

