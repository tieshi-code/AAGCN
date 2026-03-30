import math
from typing import List, Sequence, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.graph import Graph
from net.utils.tgcn import ConvTemporalGraphical


# AlphaPose (17 joints) indices
# right_leg joints: RHip=12, RKnee=14, RAnkle=16
# left_leg joints: LHip=11, LKnee=13, LAnkle=15
RIGHT_LEG_JOINTS: List[int] = [12, 14, 16]
LEFT_LEG_JOINTS: List[int] = [11, 13, 15]


def make_joint_prior_mask(
    amputation_type: str,
    num_joints: int = 17,
    alpha: float = 2.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Amputation-aware prior mask m (V,).

    According to the paper requirement:
      - amputated-limb joints have weight alpha
      - other joints have weight 1.0
    """
    if dtype is None:
        dtype = torch.float32

    m = torch.ones(num_joints, device=device, dtype=dtype)
    t = (amputation_type or "none").strip().lower()

    # Support both paper-style and dataset-style naming.
    if t in {"right_limb", "right_leg", "right", "right_amputee", "right_limb_amputation"}:
        for j in RIGHT_LEG_JOINTS:
            if 0 <= j < num_joints:
                m[j] = alpha
    elif t in {"left_limb", "left_leg", "left", "left_amputee", "left_limb_amputation"}:
        for j in LEFT_LEG_JOINTS:
            if 0 <= j < num_joints:
                m[j] = alpha
    elif t in {"both_limbs", "both_legs", "both", "both_limbs_amputation"}:
        for j in RIGHT_LEG_JOINTS + LEFT_LEG_JOINTS:
            if 0 <= j < num_joints:
                m[j] = alpha
    else:
        # 'none' or unknown -> all ones
        pass

    return m


class AmputeeLimbAttention(nn.Module):
    """
    Amputee Limb Attention Module (ALAM).

    Implements the paper's two-step attention:
      1) Channel attention:  X_c = X ⊙ a_c
      2) Prior-guided joint attention: X_a = X_c ⊙ m

    Where:
      - X is (N, C, T, V)
      - a_c is (N, C, 1, 1)
      - m is a joint prior mask (V,) expanded/broadcast to (N, C, T, V)
    """

    def __init__(
        self,
        channels: int,
        num_joints: int = 17,
        reduction: int = 4,
        alpha: float = 2.0,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.alpha = float(alpha)

        # Channel attention (SE-like): global pooling over (T, V) then MLP.
        hidden = max(channels // reduction, 1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # (N,C,T,V) -> (N,C,1,1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def _expand_prior_mask(
        self,
        amputation_type: Union[str, Sequence[str]],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Convert amputation_type into a prior mask batch m (N, V).
        Each sample has its own mask if amputation_type is a list.
        """
        if isinstance(amputation_type, (list, tuple)):
            # Expected length: N (per sample). If not, fallback to first element.
            if len(amputation_type) != batch_size and len(amputation_type) > 0:
                amputation_type = [amputation_type[0]] * batch_size
            ms = [
                make_joint_prior_mask(t, self.num_joints, alpha=self.alpha, device=device, dtype=dtype)
                for t in amputation_type
            ]
            m = torch.stack(ms, dim=0)  # (N, V)
        else:
            m_vec = make_joint_prior_mask(
                amputation_type,
                self.num_joints,
                alpha=self.alpha,
                device=device,
                dtype=dtype,
            )  # (V,)
            m = m_vec.unsqueeze(0).expand(batch_size, -1)  # (N, V)
        return m

    def forward(
        self,
        x: torch.Tensor,
        amputation_type: Union[str, Sequence[str]] = "none",
        amputee_type: Optional[Union[str, Sequence[str]]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, V)
            amputation_type: 'left_limb'/'right_limb'/'both_limbs' (or dataset variants)
            amputee_type: backward-compatible alias (if provided, overrides amputation_type)
        Returns:
            x_a: (N, C, T, V)
        """
        if amputee_type is not None and (amputation_type is None or amputation_type == "none"):
            amputation_type = amputee_type

        N, C, T, V = x.shape
        assert V == self.num_joints, f"Expected V={self.num_joints}, got V={V}"

        # (1) Channel attention: X_c = X ⊙ a_c
        a_c = self.global_pool(x)  # (N, C, 1, 1)
        a_c = self.fc1(a_c)
        a_c = self.relu(a_c)
        a_c = self.fc2(a_c)
        a_c = self.sigmoid(a_c)  # (N, C, 1, 1)
        x_c = x * a_c  # broadcast along T,V

        # (2) Prior-guided joint attention: X_a = X_c ⊙ m
        m = self._expand_prior_mask(amputation_type, batch_size=N, device=x.device, dtype=x.dtype)  # (N,V)
        m_bc = m[:, None, None, :]  # (N,1,1,V) -> broadcast along C,T
        x_a = x_c * m_bc
        return x_a


class AAGCNBlock(nn.Module):
    """
    Structural + Actional Graph Convolution block.

    Spatial update follows the mandatory formulation:
      H^{(l+1)} = σ( W_st H^{(l)} Â_st + W_at H^{(l)} Â_at )

    - Structural branch: uses fixed adjacency Â_st from Graph (non-learnable).
    - Actional branch: uses an adaptive relation matrix Â_at produced by
      Q/K self-attention (learnable via θ, φ).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int],
        stride: int = 1,
        dropout: float = 0.0,
        residual: bool = True,
        use_amputation_attention: bool = True,
        attention_alpha: float = 2.0,
        reduction: int = 4,
        lambda_action: float = 0.5,
    ):
        super().__init__()
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1  # odd temporal kernel

        t_kernel, spatial_kernel = int(kernel_size[0]), int(kernel_size[1])
        padding = ((t_kernel - 1) // 2, 0)

        self.lambda_action = float(lambda_action)

        # W_st H Â_st : ConvTemporalGraphical applies the linear projection and
        # then mixes with fixed adjacency partitions Â_st (K,V,V).
        self.struct_gcn = ConvTemporalGraphical(in_channels, out_channels, spatial_kernel)

        # W_at H and Â_at computation from the *original* H^{(l)}.
        inter_channels = max(out_channels // 2, 1)
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.W_at = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # Temporal convolution after spatial fusion.
        # Keep it close to original ST-GCN: BN -> Conv -> BN -> Dropout (no extra ReLU here).
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                (t_kernel, 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        # Optional Amputation-aware attention after TCN.
        self.use_amputation_attention = use_amputation_attention
        if use_amputation_attention:
            self.alam = AmputeeLimbAttention(
                out_channels,
                num_joints=17,
                reduction=reduction,
                alpha=attention_alpha,
            )

        # Residual mapping.
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.sigma = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        A_st: torch.Tensor,
        amputation_type: Union[str, Sequence[str]] = "none",
        amputee_type: Optional[Union[str, Sequence[str]]] = None,
    ):
        if amputee_type is not None and (amputation_type is None or amputation_type == "none"):
            amputation_type = amputee_type

        # Input H^{(l)}: (N, C_in, T, V)
        # Residual branch.
        res = self.residual(x)

        # (Structural) W_st H Â_st
        x_struct, _ = self.struct_gcn(x, A_st)  # (N, C_out, T, V)

        # (Actional) Â_at from adaptive Q/K, then apply W_at H.
        N, _, T, V = x.shape
        q = self.theta(x)  # (N, d, T, V)
        k = self.phi(x)  # (N, d, T, V)
        v = self.W_at(x)  # (N, C_out, T, V)

        d = q.size(1)
        # q: (N*T, V, d), k: (N*T, d, V), so attn: (N*T, V, V)
        q = q.permute(0, 2, 3, 1).contiguous().view(N * T, V, d)
        k = k.permute(0, 2, 1, 3).contiguous().view(N * T, d, V)
        attn = torch.bmm(q, k) / math.sqrt(d + 1e-6)
        attn = F.softmax(attn, dim=-1)  # Â_at (adaptive relation)

        # Apply Â_at to values v (W_at H): (N*T, V, C_out)
        v = v.permute(0, 2, 3, 1).contiguous().view(N * T, V, -1)
        x_action = torch.bmm(attn, v)  # (N*T, V, C_out)
        x_action = x_action.view(N, T, V, -1).permute(0, 3, 1, 2).contiguous()  # (N,C_out,T,V)

        # Spatial fusion following the mandatory equation:
        # σ(W_st H Â_st + W_at H Â_at)
        x_spatial = x_struct + self.lambda_action * x_action
        x_spatial = self.sigma(x_spatial)

        # Temporal convolution + optional amputee-aware attention.
        x_out = self.tcn(x_spatial)
        if self.use_amputation_attention:
            x_out = self.alam(x_out, amputation_type=amputation_type, amputee_type=amputee_type)

        x_out = x_out + res
        return x_out, A_st


class AAGCN(nn.Module):
    """
    Amputee-Aware Graph Convolutional Network (AAGCN).

    Pipeline:
      1) (N,C,T,V,M) skeleton sequence -> ST-GCN normalization
      2) Multi-layer Structural+Actional graph conv backbone
      3) Amputation-aware joint/channel attention (ALAM) guided by prior mask m
      4) Global Average Pooling (GAP) -> Fully Connected (FC) to 5 classes
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_class: int = 5,
        graph_args=None,
        edge_importance_weighting: bool = False,
        use_amputee_attention: bool = True,
        attention_weight: float = 2.0,
        lambda_action: float = 0.5,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        # Enforce exact 5-class output as required by the paper.
        if num_class != 5:
            raise ValueError(f"AAGCN requires num_class=5, got num_class={num_class}")

        if graph_args is None:
            graph_args = {"layout": "alphapose", "strategy": "spatial"}

        self.graph = Graph(**graph_args)
        A_st = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A_st", A_st)  # fixed structural adjacency

        # edge_importance_weighting is intentionally ignored for strict "fixed Â_st".
        _ = edge_importance_weighting

        self.use_amputee_attention = use_amputee_attention
        self.attention_weight = float(attention_weight)
        self.lambda_action = float(lambda_action)
        self.dropout = float(dropout)

        spatial_kernel_size = self.A_st.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)

        # Data normalization (ST-GCN style).
        self.data_bn = nn.BatchNorm1d(in_channels * self.A_st.size(1))

        # Backbone blocks (10 layers as in the existing skeleton code).
        def make_block(cin, cout, s, first=False):
            return AAGCNBlock(
                cin,
                cout,
                kernel_size=kernel_size,
                stride=s,
                dropout=self.dropout,
                residual=not first,
                use_amputation_attention=self.use_amputee_attention,
                attention_alpha=self.attention_weight,
                lambda_action=self.lambda_action,
                **kwargs,
            )

        self.blocks = nn.ModuleList(
            [
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
            ]
        )

        # Global Average Pooling -> FC(256 -> 5)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 5)

    def _expand_amputation_type(
        self,
        amputation_type: Union[str, Sequence[str]],
        N: int,
        M: int,
    ) -> Union[str, List[str]]:
        """
        Expand sample-level amputation_type list (length N) to person-level (length N*M).
        If a scalar string is provided, replicate it.
        """
        if isinstance(amputation_type, (list, tuple)):
            if len(amputation_type) == N:
                expanded: List[str] = []
                for t in amputation_type:
                    expanded.extend([t] * M)
                return expanded
            if len(amputation_type) == N * M:
                return list(amputation_type)
            if len(amputation_type) > 0:
                return [amputation_type[0]] * (N * M)
            return ["none"] * (N * M)

        # scalar string
        return [str(amputation_type)] * (N * M)

    def forward(
        self,
        x: torch.Tensor,
        amputation_type: Union[str, Sequence[str]] = "none",
        amputee_type: Optional[Union[str, Sequence[str]]] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (N, C, T, V, M)
            amputation_type: 'left_limb'/'right_limb'/'both_limbs' (or dataset variants)
            amputee_type: backward-compatible alias from existing feeder/processor
        """
        if amputee_type is not None and (amputation_type is None or amputation_type == "none"):
            amputation_type = amputee_type

        N, C, T, V, M = x.size()

        # ST-GCN normalization: (N,C,T,V,M) -> (N*M,C,T,V)
        x = x.permute(0, 4, 3, 1, 2).contiguous()  # (N,M,V,C,T)
        x = x.view(N * M, V * C, T)  # (N*M, V*C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (N,M,C,T,V)
        x = x.view(N * M, C, T, V)

        # Expand amputation_type to match person-level batch (N*M,).
        current_type = self._expand_amputation_type(amputation_type, N, M)

        # Backbone: structural+actional fusion in each block, with ALAM guided by prior mask m.
        for blk in self.blocks:
            x, _ = blk(x, self.A_st, amputation_type=current_type)

        # Classification head: GAP then FC (exactly 5 classes).
        x = self.gap(x)  # (N*M, 256, 1, 1)
        x = x.squeeze(-1).squeeze(-1)  # (N*M, 256)
        x = x.view(N, M, -1).mean(dim=1)  # (N, 256)
        x = self.fc(x)  # (N, 5)
        return x


# Backward-compatible alias (optional): allow importing old name.
# This file defines the strict AAGCN; older code can still call AAGCN directly.
