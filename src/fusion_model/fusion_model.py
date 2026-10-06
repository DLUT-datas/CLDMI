import torch
import torch.nn as nn


class FeatureGatedResidualFusion(nn.Module):
    """
    Feature-Gated Residual Fusion Module

    输入：
        h_g: CMPN graph embedding, shape = [B, dim_g]
        h_l: line-graph Transformer embedding, shape = [B, dim_l]

    输出：
        pred: prediction logits/regression output, shape = [B, out_dim]
        fusion_info: dict, 包含 gate 和 fusion embedding
    """

    def __init__(
        self,
        dim_g: int,
        dim_l: int,
        hidden_dim: int = 256,
        out_dim: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_residual: bool = True,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 1. modality projection
        self.proj_g = nn.Sequential(
            nn.Linear(dim_g, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            act,
            nn.Dropout(dropout),
        )

        self.proj_l = nn.Sequential(
            nn.Linear(dim_l, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            act,
            nn.Dropout(dropout),
        )

        # 2. cross gate
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        # 3. residual refinement
        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 4. prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

        # self.sigmoid = nn.Sigmoid()

    def forward(self, h_g, h_l):
        """
        h_g: graph embedding from CMPN
        h_l: line graph embedding from Transformer
        """

        z_g = self.proj_g(h_g)
        z_l = self.proj_l(h_l)

        gate_input = torch.cat([z_g, z_l], dim=-1)
        gate = self.gate_net(gate_input)

        # gated fusion
        h_fused = gate * z_g + (1.0 - gate) * z_l

        # residual refinement
        h_refined = self.fusion_ffn(h_fused)

        if self.use_residual:
            h_out = h_fused + h_refined
        else:
            h_out = h_refined

        pred = self.predictor(h_out)

        # pred = self.sigmoid(pred)

        fusion_info = {
            "gate": gate,
            "z_g": z_g,
            "z_l": z_l,
            "fusion_embedding": h_out
        }

        return pred, fusion_info
    


class FeatureGatedResidualFusion_ab_ConcatFusion(nn.Module):
    """
    输入：
        h_g: CMPN graph embedding, shape = [B, dim_g]
        h_l: line-graph Transformer embedding, shape = [B, dim_l]

    输出：
        pred: prediction logits/regression output, shape = [B, out_dim]
        fusion_info: dict, 包含 gate 和 fusion embedding
    """

    def __init__(
        self,
        dim_g: int,
        dim_l: int,
        hidden_dim: int = 256,
        out_dim: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_residual: bool = True,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 1. modality projection
        self.proj = nn.Sequential(
            nn.Linear(dim_g+dim_l, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            act,
            nn.Dropout(dropout),
        )


        self.predictor = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, h_g, h_l):
        """
        h_g: graph embedding from CMPN
        h_l: line graph embedding from Transformer
        """


        h_fused = torch.cat([z_g, z_l], dim=-1)
        z = self.proj(h_concated)
        pred = self.predictor(z)

        fusion_info = {"z_g": z_g, "z_l": z_l, "fusion_embedding": h_fused}

        return pred, fusion_info
    


class FeatureGatedResidualFusion_ab_avgFusion(nn.Module):
    """
    输入：
        h_g: CMPN graph embedding, shape = [B, dim_g]
        h_l: line-graph Transformer embedding, shape = [B, dim_l]

    输出：
        pred: prediction logits/regression output, shape = [B, out_dim]
        fusion_info: dict, 包含 gate 和 fusion embedding
    """

    def __init__(
        self,
        dim_g: int,
        dim_l: int,
        hidden_dim: int = 256,
        out_dim: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_residual: bool = True,
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.use_residual = use_residual
        self.use_layernorm = use_layernorm

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 1. modality projection
        self.proj_g = nn.Sequential(
            nn.Linear(dim_g, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            act,
            nn.Dropout(dropout),
        )

        self.proj_l = nn.Sequential(
            nn.Linear(dim_l, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            act,
            nn.Dropout(dropout),
        )

        # # 2. cross gate
        # self.gate_net = nn.Sequential(
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     act,
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.Sigmoid()
        # )

        # 3. residual refinement
        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity(),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 4. prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, h_g, h_l):
        """
        h_g: graph embedding from CMPN
        h_l: line graph embedding from Transformer
        """

        z_g = self.proj_g(h_g)
        z_l = self.proj_l(h_l)

        # gate_input = torch.cat([z_g, z_l], dim=-1)
        # gate = self.gate_net(gate_input)
        # # gated fusion
        # h_fused = gate * z_g + (1.0 - gate) * z_l

        # fixed equal weighting
        h_fused = 0.5 * z_g + 0.5 * z_l

        # residual refinement
        h_refined = self.fusion_ffn(h_fused)

        if self.use_residual:
            h_out = h_fused + h_refined
        else:
            h_out = h_refined

        pred = self.predictor(h_out)

        fusion_info = {"z_g": z_g, "z_l": z_l, "fusion_embedding": h_out}

        return pred, fusion_info
    

class Attention_ab_Fusion(nn.Module):
    """
    Attention-based Representation Fusion (Ablation)

    Replace feature-wise gate with view-level attention weighting.

    Inputs:
        h_g: molecular graph embedding
        h_l: knowledge-enhanced line graph embedding

    Output:
        pred
    """

    def __init__(
        self,
        dim_g: int,
        dim_l: int,
        hidden_dim: int = 256,
        out_dim: int = 1,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "gelu":
            act = nn.GELU()
        else:
            raise ValueError(
                f"Unsupported activation: {activation}"
            )


        # 1. projection (same as proposed model)
        self.proj_g = nn.Sequential(
            nn.Linear(dim_g, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act,
            nn.Dropout(dropout),
        )

        self.proj_l = nn.Sequential(
            nn.Linear(dim_l, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act,
            nn.Dropout(dropout),
        )


        # 2. attention weight generation
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )


        # 3. residual refinement
        self.fusion_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )


        # 4. prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )


    def forward(self, h_g, h_l):

        # representation projection
        z_g = self.proj_g(h_g)
        z_l = self.proj_l(h_l)


        # attention weights
        att_input = torch.cat([z_g, z_l], dim=-1)

        alpha = torch.softmax(self.attention(att_input), dim=-1)

        alpha_g = alpha[:, 0].unsqueeze(-1)
        alpha_l = alpha[:, 1].unsqueeze(-1)

        # attention fusion
        h_fused = (alpha_g * z_g + alpha_l * z_l)


        # refinement
        h_refined = self.fusion_ffn(h_fused)
        
        h_out = h_fused + h_refined

        pred = self.predictor(h_out)

        fusion_info = {"attention": alpha, "z_g": z_g, "z_l": z_l, "fusion_embedding": h_out}

        return pred, fusion_info


class MultimodalModel(nn.Module):
    """
    Master Model for multimodal molecular representation learning

    Modules:
    - CMPNN encoder (graph view)
    - Line-Graph Transformer (line graph view)
    - Fusion module (cross-modal interaction)
    """

    def __init__(
        self,
        graph_module,
        linegraph_module,
        fusion_module
    ):
        super(MultimodalModel, self).__init__()

        self.graph_module = graph_module
        self.linegraph_module = linegraph_module
        self.fusion_module = fusion_module

    # =========================================================
    # forward (standard multimodal inference)
    # =========================================================
    def forward(self, smiles, g, ecfp, md):
        """
        Args:
            smiles: graph input for CMPNN
            g, ecfp, md: inputs for Line-Graph Transformer
        """

        pred_cmpn, h_cmpn = self.graph_module(smiles)  # (B, d1)

        # =========================
        # 2. Line-Graph Transformer embedding
        # =========================
        pred_lgt, h_lgt = self.linegraph_module.forward_tune(g, ecfp, md)    # (B, d2)

        # =========================
        # 3. fusion
        # =========================
        pred_fusion, fusion_info = self.fusion_module(h_cmpn, h_lgt)

        return pred_fusion, fusion_info, pred_cmpn, pred_lgt

    # =========================================================
    # helper: freeze / unfreeze utilities
    # =========================================================
    def freeze_cmpnn(self):
        for p in self.cmpnn.parameters():
            p.requires_grad = False

    def freeze_lg(self):
        for p in self.lg.parameters():
            p.requires_grad = False

    def freeze_fusion(self):
        for p in self.fusion.parameters():
            p.requires_grad = False

    def unfreeze_cmpnn(self):
        for p in self.cmpnn.parameters():
            p.requires_grad = True

    def unfreeze_lg(self):
        for p in self.lg.parameters():
            p.requires_grad = True

    def unfreeze_fusion(self):
        for p in self.fusion.parameters():
            p.requires_grad = True
