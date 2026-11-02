import torch
import torch.nn as nn


class TextMSTFF(nn.Module):
    """
    Adaptive Multi-Strategy Text Feature Fusion

    Input:
        token_features:
            [B, L, D]

        eot_feature:
            [B, D]

    Output:
        fused text feature:
            [B, D]
    """

    def __init__(
        self,
        embed_dim=512,
        max_weights=(0.6, 0.3, 0.5),
    ):
        super().__init__()


        self.embed_dim = embed_dim


        # ==========================
        # Token Importance Learning
        # ==========================

        self.token_selector = nn.Sequential(
            nn.Linear(
                embed_dim,
                embed_dim // 2
            ),
            nn.ReLU(inplace=True),
            nn.Linear(
                embed_dim // 2,
                1
            )
        )


        # ==========================
        # Feature Projection
        # ==========================

        self.eot_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.mean_proj = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.att_proj = nn.Linear(
            embed_dim,
            embed_dim
        )


        # ==========================
        # Adaptive Fusion Weight
        #
        # FB-CLIP:
        # EOT:Mean:Attention
        # 1:0.5:0.5
        #
        # normalized:
        # 0.5:0.25:0.25
        #
        # ==========================

        init_weight = torch.tensor(
            [
                1.0,
                0.5,
                0.5
            ]
        )


        self.fusion_logits = nn.Parameter(
            torch.log(init_weight)
        )


        # 最大限制
        self.register_buffer(
            "max_weights",
            torch.tensor(max_weights)
        )


    def forward(
        self,
        token_features,
        eot_feature
    ):


        # --------------------------------
        # 1. EOT branch
        # --------------------------------

        eot_feature = self.eot_proj(
            eot_feature
        )


        # --------------------------------
        # 2. Mean pooling branch
        # --------------------------------

        mean_feature = token_features.mean(
            dim=1
        )


        mean_feature = self.mean_proj(
            mean_feature
        )


        # --------------------------------
        # 3. Attention pooling branch
        # --------------------------------

        token_score = self.token_selector(
            token_features
        )


        token_weight = torch.softmax(
            token_score,
            dim=1
        )


        att_feature = torch.sum(
            token_weight *
            token_features,
            dim=1
        )


        att_feature = self.att_proj(
            att_feature
        )


        # --------------------------------
        # 4. Adaptive fusion
        # --------------------------------

        weights = torch.softmax(
            self.fusion_logits,
            dim=0
        )


        # weight constraint

        weights = torch.minimum(
            weights,
            self.max_weights
        )


        # re-normalization

        weights = weights / weights.sum()



        fused_feature = (
            weights[0] * eot_feature
            +
            weights[1] * mean_feature
            +
            weights[2] * att_feature
        )


        return fused_feature, weights