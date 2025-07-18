import math
import os
from typing import Any, Optional, Union, Tuple, List

import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights

from models.transformer import Transformer, TemporalTransformer
from models.future_prediction import AVTh
from models.MC_Loss import supervisor
from einops import rearrange, repeat

# Enable CUDA blocking for easier debugging
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


class MLP(nn.Module):
    """ Multi-layer Perceptron with ReLU and dropout. """
    def __init__(self, in_features: int, out_features: int, nlayers: int, dropout_p: float = 0.3, **kwargs: Any):
        super().__init__()
        layers = []
        for _ in range(nlayers - 1):
            layers += [nn.Linear(in_features, in_features, **kwargs), nn.ReLU(), nn.Dropout(dropout_p)]
        layers.append(nn.Linear(in_features, out_features))
        self.cls = nn.Sequential(*layers)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.cls(inp)


class SinusoidalPositionalEncoding(nn.Module):
    """ Implements sinusoidal positional encoding from the Transformer paper. """
    def __init__(self, length: int, embedding_dim: int):
        super().__init__()
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe = torch.zeros(length, embedding_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('positional_encoding', pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding[positions]


class McDecoder(nn.Module):
    def __init__(
        self,
        num_classes: Optional[List[int]] = None,
        recognition_length: int = 10,
        anticipation_length: int = 1,
        batch_size: int = 110,
        input_dim: int = 2048,
        inter_dim: int = 420,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout_early: float = 0.1,
        dropout_late: float = 0.3,
        model_size_ratio: float = 1,
        mc_instrument_extractor: int = 2040,
        mc_verb_extractor: int = 2000,
        target_shape: int = 2040,
        patch_size: int = 256,
        number_of_patches: int = 16,
        classifier_dim: int = 2040
    ):
        super().__init__()

        num_classes = num_classes or [6, 10, 15, 100]

        # Preprocessing parameters
        self.device = "cuda"
        self.recognition_length = recognition_length
        self.anticipation_length = anticipation_length
        self.batch_size = batch_size
        self.model_size_ratio = model_size_ratio

        self.input_dim = input_dim
        self.target_shape = target_shape // model_size_ratio
        self.mc_instrument_extractor = mc_instrument_extractor // model_size_ratio
        self.mc_verb_extractor = mc_verb_extractor // model_size_ratio
        self.patch_size = patch_size // (model_size_ratio * 2)
        self.number_of_patches = number_of_patches
        self.classifier_dim = classifier_dim // (model_size_ratio * 2)

        # Feature extractor
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.share = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            nn.Conv2d(2048, input_dim, kernel_size=1)
        )

        self.avg_p = resnet.avgpool

        # Task-specific projections
        self.features2fixed_i = nn.Conv2d(input_dim, self.mc_instrument_extractor, kernel_size=1)
        self.features2fixed_v = nn.Conv2d(input_dim, self.mc_verb_extractor, kernel_size=1)
        self.linear_layer_i = nn.Linear(self.mc_instrument_extractor, 1024 // model_size_ratio)
        self.linear_layer_v = nn.Linear(self.mc_verb_extractor, 1024 // model_size_ratio)
        self.norm_xi = nn.LayerNorm(1024 // model_size_ratio)
        self.norm_xv = nn.LayerNorm(1024 // model_size_ratio)

        # Class counts for MC loss
        self.cnum_i = 340 // model_size_ratio
        self.cnum_v = 200 // model_size_ratio
        self.cnum_t = 136 // model_size_ratio
        self.cnum_ivt = 20 // model_size_ratio

        # Transformers
        self.spatial_transformer = Transformer(
            dim=self.patch_size, depth=6, heads=num_heads, dim_head=64, mlp_dim=2048, dropout=dropout_early)
        self.temporal_transformer = TemporalTransformer(
            dim=self.patch_size, depth=6, heads=num_heads, dim_head=64, mlp_dim=2048, dropout=dropout_early)

        self.cls_space_tokens = nn.Parameter(torch.randn(1, 1, self.patch_size))
        self.cls_temporal_tokens = nn.Parameter(torch.randn(1, 1, self.patch_size))

        # Positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(number_of_patches, self.patch_size)
        positions = torch.arange(number_of_patches).unsqueeze(0).repeat(recognition_length, 1)
        self.register_buffer("sinusoidal_positional_embeddings", self.pos_encoding(positions))

        # Future prediction decoder
        self.future_predictor = AVTh(
            in_features=self.patch_size,
            output_len=anticipation_length,
            output_len_eval=1,
            avg_last_n=1,
            inter_dim=inter_dim,
            future_pred_loss=None,
            return_past_too=False,
            drop_last_n=0,
            quantize_before_rollout=False,
            freeze_encoder_decoder=False,
            _recursive_=False
        )

        # Final MLP heads
        self.fc_h1 = nn.Linear(self.patch_size, self.classifier_dim)
        self.fc_h2 = nn.Linear(self.patch_size, self.classifier_dim)
        init.xavier_uniform_(self.fc_h1.weight)
        init.xavier_uniform_(self.fc_h2.weight)

        self.classifier_1 = MLP(self.classifier_dim, num_classes[0], 3, dropout_late)
        self.classifier_2 = MLP(self.classifier_dim, num_classes[1], 3, dropout_late)
        self.classifier_3 = MLP(self.classifier_dim, num_classes[2], 3, dropout_late)
        self.classifier_4 = MLP(self.classifier_dim, num_classes[3], 3, dropout_late)

    def forward(
        self,
        frames: torch.Tensor,
        labels_recog: List[torch.Tensor],
        labels_anticp: List[torch.Tensor],
        class_loss_weights: Optional[dict] = None
    ) -> Union[
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[float]],
        Tuple[List[torch.Tensor], List[torch.Tensor]]
    ]:
        if class_loss_weights is None:
            class_loss_weights = {}

        labels_1, labels_2, labels_3, labels_4 = labels_recog

        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W)

        # Feature extraction
        x = self.share(frames.to(memory_format=torch.channels_last))
        x_i = self.features2fixed_i(x)
        x_v = self.features2fixed_v(x)

        # MC Loss
        MC_loss_1 = supervisor(x_i, labels_1, height=7, cnum=self.cnum_i)
        MC_loss_2 = supervisor(x_v, labels_2, height=7, cnum=self.cnum_v)
        MC_loss_3 = supervisor(x_i, labels_3, height=7, cnum=self.cnum_t)
        MC_loss_4 = supervisor(x_v, labels_4, height=7, cnum=self.cnum_ivt)

        # Average pool and reshape
        x_i = self.avg_p(x_i).view(-1, self.recognition_length, self.mc_instrument_extractor)
        x_v = self.avg_p(x_v).view(-1, self.recognition_length, self.mc_verb_extractor)

        # Project and normalize
        x_i = self.norm_xi(self.linear_layer_i(x_i))
        x_v = self.norm_xv(self.linear_layer_v(x_v))

        all_features = torch.cat([x_i, x_v], dim=2)
        all_features = all_features.view(-1, self.recognition_length, self.number_of_patches, self.patch_size)
        all_features = all_features + self.sinusoidal_positional_embeddings.to(all_features.device)

        # Spatial transformer
        spatial_cls_tokens = repeat(self.cls_space_tokens, '1 1 d -> b t 1 d', b=all_features.size(0), t=self.recognition_length)
        all_features = torch.cat((spatial_cls_tokens, all_features), dim=2)
        all_features = rearrange(all_features, 'b t n d -> (b t) n d')

        spatial_output = self.spatial_transformer(all_features)
        spatial_output = rearrange(spatial_output, '(b t) n d -> b t n d', b=x_i.size(0))[:, :, 0]

        # Temporal transformer
        temporal_cls_tokens = repeat(self.cls_temporal_tokens, '1 1 d -> b 1 d', b=x_i.size(0))
        spatial_output = torch.cat((temporal_cls_tokens, spatial_output), dim=1)
        temporal_feat = self.temporal_transformer(spatial_output)

        # Future prediction
        _, feats_future, _, _ = self.future_predictor(temporal_feat, (x_i.size(0), self.anticipation_length, self.target_shape))

        # Final heads
        y = self.fc_h1(feats_future.view(-1, self.patch_size))
        z = self.fc_h2(temporal_feat[:, :T, :].contiguous()).view(-1, self.classifier_dim)

        # Classifier outputs
        y_anticp = [clf(y) for clf in [self.classifier_1, self.classifier_2, self.classifier_3, self.classifier_4]]
        y_recog = [clf(z) for clf in [self.classifier_1, self.classifier_2, self.classifier_3, self.classifier_4]]

        # Losses
        def compute_loss(loss_fn, pred, label):
            return loss_fn(pred, label.float())

        bce_kwargs = lambda key: {'pos_weight': class_loss_weights[key].to(self.device)} if key in class_loss_weights else {}

        loss_fns = [
            nn.BCEWithLogitsLoss(**bce_kwargs('i')),
            nn.BCEWithLogitsLoss(**bce_kwargs('v')),
            nn.BCEWithLogitsLoss(**bce_kwargs('t')),
            nn.BCEWithLogitsLoss(**bce_kwargs('ivt'))
        ]

        losses_anticp = [compute_loss(fn, pred, label) for fn, pred, label in zip(loss_fns, y_anticp, labels_anticp)]
        losses_recog = [compute_loss(fn, pred, label) for fn, pred, label in zip(loss_fns, y_recog, labels_recog)]

        return y_anticp, y_recog, losses_anticp, losses_recog, [MC_loss_1, MC_loss_2, MC_loss_3, MC_loss_4]