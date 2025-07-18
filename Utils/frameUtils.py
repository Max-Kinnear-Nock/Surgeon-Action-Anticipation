import torch
from typing import List, Tuple

def extract_frames(
    labels: List[torch.Tensor],
    recognition_length: int,
    anticipation_length: int,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Splits the labels into recognition and anticipation targets by masking
    the last N frames of each sequence (anticipation targets).
    
    Args:
        labels (List[Tensor]): A list of tensors with shape [B * recognition_length, C].
        recognition_length (int): Number of frames used for recognition.
        anticipation_length (int): Number of anticipation target frames at end of each sequence.

    Returns:
        Tuple[List[Tensor], List[Tensor]]:
            labels_recog: Frames used for recognition [B * recognition_length, C]
            labels_anticp: Frames used for anticipation [B * anticipation_length, C]
    """
    if not labels or not isinstance(labels, list):
        raise ValueError("Expected a non-empty list of tensors for `labels`.")

    device = labels[0].device
    total_frames = labels[0].shape[1]

    if total_frames % (recognition_length + anticipation_length) != 0:
        raise ValueError("Label tensor size must be divisible by recognition_length.")

    batch_size = labels[0].shape[0]

    # Build a single sequence mask: keep first recognition_length, drop last anticipation_length
    mask = torch.tensor(
        [True] * recognition_length + [False] * anticipation_length,
        dtype=torch.bool,
        device=device
    )

    # Repeat the mask for each batch
    recog_mask = mask.repeat(batch_size)
    anticp_mask = ~recog_mask

    labels_recog = []
    labels_anticp = []

    for label in labels:
        flat_label = label.view(-1, label.shape[-1])  # [B*T, C]
        labels_recog.append(flat_label[recog_mask])   # [B*recog_len, C]
        labels_anticp.append(flat_label[anticp_mask]) # [B*ant_len, C]

    return labels_recog, labels_anticp
