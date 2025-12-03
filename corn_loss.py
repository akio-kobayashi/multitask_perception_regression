import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def corn_loss(logits, target_binary):
    """
    logits: (B, K-1)
    target_binary: (B, K-1), CORN用バイナリラベル（1 or 0）
    """
    return F.binary_cross_entropy_with_logits(logits, target_binary, reduction='mean')
