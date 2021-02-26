import torch 
import torch.nn as nn 
import torch.nn.functional as F 

from typing import Optional, List, Tuple

def focal_loss(logits, targets, alpha: float, gamma: float, normalizer):
    """
    Focal loss = -(1 - pt)^gamma * log(pt) 
    Args:
        logits: [B, H, W, N]
        targets: [B, H, W, N]
        alpha: alpha * pos, (1 - alpha) * neg
        gamma: Modulating loss from hard and easy examples
        normalizer: Normalize total loss from all examples.
    Returns:
        loss: Normalized total loss
    """
    positive_label_mask = targets == 1.0
    cross_entropy = F.binary_cross_entropy_with_logits(logits, targets.to(logits.dtype), reduction='none')
    neg_logits = -1.0 * logits
    modulator = torch.exp(gamma * targets * neg_logits -  gamma * torch.log1p(torch.exp(neg_logits)))

    loss = modulator * cross_entropy
    weighted_loss = torch.where(positive_label_mask, alpha * loss, (1.0 - alpha) * loss)
    
    assert normalizer > 0
    return weighted_loss / normalizer 


def huber_loss(input, target, delta = 1, weights: Optional[torch.Tensor] = None, size_average = True):
    err = input - target
    abs_err = err.abs() 
    quadratic = torch.clamp(abs_err, max=delta) 
    linear = abs_err - quadratic
    loss = 0.5 * quadratic.pow(2) + delta * linear 
    if weights is not None: 
        loss *= weights 
    if size_average:
        return loss.mean() 
    else:
        return loss.sum()  
    
def _box_loss(box_outputs, box_targets, num_positives, delta: float = 0.1):
    """ Compute box regression loss"""

    normalizer = num_positives * 4
    mask = box_targets != 0.0
    box_loss = huber_loss(box_outputs, box_targets, weights = mask, delta = delta, size_average = False)
    return box_loss / normalizer 

def one_hot(lbl, numclasses):
    return lbl 

def loss_fn(
            cls_outputs: List[torch.Tensor],
            box_outputs: List[torch.Tensor],
            cls_targets: List[torch.Tensor],
            box_targets: List[torch.Tensor],
            num_positives: torch.Tensor,
            num_classes: int,
            alpha: float,
            gamma: float, 
            delta: float, 
            box_loss_weight: bool = False,
            ):
    """Computes total detection loss.
    Computes total detection loss including box and class loss from all levels.
    Args:
        cls_outputs: a List with values representing logits in [B, H, W, A] (A: num_anchors)
            at each feature level(index)
        box_outputs: a List with values representing box regression targets in [B, H, W, A * 4] 
            at each feature level 
        cls_targets: 
        box_targets: 
        num_positives: num positive groundtruth anchors 
    Returns:
        total_loss: 
        cls_loss:
        box_loss:

    """
    num_positives_sum = (num_positives.sum() + 1.0).float() 
    levels = len(cls_outputs)

    cls_losses = []
    box_losses = []
    for l in range(levels):
        cls_targets_at_level = cls_targets[l] 
        box_targets_at_level = box_targets[l]

        # Onehot encoding for classification labels 
        cls_targets_at_level_oh = one_hot(cls_targets_at_level, num_classes)
        B, H, W, _, _ = cls_targets_at_level_oh.shape 
        cls_targets_at_level_oh = cls_targets_at_level_oh.view(B, H, W, -1) 
        cls_outputs_at_level = cls_outputs[l].permute(0, 2, 3, 1).float() 

        cls_loss = focal_loss(
            cls_outputs_at_level, cls_targets_at_level_oh,
            alpha = alpha, gamma = gamma, normalizer = num_positives_sum
        )

        cls_loss = cls_loss.view(B, H, W, -1, num_classes)
        cls_loss = cls_loss * (cls_targets_at_level != -2).unsqueeze(-1)
        cls_losses.append(cls_loss.sum()) # MAYBE BUGHERE 
        
        box_losses.append(_box_loss(
            box_outputs[l].permute(0, 2, 3, 1).float(),
            box_targets_at_level, 
            num_positives_sum,
            delta = delta
        ))
class DetectionLoss(nn.Module):

    __constants__ = ['num_classes']

    def __init__(self, config):
        super(DetectionLoss, self).__init__() 
        self.config = config 
        self.num_classes = config.num_classes 
        
    def forward(
                self, 
                cls_outputs: List[torch.Tensor],
                box_outputs: List[torch.Tensor],
                cls_targets: List[torch.Tensor],
                box_targets: List[torch.Tensor],
                num_positives: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        l_fn = loss_fn 
        return l_fn(cls_outputs, box_outputs, cls_targets, box_targets, num_positives,
                )   