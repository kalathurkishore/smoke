import torch
import torch.nn as nn
import torch.nn.functional as F

def focal_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    pos_loss = -torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = -torch.log(1 - pred) * torch.pow(pred, 2) * torch.pow(1 - gt, 4) * neg_inds

    num_pos = pos_inds.sum()

    loss = pos_loss.sum() + neg_loss.sum()

    if num_pos > 0:
        loss = loss / num_pos

    return loss

def regression_loss(pred, gt, mask):
    return F.l1_loss(pred * mask, gt * mask)
