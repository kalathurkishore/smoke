import torch
from torch import nn

from smoke.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..heads.heads import build_heads

from ..heads.vld_head import VLDHead
from ..loss.vld_loss import focal_loss, regression_loss
from ..utils.vld_target import generate_vld_targets


class KeypointDetector(nn.Module):
    '''
    Generalized structure for keypoint based object detector.
    main parts:
    - backbone
    - heads
    '''

    def __init__(self, cfg):
        super(KeypointDetector, self).__init__()

        self.backbone = build_backbone(cfg)
        self.heads = build_heads(cfg, self.backbone.out_channels)
        self.vld_head = VLDHead(in_channels=self.backbone.out_channels)

    def forward(self, images, targets=None):
        """
        Args:
            images:
            targets:

        Returns:

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.backbone(images.tensors)
        result, detector_losses = self.heads(features, targets)

        vld_feat = features[-1] if isinstance(features, (list, tuple)) else features
        vld_heatmap, vld_reg = self.vld_head(vld_feat)

        if self.training:
            losses = dict(detector_losses)

            output_size = (vld_feat.shape[2], vld_feat.shape[3]) # H, W
            gt_heatmaps = []
            gt_regs = []
            gt_masks = []
            for t in targets:
                bboxes = t.get_field("bboxes")
                valid_mask = t.get_field("reg_mask") == 1
                valid_bboxes = bboxes[valid_mask]
                hm, reg, mask = generate_vld_targets(valid_bboxes.cpu().numpy(), output_size, stride=1)
                gt_heatmaps.append(hm)
                gt_regs.append(reg)
                gt_masks.append(mask)

            gt_heatmaps = torch.stack(gt_heatmaps).to(vld_heatmap.device)
            gt_heatmaps = gt_heatmaps.unsqueeze(1)
            gt_regs = torch.stack(gt_regs).to(vld_reg.device)
            gt_masks = torch.stack(gt_masks).to(vld_reg.device)

            vld_loss_heatmap = focal_loss(vld_heatmap, gt_heatmaps)
            vld_loss_reg = regression_loss(vld_reg, gt_regs, gt_masks)
            
            losses["vld_hm_loss"] = vld_loss_heatmap
            losses["vld_reg_loss"] = 0.1 * vld_loss_reg

            return losses

        return {
            "vld_heatmap": vld_heatmap,
            "vld_regression": vld_reg,
            "smoke_result": result
        }