import numpy as np
import torch
from .heatmap import draw_gaussian, gaussian_radius

def generate_vld_targets(bboxes, output_size, stride=4):
    H, W = output_size

    heatmap = np.zeros((H, W), dtype=np.float32)
    reg = np.zeros((4, H, W), dtype=np.float32)
    mask = np.zeros((1, H, W), dtype=np.float32)

    for bbox in bboxes:
        # Expected bbox format: [x1, y1, x2, y2]
        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) / 2 / stride
        cy = (y1 + y2) / 2 / stride

        w = (x2 - x1) / stride
        h = (y2 - y1) / stride

        cx_int = int(cx)
        cy_int = int(cy)

        # adaptive gaussian radius
        # Ensure dimensions are valid for the bounding box
        if h > 0 and w > 0:
            radius = gaussian_radius((h, w))
            radius = max(0, int(radius))

            heatmap = draw_gaussian(heatmap, (cx_int, cy_int), radius)

            reg[0, cy_int, cx_int] = cx - cx_int
            reg[1, cy_int, cx_int] = cy - cy_int
            reg[2, cy_int, cx_int] = np.log(h)
            reg[3, cy_int, cx_int] = np.log(w)

            mask[0, cy_int, cx_int] = 1

    return (
        torch.from_numpy(heatmap),
        torch.from_numpy(reg),
        torch.from_numpy(mask),
    )
