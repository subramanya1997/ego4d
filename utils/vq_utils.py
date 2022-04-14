import cv2
import torch
import torch.nn as nn
import numpy as np
import os
from typing import Sequence, Union

def extract_window_with_context(
    image: torch.Tensor,
    bbox: Sequence[Union[int, float]],
    p: int = 16,
    size: int = 256,
    pad_value: int = 0,
) -> torch.Tensor:
    """
    Extracts region from a bounding box in the image with some context padding.
    Arguments:
        image - (1, c, h, w) Tensor
        bbox - bounding box specifying (x1, y1, x2, y2)
        p - number of pixels of context to include around window
        size - final size of the window
        pad_value - value of pixels padded
    """
    H, W = image.shape[2:]
    bbox = tuple([int(x) for x in bbox])
    x1, y1, x2, y2 = bbox
    x1 = max(x1 - p, 0)
    y1 = max(y1 - p, 0)
    x2 = min(x2 + p, W)
    y2 = min(y2 + p, H)
    window = image[:, :, y1:y2, x1:x2]
    H, W = window.shape[2:]
    # Zero pad and resize
    left_pad = 0
    right_pad = 0
    top_pad = 0
    bot_pad = 0
    if H > W:
        left_pad = (H - W) // 2
        right_pad = (H - W) - left_pad
    elif H < W:
        top_pad = (W - H) // 2
        bot_pad = (W - H) - top_pad
    if H != W:
        window = nn.functional.pad(
            window, (left_pad, right_pad, top_pad, bot_pad), value=pad_value
        )
    window = nn.functional.interpolate(
        window, size=size, mode="bilinear", align_corners=False
    )

    return window

def perform_retrieval(image_path, visual_crop):
    if not os.path.exists(image_path):
        return
    img = cv2.imread(image_path)
    reference = cv2.imread(image_path)
    owidth, oheight = visual_crop["original_width"], visual_crop["original_height"]
    if (reference.shape[0] != oheight) or (reference.shape[1] != owidth):
        reference = cv2.resize(reference, (owidth, oheight))
    reference = torch.as_tensor(np.ascontiguousarray(reference.transpose(2, 0, 1)))
    reference = reference.unsqueeze(0).float()
    ref_bbox = (
        visual_crop["x"],
        visual_crop["y"],
        visual_crop["x"] + visual_crop["width"],
        visual_crop["y"] + visual_crop["height"],
    )
    reference = extract_window_with_context(
            reference,
            ref_bbox,
            p=16,
            size=256,
            pad_value=125,
        )
    return img, reference.squeeze(0).byte()
