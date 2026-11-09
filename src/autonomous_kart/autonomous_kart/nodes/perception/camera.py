"""One camera frame -> kart centres in the body frame, with no ROS.

https://github.com/EVC-Purdue/A27/issues/30
"""

from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np

from autonomous_kart.nodes.perception.ranging import CameraModel, RangingParams, range_boxes


def camera_objects(
    detector: Callable,
    camera: CameraModel,
    ranging: RangingParams,
    roi_frac: Sequence[float],
    bgr: np.ndarray,
    max_range_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xy_m (M, 2), cov_m2 (M, 2, 2), boxes_px (K, 4)) for one frame.

    The detector only sees the ROI (our own nose is outside it); its boxes are
    moved back to full-frame pixels before ranging. The ROI bottom acts as the
    image edge, so a box cut off there is treated as truncated rather than
    trusted to show the kart's ground contact.
    """
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = (round(f * n) for f, n in zip(roi_frac, (w, h, w, h), strict=True))
    det = detector(bgr[y1:y2, x1:x2])
    boxes = np.asarray(det.boxes, dtype=np.float64).reshape(-1, 4) + (x1, y1, x1, y1)
    if not len(boxes):
        return np.zeros((0, 2)), np.zeros((0, 2, 2)), boxes
    cam = replace(camera.scaled(w, h), height_px=y2)
    rb = range_boxes(cam, boxes, ranging)
    keep = rb.valid & (rb.range_m <= max_range_m)
    return rb.xy[keep], rb.cov[keep], boxes
