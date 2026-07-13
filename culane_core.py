"""
CULane core utilities — developed and verified against the real local subset.
Covers: .lines.txt parsing, scenario-split parsing (driver_37 filter),
classical CV lane detector, and the official-style geometric F1 metric.

No torch dependency here (runs anywhere). The Colab notebook imports the same logic.
"""
import os, glob
import numpy as np
import cv2

IMG_W, IMG_H = 1640, 590          # CULane native resolution
EVAL_W = 30                       # official CULane line width for IoU matching
IOU_THRESH = 0.5                  # official matching threshold


# --------------------------------------------------------------------------
# 1. Annotation (.lines.txt) parsing
# --------------------------------------------------------------------------
def read_lines_txt(path):
    """Return list of lanes; each lane is an (N,2) float array of (x,y) points."""
    lanes = []
    if not os.path.exists(path):
        return lanes
    with open(path) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 4:        # need >=2 points
                continue
            xs = np.array(vals[0::2], dtype=np.float32)
            ys = np.array(vals[1::2], dtype=np.float32)
            pts = np.stack([xs, ys], axis=1)
            lanes.append(pts)
    return lanes


# --------------------------------------------------------------------------
# 2. Scenario split parsing (filter to frames that exist on disk)
# --------------------------------------------------------------------------
SCENARIO_FILES = {
    "normal":  "list/test_split/test0_normal.txt",
    "crowded": "list/test_split/test1_crowd.txt",
    "noline":  "list/test_split/test4_noline.txt",
    "night":   "list/test_split/test8_night.txt",
}

def load_scenario(root, scenario, only_driver="driver_37_30frame"):
    """Return list of frame-relative paths for a scenario, filtered to frames on disk."""
    fp = os.path.join(root, SCENARIO_FILES[scenario])
    out = []
    with open(fp) as f:
        for line in f:
            rel = line.strip().lstrip("/")
            if only_driver and only_driver not in rel:
                continue
            if os.path.exists(os.path.join(root, rel)):
                out.append(rel)
    return out


def load_split(root, split_file):
    """Generic list-file loader (train_split.txt / val_split.txt). Returns img rel-paths."""
    out = []
    with open(os.path.join(root, split_file)) as f:
        for line in f:
            rel = line.strip().split()[0].lstrip("/")
            out.append(rel)
    return out


def mask_path_for(root, img_rel, train=True):
    """Map an image rel-path to its seg-mask path."""
    sub = "laneseg_label_w16" if train else "laneseg_label_w16_test"
    return os.path.join(root, sub, img_rel.replace(".jpg", ".png"))


# --------------------------------------------------------------------------
# 3. Geometric F1 metric (official CULane style)
# --------------------------------------------------------------------------
def _draw_lane(pts, w=IMG_W, h=IMG_H, width=EVAL_W):
    """Rasterize one lane polyline onto a binary canvas with given thickness."""
    canvas = np.zeros((h, w), dtype=np.uint8)
    p = np.round(pts).astype(np.int32)
    for i in range(len(p) - 1):
        cv2.line(canvas, tuple(p[i]), tuple(p[i + 1]), 1, width)
    return canvas


def _lane_iou(a_pts, b_pts, w=IMG_W, h=IMG_H, width=EVAL_W):
    A = _draw_lane(a_pts, w, h, width).astype(bool)
    B = _draw_lane(b_pts, w, h, width).astype(bool)
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return inter / union if union > 0 else 0.0


def match_frame(pred_lanes, gt_lanes, iou_thresh=IOU_THRESH):
    """Greedy IoU matching for one frame. Returns (tp, fp, fn)."""
    if len(gt_lanes) == 0:
        return 0, len(pred_lanes), 0
    if len(pred_lanes) == 0:
        return 0, 0, len(gt_lanes)
    iou = np.zeros((len(pred_lanes), len(gt_lanes)))
    for i, p in enumerate(pred_lanes):
        for j, g in enumerate(gt_lanes):
            iou[i, j] = _lane_iou(p, g)
    matched_gt, matched_pred, tp = set(), set(), 0
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for i, j in order:
        if iou[i, j] < iou_thresh:
            break
        if i in matched_pred or j in matched_gt:
            continue
        matched_pred.add(i); matched_gt.add(j); tp += 1
    fp = len(pred_lanes) - tp
    fn = len(gt_lanes) - tp
    return tp, fp, fn


def lane_lateral_distance(pred_lanes, gt_lanes, iou_thresh=IOU_THRESH,
                          y_eval=np.arange(300, IMG_H, 20)):
    """Geometric (not pixel-overlap) metric: for each IoU-matched pred-GT pair,
    mean absolute lateral (x) distance in pixels at fixed y rows. Captures
    smoothing / gap-filling differences that the lenient F1@IoU misses.
    Returns (mean_px_distance, n_matched_lanes)."""
    if not pred_lanes or not gt_lanes:
        return float("nan"), 0
    iou = np.zeros((len(pred_lanes), len(gt_lanes)))
    for i, p in enumerate(pred_lanes):
        for j, g in enumerate(gt_lanes):
            iou[i, j] = _lane_iou(p, g)
    dists, matched_p, matched_g = [], set(), set()
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    for i, j in order:
        if iou[i, j] < iou_thresh:
            break
        if i in matched_p or j in matched_g:
            continue
        matched_p.add(i); matched_g.add(j)
        p, g = pred_lanes[i], gt_lanes[j]
        # np.interp needs ascending y -> sort both by y
        p = p[np.argsort(p[:, 1])]; g = g[np.argsort(g[:, 1])]
        # interpolate x at common y rows, average |dx|
        d = []
        for y in y_eval:
            px = np.interp(y, p[:, 1], p[:, 0], left=np.nan, right=np.nan)
            gx = np.interp(y, g[:, 1], g[:, 0], left=np.nan, right=np.nan)
            if not (np.isnan(px) or np.isnan(gx)):
                d.append(abs(px - gx))
        if d:
            dists.append(np.mean(d))
    if not dists:
        return float("nan"), 0
    return float(np.mean(dists)), len(dists)


def prf1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


# --------------------------------------------------------------------------
# 4. Classical CV lane detector (training-free, "curve-based" baseline)
# --------------------------------------------------------------------------
# Fixed bird's-eye homography tuned for CULane dashcam geometry (1640x590).
_BEV_SRC = np.float32([[150, 580], [1490, 580], [1010, 345], [630, 345]])
_BEV_DST = np.float32([[300, 590], [1340, 590], [1340, 0], [300, 0]])
_BEV_M = cv2.getPerspectiveTransform(_BEV_SRC, _BEV_DST)
_BEV_Minv = cv2.getPerspectiveTransform(_BEV_DST, _BEV_SRC)


def _lane_binary(img_bgr):
    """White/yellow color + Sobel-x gradient -> binary lane-pixel image."""
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    white = cv2.inRange(hls, (0, 165, 0), (255, 255, 255))
    yellow = cv2.inRange(hls, (15, 60, 60), (45, 255, 255))
    sx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    sx = np.uint8(255 * sx / (sx.max() + 1e-6))
    grad = (sx > 45).astype(np.uint8) * 255
    return cv2.bitwise_or(cv2.bitwise_or(white, yellow), grad)


def detect_lanes_bev(img_bgr, n_windows=10, margin=90, minpix=50, sample_step=10):
    """
    Classical bird's-eye-view pipeline (training-free):
      lane-pixel binary -> perspective warp to top-down -> column histogram
      peaks -> sliding-window tracking -> 2nd-order poly fit in BEV ->
      sample points -> inverse-warp back to image coordinates.
    Returns list of lanes (each (N,2) array in original image coords).
    """
    h, w = img_bgr.shape[:2]
    binary = _lane_binary(img_bgr)
    bev = cv2.warpPerspective(binary, _BEV_M, (w, h))

    hist = bev[h // 2:, :].sum(axis=0)
    if hist.max() == 0:
        return []
    peaks, h2 = [], hist.copy()
    for _ in range(4):
        c = int(np.argmax(h2))
        if h2[c] < hist.max() * 0.25:
            break
        peaks.append(c)
        h2[max(0, c - 160):min(w, c + 160)] = 0
    peaks.sort()

    win_h = h // n_windows
    lanes = []
    for base in peaks:
        cur = base
        wx, wy = [], []
        for win in range(n_windows):
            y_hi = h - win * win_h
            y_lo = y_hi - win_h
            x_lo, x_hi = max(0, cur - margin), min(w, cur + margin)
            ys, xs = np.where(bev[y_lo:y_hi, x_lo:x_hi] > 0)
            if len(xs) > minpix:
                cur = int(xs.mean()) + x_lo
                wx.append(cur); wy.append((y_lo + y_hi) // 2)
        if len(wx) < 3:
            continue
        wy, wx = np.array(wy), np.array(wx)
        try:
            coef = np.polyfit(wy, wx, 2)
        except Exception:
            continue
        ploty = np.arange(0, h, sample_step)
        plotx = np.polyval(coef, ploty)
        bev_pts = np.stack([plotx, ploty], axis=1).astype(np.float32)[None]
        img_pts = cv2.perspectiveTransform(bev_pts, _BEV_Minv)[0]
        img_pts = img_pts[(img_pts[:, 0] >= 0) & (img_pts[:, 0] < w) &
                          (img_pts[:, 1] >= 0) & (img_pts[:, 1] < h)]
        if len(img_pts) >= 2:
            lanes.append(img_pts.astype(np.float32))
    return lanes


def _roi_trapezoid(shape, y_top):
    """Binary trapezoid mask focusing on the road region (wide bottom, narrow top)."""
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    poly = np.array([[
        (int(0.02 * w), h), (int(0.40 * w), y_top),
        (int(0.60 * w), y_top), (int(0.98 * w), h)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, poly, 1)
    return mask


def detect_lanes_classical(img_bgr, n_windows=12, margin=80, minpix=40,
                           y_top=300, sample_step=10):
    """
    Classical pipeline: white/yellow color threshold + Sobel-x gradient,
    restricted to a trapezoidal road ROI -> histogram peak seeding ->
    sliding-window tracking -> 2nd-order polynomial fit per lane.
    Returns list of lanes (each (N,2) array of (x,y) points in original coords).
    Deliberately hand-crafted & training-free; expected to degrade on
    crowded / night / no-line scenes (that is the error-analysis story).
    """
    h, w = img_bgr.shape[:2]
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # white (high L) + yellow color masks
    white = cv2.inRange(hls, (0, 170, 0), (255, 255, 255))
    yellow = cv2.inRange(hls, (15, 60, 60), (45, 255, 255))
    color = cv2.bitwise_or(white, yellow)
    # Sobel-x: vertical-ish bright edges (lane markings), thresholded
    sx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    sx = np.uint8(255 * sx / (sx.max() + 1e-6))
    grad = (sx > 40).astype(np.uint8) * 255
    binary = cv2.bitwise_or(color, grad)

    # restrict to trapezoidal road ROI (kills buildings / side clutter)
    roi = cv2.bitwise_and(binary, binary, mask=_roi_trapezoid((h, w), y_top))

    # histogram peaks on bottom third -> seed columns
    hist = roi[2 * h // 3:, :].sum(axis=0)
    if hist.max() == 0:
        return []
    peaks = []
    h2 = hist.copy()
    for _ in range(4):
        c = int(np.argmax(h2))
        if h2[c] < hist.max() * 0.25:
            break
        peaks.append(c)
        lo, hi = max(0, c - 140), min(w, c + 140)
        h2[lo:hi] = 0
    peaks.sort()

    lanes = []
    win_h = (h - y_top) // n_windows
    for base in peaks:
        cur = base
        xs, ys = [], []
        for win in range(n_windows):
            y_hi = h - win * win_h
            y_lo = y_hi - win_h
            x_lo, x_hi = max(0, cur - margin), min(w, cur + margin)
            window = roi[y_lo:y_hi, x_lo:x_hi]
            pix = np.where(window > 0)
            if len(pix[1]) > minpix:
                cur = int(pix[1].mean()) + x_lo
            xs.append(cur); ys.append((y_lo + y_hi) // 2)
        ys = np.array(ys); xs = np.array(xs)
        if len(set(xs)) < 3:
            continue
        try:
            coef = np.polyfit(ys, xs, 2)        # x = f(y), 2nd order
        except Exception:
            continue
        ploty = np.arange(y_top, h, sample_step)
        plotx = np.polyval(coef, ploty)
        pts = np.stack([plotx, ploty], axis=1)
        pts = pts[(plotx >= 0) & (plotx < w)]
        if len(pts) >= 2:
            lanes.append(pts.astype(np.float32))
    return lanes


# --------------------------------------------------------------------------
# 5. Segmentation mask -> lane points (for evaluating the seg model)
# --------------------------------------------------------------------------
def mask_to_lanes(mask, sample_step=10, min_pts=2):
    """SEGMENTATION method: HxW lane-id mask (1..4) -> per-lane (x,y) point lists
    via row-wise centroid. Raw, no smoothing (occlusion gaps stay as gaps)."""
    h, w = mask.shape
    lanes = []
    for lane_id in range(1, 5):
        xs, ys = [], []
        for y in range(0, h, sample_step):
            cols = np.where(mask[y] == lane_id)[0]
            if len(cols):
                xs.append(cols.mean()); ys.append(y)
        if len(xs) >= min_pts:
            lanes.append(np.stack([xs, ys], axis=1).astype(np.float32))
    return lanes


def lanes_to_curves(lanes, order=2, sample_step=10, y_min=None, y_max=IMG_H,
                    extrapolate=True):
    """CURVE method: fit an order-N polynomial x = f(y) to each lane's points and
    resample densely. Interpolates across occlusion gaps and (optionally)
    extrapolates to the full observed y-range -> smoother, gap-filled curves.
    Input: list of (N,2) point arrays (e.g. from mask_to_lanes or the seg model).
    """
    curves = []
    for pts in lanes:
        if len(pts) < order + 1:
            curves.append(pts)            # too few points to fit; keep as-is
            continue
        ys, xs = pts[:, 1], pts[:, 0]
        try:
            coef = np.polyfit(ys, xs, order)
        except Exception:
            curves.append(pts); continue
        lo = int(ys.min()) if (y_min is None and not extrapolate) else int(ys.min())
        hi = int(ys.max())
        if extrapolate:                   # span the lane's full observed extent
            lo, hi = int(ys.min()), int(ys.max())
        grid = np.arange(lo, hi + 1, sample_step)
        cx = np.polyval(coef, grid)
        c = np.stack([cx, grid], axis=1).astype(np.float32)
        c = c[(c[:, 0] >= 0) & (c[:, 0] < IMG_W)]
        if len(c) >= 2:
            curves.append(c)
    return curves
