"""
Calibration System and Screen-Space Gaze Mapping.
Maps raw eye gaze ratios (horizontal, vertical) to screen coordinates (X, Y)
using Linear, 2nd-Degree Polynomial, Affine, or Homography models.
Includes auto-centering, sensitivity gain tuning, outlier rejection, and JSON profile persistence.
"""

import json
import os
import cv2
import numpy as np

from .reach import ReachModel
from typing import List, Tuple, Dict, Optional, Any


class CalibrationPoint:
    def __init__(self, screen_x: float, screen_y: float, is_validation: bool = False):
        self.screen_x = screen_x
        self.screen_y = screen_y
        self.is_validation = is_validation   # extra edge-anchor point (3% margin); error reported separately
        self.samples: List[Tuple[float, float]] = []
        self.head_samples: List[Tuple[float, float]] = []

    def add_sample(self, hr: float, vr: float, head: Optional[Tuple[float, ...]] = None) -> None:
        """head = extra pose features: (yaw, pitch) or (yaw, pitch, lid_aperture)."""
        if hr is not None and vr is not None and 0.0 <= hr <= 1.0 and 0.0 <= vr <= 1.0:
            self.samples.append((float(hr), float(vr)))
            self.head_samples.append(tuple(float(v) for v in head) if head is not None else (np.nan, np.nan))

    def get_filtered_head(self) -> Optional[Tuple[float, float]]:
        """Median head pose (yaw, pitch proxies) while looking at this target."""
        if len(self.head_samples) < 3:
            return None
        if len({len(h) for h in self.head_samples}) != 1:
            return None
        arr = np.array(self.head_samples, float)
        arr = arr[int(len(arr) * 0.2):] if len(arr) >= 10 else arr
        if np.isnan(arr).any():
            return None
        return tuple(float(v) for v in np.median(arr, axis=0))

    def get_filtered_center(self) -> Optional[Tuple[float, float]]:
        """Rejects outlier samples using distance from median and computes average."""
        if len(self.samples) < 3:
            return None

        arr = np.array(self.samples)
        # Drop the first 20% of samples: eye is often still settling / correcting after the saccade
        arr = arr[int(len(arr) * 0.2):] if len(arr) >= 10 else arr
        median = np.median(arr, axis=0)
        dists = np.linalg.norm(arr - median, axis=1)
        med_dist = np.median(dists)
        threshold = max(0.04, med_dist * 2.5)

        valid_mask = dists <= threshold
        valid_samples = arr[valid_mask]
        if len(valid_samples) == 0:
            valid_samples = arr

        # median, not mean: robust to residual outliers
        med_vals = np.median(valid_samples, axis=0)
        return float(med_vals[0]), float(med_vals[1])


class GazeCalibration:
    SUPPORTED_MODELS = ["polynomial", "affine", "homography", "linear"]

    def __init__(
        self,
        screen_width: int = 1920,
        screen_height: int = 1080,
        model_type: str = "polynomial",
        gain_x: float = 2.4,
        gain_y: float = 2.4,
        center_hr: float = 0.43,
        center_vr: float = 0.39
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.model_type = model_type if model_type in self.SUPPORTED_MODELS else "polynomial"

        # Adaptive sensitivity and centering
        self.gain_x = gain_x
        self.gain_y = gain_y
        self.center_hr = center_hr
        self.center_vr = center_vr

        self.is_calibrated = False
        self.rmse_error_px = 0.0
        self.loo_error_px: Optional[float] = None   # leave-one-out error = honest accuracy estimate
        self.validation_error_px: Optional[float] = None         # error on held-out EDGE targets
        self._poly_degree = 2
        self._edge_mask: List[bool] = []
        self._use_head = False
        self._extra_mask: List[bool] = []
        self._head_ref: Tuple[float, ...] = (0.0, 0.0)
        self._fit_head_arr = None
        self.reach = ReachModel(screen_w=screen_width, screen_h=screen_height)   # set by reach tuning
        self.edge_gain_x = 1.0     # manual stretch about screen centre (config edge_gain_x/y)
        self.edge_gain_y = 1.0
        self.last_fit_message: str = ""

        # Calibrated model parameters
        self._poly_cx: Optional[np.ndarray] = None
        self._poly_cy: Optional[np.ndarray] = None
        self._affine_mat: Optional[np.ndarray] = None
        self._homography_mat: Optional[np.ndarray] = None
        self._linear_params: Optional[Dict[str, float]] = None

        # Point storage
        self.calibration_points: List[CalibrationPoint] = []
        self._paired_data: List[Dict[str, float]] = []

    def set_center(self, hr: float, vr: float):
        """Re-centers screen gaze coordinate mapping to the user's current resting point."""
        if hr is not None and vr is not None and 0.2 < hr < 0.8 and 0.2 < vr < 0.8:
            self.center_hr = float(hr)
            self.center_vr = float(vr)

    def setup_grid(self, rows: int = 4, cols: int = 4, margin_x: float = 0.05, margin_y: float = 0.05) -> List[Tuple[int, int]]:
        """Generates the calibration grid (default 4x4 = 16 points, 5% from the edges)."""
        self.calibration_points.clear()
        screen_pts = []

        xs = np.linspace(self.screen_width * margin_x, self.screen_width * (1.0 - margin_x), cols)
        ys = np.linspace(self.screen_height * margin_y, self.screen_height * (1.0 - margin_y), rows)

        for y in ys:
            for x in xs:
                pt = (int(x), int(y))
                screen_pts.append(pt)
                self.calibration_points.append(CalibrationPoint(pt[0], pt[1]))

        return screen_pts

    def add_validation_points(self, margin: float = 0.03) -> List[Tuple[int, int]]:
        """Appends 8 extra targets on the extreme screen edges (4 corners + 4 mid-edges, 3% margin).
        They anchor the mapping at the borders so the edge rows/columns are reached, and their
        leave-one-out error is reported separately as the 'edge error'."""
        W, H = self.screen_width, self.screen_height
        fr = [(margin, margin), (0.5, margin), (1 - margin, margin),
              (margin, 0.5), (1 - margin, 0.5),
              (margin, 1 - margin), (0.5, 1 - margin), (1 - margin, 1 - margin)]
        pts = [(int(W * fx), int(H * fy)) for fx, fy in fr]
        for p in pts:
            self.calibration_points.append(CalibrationPoint(p[0], p[1], is_validation=True))
        return pts

    def fit(self) -> bool:
        """Fits calibration models using collected points."""
        src_points = []
        dst_points = []
        self._paired_data.clear()
        self.reach = ReachModel(screen_w=self.screen_width, screen_h=self.screen_height)  # re-tune after recalibration

        self._edge_mask = []
        heads = []
        for cp in self.calibration_points:
            center = cp.get_filtered_center()
            if center is not None:
                heads.append(cp.get_filtered_head())
                self._edge_mask.append(cp.is_validation)
                src_points.append(center)
                dst_points.append((cp.screen_x, cp.screen_y))
                self._paired_data.append({
                    "hr": center[0], "vr": center[1],
                    "screen_x": cp.screen_x, "screen_y": cp.screen_y
                })

        if len(src_points) < 4:
            self.is_calibrated = False
            return False

        src_arr = np.array(src_points, dtype=np.float64)
        dst_arr = np.array(dst_points, dtype=np.float64)

        # Coverage check: every border of the grid must have at least one usable point,
        # otherwise the mapping has to extrapolate there (-> unreachable edge rows/columns).
        W, H = self.screen_width, self.screen_height
        if len(self.calibration_points) >= 9:
            got_x, got_y = dst_arr[:, 0], dst_arr[:, 1]
            missing = [name for name, ok in (
                ("left edge", np.any(got_x <= 0.10 * W)),
                ("right edge", np.any(got_x >= 0.90 * W)),
                ("top edge", np.any(got_y <= 0.10 * H)),
                ("bottom edge", np.any(got_y >= 0.90 * H)))
                if not ok]
            if missing:
                self.last_fit_message = ("Rejected: no valid samples for the " + ", ".join(missing) +
                                         ". Check lighting / camera angle so the eyes stay visible there, then recalibrate.")
                self.is_calibrated = False
                return False

        # Sanity check: if the eye features barely moved across the screen, the mapping will
        # amplify noise enormously (the old profile had gain ~12,000 px per unit ratio).
        hr_span = float(np.ptp(src_arr[:, 0]))
        vr_span = float(np.ptp(src_arr[:, 1]))
        if hr_span < self.MIN_HR_SPAN or vr_span < self.MIN_VR_SPAN:
            self.last_fit_message = (f"Rejected: eye-feature span too small (hr {hr_span:.3f}, vr {vr_span:.3f}). "
                                     "Move closer to the camera / improve lighting and recalibrate.")
            self.is_calibrated = False
            return False

        try:
            # Update center from median of calibration points
            self.center_hr = float(np.median(src_arr[:, 0]))
            self.center_vr = float(np.median(src_arr[:, 1]))

            # 1. Fit Polynomial (2nd degree bivariate)
            u = src_arr[:, 0]
            v = src_arr[:, 1]
            self._poly_degree = 3 if len(u) >= self.CUBIC_MIN_POINTS else 2

            # Head-pose compensation: people turn/tilt their head towards the screen edges,
            # which REDUCES how far the eyes rotate (on the candidate video head yaw correlated
            # -0.6 with the horizontal eye ratio, pitch -0.8 with the vertical). Without head
            # terms the mapping under-reaches the right columns / bottom rows whenever the user
            # moves the head more than during calibration. Head terms are kept only if they
            # lower the leave-one-out error.
            #
            # v7: the extra vector may also carry the eyelid aperture (3rd value). Looking down
            # lowers the lids; how strongly differs per person, so instead of a fixed fusion weight
            # the calibration learns it. Every subset (none / head / lid / head+lid) is tried and
            # the one with the lowest leave-one-out error wins -> never worse than without.
            head_arr = None
            if all(h is not None for h in heads) and len(heads) >= 12 and len({len(h) for h in heads}) == 1:
                head_arr = np.array(heads, float)
            self._use_head = False
            self._extra_mask = []
            self._head_ref = (0.0, 0.0)
            best_loo = self._leave_one_out_error(src_arr, dst_arr, None)
            best_mask = None
            if head_arr is not None:
                k = head_arr.shape[1]
                spans_ok = [np.ptp(head_arr[:, j]) > self.MIN_HEAD_SPAN for j in range(k)]
                cands = [[True, True] + [False] * (k - 2)]
                if k >= 3:
                    cands += [[False, False, True], [True, True, True]]
                ref = tuple(float(np.median(head_arr[:, j])) for j in range(k))
                for mask in cands:
                    if not any(m and ok for m, ok in zip(mask, spans_ok)):
                        continue
                    self._use_head, self._extra_mask, self._head_ref = True, mask, ref
                    loo = self._leave_one_out_error(src_arr, dst_arr, head_arr)
                    if loo < best_loo * 0.98:          # must help by >2% to be worth the extra terms
                        best_loo, best_mask = loo, mask
            if best_mask is not None:
                self._use_head, self._extra_mask = True, best_mask
            else:
                self._use_head, self._extra_mask = False, []
            self._fit_head_arr = head_arr if self._use_head else None
            A = self._poly_basis(u, v, self._fit_head_arr)

            cx, _, _, _ = np.linalg.lstsq(A, dst_arr[:, 0], rcond=None)
            cy, _, _, _ = np.linalg.lstsq(A, dst_arr[:, 1], rcond=None)
            self._poly_cx = cx
            self._poly_cy = cy

            # 2. Fit Affine
            affine, _ = cv2.estimateAffine2D(src_arr, dst_arr)
            if affine is not None:
                self._affine_mat = affine

            # 3. Fit Homography
            H, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC, 10.0)
            if H is not None:
                self._homography_mat = H

            # 4. Fit Linear
            min_hr, max_hr = float(np.min(u)), float(np.max(u))
            min_vr, max_vr = float(np.min(v)), float(np.max(v))
            min_sx, max_sx = float(np.min(dst_arr[:, 0])), float(np.max(dst_arr[:, 0]))
            min_sy, max_sy = float(np.min(dst_arr[:, 1])), float(np.max(dst_arr[:, 1]))

            self._linear_params = {
                "min_hr": min_hr, "max_hr": max_hr,
                "min_vr": min_vr, "max_vr": max_vr,
                "min_sx": min_sx, "max_sx": max_sx,
                "min_sy": min_sy, "max_sy": max_sy
            }

            self.is_calibrated = True

            errors = []
            for i, (s_pt, d_pt) in enumerate(zip(src_points, dst_points)):
                hd = tuple(self._fit_head_arr[i]) if self._fit_head_arr is not None else None
                est = self.map_gaze(s_pt[0], s_pt[1], hd)
                if est is not None:
                    errors.append((est[0] - d_pt[0])**2 + (est[1] - d_pt[1])**2)
            self.rmse_error_px = float(np.sqrt(np.mean(errors))) if errors else 0.0
            self.loo_error_px = self._leave_one_out_error(src_arr, dst_arr, self._fit_head_arr)
            msg = f"Expected error (leave-one-out, all targets): {self.loo_error_px:.0f}px"
            if self.validation_error_px is not None:
                msg += f" | at screen edges: {self.validation_error_px:.0f}px"
            if self._use_head:
                parts = (["head pose"] if any(self._extra_mask[:2]) else []) + \
                        (["eyelid"] if len(self._extra_mask) > 2 and self._extra_mask[2] else [])
                msg += " | compensation: " + " + ".join(parts)
            geo = getattr(self, "geometry", None)          # (diagonal_in, distance_cm) set by the app
            if geo:
                from .geometry import px_to_deg
                deg = px_to_deg(self.loo_error_px, self.screen_width, self.screen_height, *geo)
                msg += f" (~{deg:.1f} deg)"
            self.last_fit_message = msg
            return True

        except Exception:
            self.is_calibrated = False
            return False

    MIN_HR_SPAN = 0.05
    MIN_VR_SPAN = 0.04

    def _leave_one_out_error(self, src: np.ndarray, dst: np.ndarray, head: Optional[np.ndarray] = None) -> float:
        """RMSE predicting each calibration point from a model fitted on the others.
        Training RMSE with 6 polynomial terms on 9 points is always optimistic."""
        errs = []
        for i in range(len(src)):
            m = np.ones(len(src), bool); m[i] = False
            u, v = src[m, 0], src[m, 1]
            hm = head[m] if head is not None else None
            hi = head[i:i + 1] if head is not None else None
            if self.model_type == "polynomial" and m.sum() > len(self._poly_basis(u[:1], v[:1], hi)[0]):
                A = self._poly_basis(u, v, hm)
                basis = self._poly_basis(src[i:i + 1, 0], src[i:i + 1, 1], hi)[0]
            else:   # affine fallback
                A = np.column_stack([np.ones_like(u), u, v])
                basis = np.array([1.0, src[i, 0], src[i, 1]])
            cx = np.linalg.lstsq(A, dst[m, 0], rcond=None)[0]
            cy = np.linalg.lstsq(A, dst[m, 1], rcond=None)[0]
            errs.append((basis @ cx - dst[i, 0])**2 + (basis @ cy - dst[i, 1])**2)
        errs = np.array(errs)
        edge = np.array(self._edge_mask[:len(errs)], bool) if len(self._edge_mask) == len(errs) else np.zeros(len(errs), bool)
        self.validation_error_px = float(np.sqrt(np.mean(errs[edge]))) if edge.any() else None
        return float(np.sqrt(np.mean(errs)))

    CUBIC_MIN_POINTS = 14
    MIN_HEAD_SPAN = 0.008     # head proxy units (~0.5 deg); below this head terms are not identifiable
    HEAD_SCALE = 10.0

    def _poly_basis(self, u, v, head=None):
        """Centred/scaled polynomial basis. Degree 3 (per-axis cubic terms) when enough points:
        eye/iris response flattens towards the screen edges, which a quadratic cannot follow
        -> the bottom rows / right columns were under-reached."""
        u = (np.asarray(u, float) - 0.5) * 5.0
        v = (np.asarray(v, float) - 0.5) * 5.0
        cols = [np.ones_like(u), u, v, u * u, u * v, v * v]
        if getattr(self, "_poly_degree", 2) >= 3:
            cols += [u ** 3, v ** 3]
        if getattr(self, "_use_head", False):
            mask = list(getattr(self, "_extra_mask", []) or [True, True])
            ref = np.asarray(self._head_ref, float)
            k = len(ref)
            if head is None:
                h = np.zeros((len(u), k))
            else:
                h = np.asarray(head, float).reshape(len(u), -1)
                if h.shape[1] < k:                      # missing features -> neutral
                    h = np.hstack([h, np.tile(ref[h.shape[1]:], (len(u), 1))])
                h = (h[:, :k] - ref) * self.HEAD_SCALE
            mask = (mask + [False] * k)[:k]
            for j in range(k):
                if mask[j]:
                    cols.append(h[:, j])
        return np.column_stack(cols)

    def map_gaze(self, hr: float, vr: float, head: Optional[Tuple[float, float]] = None,
                 apply_reach: bool = True) -> Optional[Tuple[float, float]]:
        """Calibrated mapping (+ head-pose compensation if fitted) + optional manual edge gain, clipped.
        `head` = (yaw, pitch) proxies from detector.head_pose(); None -> neutral head."""
        p = self._map_base(hr, vr, clip=False, head=head)
        if p is None:
            return None
        x, y = p
        if apply_reach and self.is_calibrated and not self.reach.is_identity:
            x, y = self.reach.apply(x, y)
        cx, cy = self.screen_width / 2.0, self.screen_height / 2.0
        x = cx + (x - cx) * self.edge_gain_x
        y = cy + (y - cy) * self.edge_gain_y
        return (float(np.clip(x, 0, self.screen_width)), float(np.clip(y, 0, self.screen_height)))

    def _out(self, x: float, y: float, clip: bool) -> Tuple[float, float]:
        if clip:
            return (float(np.clip(x, 0, self.screen_width)), float(np.clip(y, 0, self.screen_height)))
        return (float(x), float(y))

    def _map_base(self, hr: float, vr: float, clip: bool = True,
                  head: Optional[Tuple[float, float]] = None) -> Optional[Tuple[float, float]]:
        """
        Maps (horizontal_ratio, vertical_ratio) to screen coordinates (X, Y).
        Uses polynomial/affine calibration if available, combined with adaptive centering gain.
        """
        if hr is None or vr is None:
            return None

        hr = float(np.clip(hr, 0.0, 1.0))
        vr = float(np.clip(vr, 0.0, 1.0))

        # If polynomial calibrated
        if self.is_calibrated and self.model_type == "polynomial" and self._poly_cx is not None and self._poly_cy is not None:
            hd = None if head is None else np.array([head], float)
            if hd is not None and np.isnan(hd).any():
                hd = None
            basis = self._poly_basis([hr], [vr], hd)[0]
            x = float(np.dot(basis, self._poly_cx))
            y = float(np.dot(basis, self._poly_cy))
            return self._out(x, y, clip)

        elif self.is_calibrated and self.model_type == "linear" and self._linear_params is not None:
            p = self._linear_params
            hr_span = max(1e-4, p["max_hr"] - p["min_hr"])
            vr_span = max(1e-4, p["max_vr"] - p["min_vr"])
            norm_x = (hr - p["min_hr"]) / hr_span
            norm_y = (vr - p["min_vr"]) / vr_span
            x = p["min_sx"] + norm_x * (p["max_sx"] - p["min_sx"])
            y = p["min_sy"] + norm_y * (p["max_sy"] - p["min_sy"])
            return self._out(x, y, clip)

        # Default high-responsiveness centered ocular projection
        # Human eye movement span is ~ +- 0.08 from resting center
        dx = (hr - self.center_hr) * self.gain_x
        dy = (vr - self.center_vr) * self.gain_y

        norm_x = 0.5 + dx
        norm_y = 0.5 + dy

        x = norm_x * self.screen_width
        y = norm_y * self.screen_height

        return self._out(x, y, clip)

    def save(self, filepath: str) -> None:
        """Serializes calibration data and model parameters to JSON."""
        data = {
            "screen_width": self.screen_width,
            "screen_height": self.screen_height,
            "model_type": self.model_type,
            "gain_x": self.gain_x,
            "gain_y": self.gain_y,
            "center_hr": self.center_hr,
            "center_vr": self.center_vr,
            "rmse_error_px": self.rmse_error_px,
            "loo_error_px": self.loo_error_px,
            "validation_error_px": self.validation_error_px,
            "poly_degree": self._poly_degree,
            "use_head": self._use_head,
            "reach": self.reach.to_dict(),
            "head_ref": list(self._head_ref),
            "extra_mask": list(self._extra_mask),
            "feature_version": 3,
            "poly_cx": self._poly_cx.tolist() if self._poly_cx is not None else None,
            "poly_cy": self._poly_cy.tolist() if self._poly_cy is not None else None,
            "affine_mat": self._affine_mat.tolist() if self._affine_mat is not None else None,
            "homography_mat": self._homography_mat.tolist() if self._homography_mat is not None else None,
            "linear_params": self._linear_params,
            "paired_data": self._paired_data
        }
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load(self, filepath: str) -> bool:
        """Loads calibration from JSON file."""
        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.screen_width = data.get("screen_width", self.screen_width)
            self.screen_height = data.get("screen_height", self.screen_height)
            self.model_type = data.get("model_type", self.model_type)
            self.gain_x = data.get("gain_x", self.gain_x)
            self.gain_y = data.get("gain_y", self.gain_y)
            self.center_hr = data.get("center_hr", self.center_hr)
            self.center_vr = data.get("center_vr", self.center_vr)
            self.rmse_error_px = data.get("rmse_error_px", 0.0)
            self.loo_error_px = data.get("loo_error_px")
            self.validation_error_px = data.get("validation_error_px")
            self._poly_degree = int(data.get("poly_degree", 2))
            self._use_head = bool(data.get("use_head", False))
            self.reach = ReachModel.from_dict(data.get("reach"))
            self._head_ref = tuple(data.get("head_ref", (0.0, 0.0)))
            self._extra_mask = list(data.get("extra_mask", [True, True] if self._use_head else []))
            if data.get("feature_version", 1) != 3:
                # older profiles use different features / polynomial basis -> recalibrate
                return False

            self._poly_cx = np.array(data["poly_cx"]) if data.get("poly_cx") else None
            self._poly_cy = np.array(data["poly_cy"]) if data.get("poly_cy") else None
            self._affine_mat = np.array(data["affine_mat"]) if data.get("affine_mat") else None
            self._homography_mat = np.array(data["homography_mat"]) if data.get("homography_mat") else None
            self._linear_params = data.get("linear_params")
            self._paired_data = data.get("paired_data", [])

            self.is_calibrated = (
                self._poly_cx is not None or 
                self._affine_mat is not None or 
                self._homography_mat is not None or 
                self._linear_params is not None
            )
            return self.is_calibrated
        except Exception:
            return False
