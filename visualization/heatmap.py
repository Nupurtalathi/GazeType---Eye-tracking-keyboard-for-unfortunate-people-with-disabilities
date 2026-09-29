"""
Gaze Heatmap Generation Module.
Computes 2D spatial gaze point density using Gaussian kernels,
maps it to an aesthetic color gradient, and renders high-resolution heatmaps.
"""

import numpy as np
import cv2
import matplotlib.cm as cm
from typing import List, Tuple, Optional


class GazeHeatmap:
    @staticmethod
    def generate(
        points: List[Tuple[float, float]],
        width: int = 1920,
        height: int = 1080,
        sigma: int = 45,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Generates a 2D density heatmap image (RGB) from a list of screen (x, y) coordinates.
        """
        if not points:
            # Return blank dark image
            return np.full((height, width, 3), 25, dtype=np.uint8)

        # Scale down for ultra-fast convolution if canvas is large
        scale = 0.5
        scaled_w = int(width * scale)
        scaled_h = int(height * scale)

        density = np.zeros((scaled_h, scaled_w), dtype=np.float32)

        for px, py in points:
            sx = int(px * scale)
            sy = int(py * scale)
            if 0 <= sx < scaled_w and 0 <= sy < scaled_h:
                density[sy, sx] += 1.0

        # Apply Gaussian Blur to diffuse gaze fixations into smooth heat blobs
        scaled_sigma = max(3, int(sigma * scale))
        ksize = scaled_sigma * 6 + 1
        blurred = cv2.GaussianBlur(density, (ksize, ksize), scaled_sigma)

        # Normalize 0 to 255
        max_val = np.max(blurred)
        if max_val > 0:
            norm = (blurred / max_val * 255).astype(np.uint8)
        else:
            norm = np.zeros((scaled_h, scaled_w), dtype=np.uint8)

        # Apply colormap
        colored = cv2.applyColorMap(norm, colormap)

        # Darken low-density background
        mask = norm < 15
        colored[mask] = [20, 20, 25]  # Charcoal background

        # Scale back to full resolution
        result = cv2.resize(colored, (width, height), interpolation=cv2.INTER_LINEAR)
        return result

    @staticmethod
    def save(image: np.ndarray, filepath: str) -> None:
        """Saves heatmap to disk."""
        cv2.imwrite(filepath, image)
