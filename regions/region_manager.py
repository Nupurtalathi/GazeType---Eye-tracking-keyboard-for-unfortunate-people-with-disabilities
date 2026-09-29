"""
Interactive Region and Dwell Dispatch Manager.
Enables UI element dwell mapping for accessibility applications,
such as eye-gaze controlled keyboards.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Callable, Tuple
import time


@dataclass
class Region:
    id: str
    x: float
    y: float
    width: float
    height: float
    dwell_time: Optional[float] = None  # None = use global default
    callback: Optional[Callable[[str], None]] = None
    label: str = ""

    def contains(self, gx: float, gy: float) -> bool:
        """Returns True if point (gx, gy) lies inside the rectangular region."""
        return (self.x <= gx <= (self.x + self.width)) and (self.y <= gy <= (self.y + self.height))


class RegionManager:
    def __init__(self, default_dwell_time: float = 1.50):
        self.default_dwell_time = default_dwell_time
        self.regions: Dict[str, Region] = {}
        self.global_dwell_callbacks: List[Callable[[str], None]] = []

        # Active dwelling tracking
        self.active_region_id: Optional[str] = None
        self.region_dwell_time: float = 0.0
        self.has_triggered: bool = False
        self.last_gaze_point: Optional[Tuple[float, float]] = None
        self.last_update_time: Optional[float] = None

    def register_region(self, region: Region) -> None:
        """Registers an interactive rectangular region."""
        self.regions[region.id] = region

    def unregister_region(self, region_id: str) -> None:
        """Removes a region by ID."""
        self.regions.pop(region_id, None)
        if self.active_region_id == region_id:
            self._reset_active()

    def clear_regions(self) -> None:
        self.regions.clear()
        self._reset_active()

    def on_dwell(self, callback: Callable[[str], None]) -> None:
        """Registers a global callback fired whenever any region reaches its dwell threshold."""
        if callback not in self.global_dwell_callbacks:
            self.global_dwell_callbacks.append(callback)

    def update(
        self,
        gaze_x: Optional[float],
        gaze_y: Optional[float],
        is_valid: bool = True,
        timestamp: Optional[float] = None
    ) -> Optional[str]:
        """
        Updates region dwelling.
        Returns region_id if a dwell event was triggered on this frame, else None.
        """
        if timestamp is None:
            timestamp = time.time()

        dt = (timestamp - self.last_update_time) if self.last_update_time is not None else 0.0
        self.last_update_time = timestamp

        if gaze_x is not None and gaze_y is not None:
            self.last_gaze_point = (gaze_x, gaze_y)

        if not is_valid or gaze_x is None or gaze_y is None:
            self._reset_active()
            return None

        # Find which region contains gaze
        hovered_region: Optional[Region] = None
        for reg in self.regions.values():
            if reg.contains(gaze_x, gaze_y):
                hovered_region = reg
                break

        if hovered_region is None:
            self._reset_active()
            return None

        # If entering a new region
        if self.active_region_id != hovered_region.id:
            self.active_region_id = hovered_region.id
            self.region_dwell_time = 0.0
            self.has_triggered = False

        # Accumulate dwell in this region
        self.region_dwell_time += dt
        target_dwell = hovered_region.dwell_time or self.default_dwell_time

        if self.region_dwell_time >= target_dwell and not self.has_triggered:
            self.has_triggered = True
            reg_id = hovered_region.id

            # Trigger region-specific callback
            if hovered_region.callback:
                try:
                    hovered_region.callback(reg_id)
                except Exception as e:
                    print(f"[RegionManager] Region callback error: {e}")

            # Trigger global callbacks
            for cb in self.global_dwell_callbacks:
                try:
                    cb(reg_id)
                except Exception as e:
                    print(f"[RegionManager] Global callback error: {e}")

            return reg_id

        return None

    def _reset_active(self) -> None:
        self.active_region_id = None
        self.region_dwell_time = 0.0
        self.has_triggered = False

    def get_gaze_position(self) -> Optional[Tuple[float, float]]:
        return self.last_gaze_point

    def get_dwell_state(self) -> Tuple[Optional[str], float, float]:
        """Returns (active_region_id, current_dwell_time, target_dwell_time)."""
        if self.active_region_id and self.active_region_id in self.regions:
            target = self.regions[self.active_region_id].dwell_time or self.default_dwell_time
            return (self.active_region_id, self.region_dwell_time, target)
        return (None, 0.0, self.default_dwell_time)
