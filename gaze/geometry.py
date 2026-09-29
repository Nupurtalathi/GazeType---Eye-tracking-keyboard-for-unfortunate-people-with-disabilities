"""Screen geometry helpers: convert pixels on screen to degrees of eye rotation."""
import math


def px_to_deg(px: float, screen_w: int, screen_h: int, diagonal_in: float, distance_cm: float) -> float:
    cm_per_px = diagonal_in * 2.54 / math.hypot(screen_w, screen_h)
    return math.degrees(math.atan2(px * cm_per_px, distance_cm))


def screen_span_deg(screen_w: int, screen_h: int, diagonal_in: float, distance_cm: float):
    """Total horizontal / vertical angle the screen covers for the eye."""
    cm_per_px = diagonal_in * 2.54 / math.hypot(screen_w, screen_h)
    hw, hh = screen_w * cm_per_px / 2, screen_h * cm_per_px / 2
    return (2 * math.degrees(math.atan2(hw, distance_cm)), 2 * math.degrees(math.atan2(hh, distance_cm)))
