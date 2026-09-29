"""
Confidence Estimation Module.
Assesses reliability of eye gaze data based on face pose,
eye openness, pupil contrast, and binocular ratio consistency.
"""

from typing import Optional, Tuple


class GazeConfidence:
    @staticmethod
    def evaluate(
        face_detected: bool,
        left_eye,
        right_eye,
        min_threshold: float = 0.35
    ) -> Tuple[float, bool, str]:
        """
        Calculates a confidence metric in [0.0, 1.0].
        Returns (confidence_score, is_valid, status_message).
        """
        if not face_detected:
            return 0.0, False, "Face not detected"

        if left_eye is None and right_eye is None:
            return 0.0, False, "Eyes not localized"

        # Check for full binocular blink (both eyes closed)
        is_blinking_left = left_eye.is_blinking if left_eye else False
        is_blinking_right = right_eye.is_blinking if right_eye else False

        if is_blinking_left and is_blinking_right:
            return 0.1, False, "Blinking"

        score = 0.50

        # Eye openness contribution
        avg_ear = 0.0
        active_eyes = 0
        if left_eye and not left_eye.is_blinking:
            avg_ear += left_eye.ear
            active_eyes += 1
        if right_eye and not right_eye.is_blinking:
            avg_ear += right_eye.ear
            active_eyes += 1

        if active_eyes > 0:
            avg_ear /= active_eyes
            openness_score = min(1.0, max(0.0, (avg_ear - 0.15) / 0.15))
            score += openness_score * 0.20

        # Pupil detection quality
        pupil_score = 0.0
        if left_eye and left_eye.pupil and left_eye.pupil.x is not None:
            pupil_score += left_eye.pupil.confidence
        if right_eye and right_eye.pupil and right_eye.pupil.x is not None:
            pupil_score += right_eye.pupil.confidence

        if active_eyes > 0:
            pupil_score /= active_eyes
            score += pupil_score * 0.15

        # Binocular agreement check
        if (left_eye and right_eye and 
            left_eye.horizontal_ratio is not None and 
            right_eye.horizontal_ratio is not None):
            h_diff = abs(left_eye.horizontal_ratio - right_eye.horizontal_ratio)
            v_diff = abs((left_eye.vertical_ratio or 0) - (right_eye.vertical_ratio or 0))
            if h_diff < 0.25 and v_diff < 0.25:
                score += 0.15
            else:
                score -= min(0.15, (h_diff + v_diff) * 0.3)

        final_score = float(max(0.0, min(1.0, score)))
        # Valid if score meets threshold and both eyes are not closed
        is_valid = (final_score >= min_threshold) and not (is_blinking_left and is_blinking_right)
        status = "Valid" if is_valid else ("Blinking" if (is_blinking_left and is_blinking_right) else "Low Confidence")

        return final_score, is_valid, status
