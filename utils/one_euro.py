"""
One Euro filter (Casiez, Roussel, Vogel, CHI 2012).

An adaptive low-pass filter: heavy smoothing while the signal moves slowly,
where jitter is visible, and almost none while it moves fast, where lag is.
"""
import math
from typing import Optional, Sequence, Tuple


def _alpha(cutoff_hz: float, dt: float) -> float:
    """Smoothing factor of a first-order low-pass at cutoff_hz for a step of dt."""
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


def _all_finite(value: Sequence[float]) -> bool:
    return all(math.isfinite(v) for v in value)


class OneEuroFilter:
    """
    One Euro filter over a fixed-length vector. All components share one
    cutoff, driven by the speed of the whole vector, so a 3D position is
    smoothed as a point rather than as three unrelated numbers.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0,
                 d_cutoff: float = 1.0, reset_after: float = 0.5):
        """
        Args:
            min_cutoff: Cutoff (Hz) at rest. Lower = steadier, laggier when still.
            beta: How fast the cutoff rises with speed. Higher = less lag in motion.
            d_cutoff: Cutoff (Hz) for the speed estimate itself.
            reset_after: Seconds without samples after which the filter starts
                over, so a hand that reappears jumps instead of gliding in.
        """
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset_after = reset_after
        self._value: Optional[Tuple[float, ...]] = None
        self._speed = 0.0
        self._time: Optional[float] = None

    def reset(self):
        """Forget all history."""
        self._value = None
        self._speed = 0.0
        self._time = None

    def __call__(self, value: Sequence[float], t: float) -> Tuple[float, ...]:
        """
        Filter one sample.

        Args:
            value: New sample
            t: Its timestamp in seconds

        Returns:
            Filtered value. A sample with NaN or infinity is ignored: the last output is
            returned, or the sample itself when there is none yet.
        """
        value = tuple(float(v) for v in value)
        if not _all_finite(value):
            # Kept in the history, it would poison every output until the next reset
            return self._value if self._value is not None else value
        # First sample, a long gap, or a non-increasing timestamp: start over
        if self._time is None or not 0.0 < t - self._time <= self.reset_after:
            self._value, self._speed, self._time = value, 0.0, t
            return value

        dt = t - self._time
        raw_speed = math.sqrt(sum((v - p) ** 2 for v, p in zip(value, self._value))) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._speed += a_d * (raw_speed - self._speed)

        cutoff = self.min_cutoff + self.beta * self._speed
        a = _alpha(cutoff, dt)
        self._value = tuple(p + a * (v - p) for v, p in zip(value, self._value))
        self._time = t
        return self._value


class ExponentialFilter:
    """
    Fixed-rate exponential smoothing: each sample moves the output a constant
    fraction of the way. The filter used before One Euro, kept for A/B
    comparison. Same call signature as OneEuroFilter.
    """

    def __init__(self, follow: float, reset_after: float = 0.5):
        """
        Args:
            follow: Fraction of each new sample to follow (1.0 = no smoothing)
            reset_after: Seconds without samples after which the filter starts over
        """
        self.follow = min(1.0, max(0.05, follow))
        self.reset_after = reset_after
        self._value: Optional[Tuple[float, ...]] = None
        self._time: Optional[float] = None

    def __call__(self, value: Sequence[float], t: float) -> Tuple[float, ...]:
        value = tuple(float(v) for v in value)
        if not _all_finite(value):
            return self._value if self._value is not None else value
        if self._time is None or not 0.0 < t - self._time <= self.reset_after:
            self._value = value
        else:
            self._value = tuple(p + self.follow * (v - p) for v, p in zip(value, self._value))
        self._time = t
        return self._value


class QuaternionExponentialFilter(ExponentialFilter):
    """ExponentialFilter for unit quaternions (qw, qx, qy, qz)."""

    def __call__(self, q: Sequence[float], t: float) -> Tuple[float, ...]:
        if not _all_finite(q):
            return super().__call__(q, t)  # ignored, history untouched
        if self._value is not None and sum(a * b for a, b in zip(q, self._value)) < 0.0:
            q = tuple(-c for c in q)
        filtered = super().__call__(q, t)
        norm = math.sqrt(sum(c * c for c in filtered)) or 1.0
        self._value = tuple(c / norm for c in filtered)
        return self._value


class PassThroughFilter:
    """No smoothing at all, for comparison."""

    def __call__(self, value: Sequence[float], t: float) -> Tuple[float, ...]:
        return tuple(float(v) for v in value)


class QuaternionOneEuroFilter(OneEuroFilter):
    """One Euro filter for unit quaternions (qw, qx, qy, qz)."""

    def __call__(self, q: Sequence[float], t: float) -> Tuple[float, ...]:
        if not _all_finite(q):
            return super().__call__(q, t)  # ignored, history untouched
        # q and -q are the same rotation; keep the sample on the same side as
        # the history so the filter does not average across the sign flip
        if self._value is not None and sum(a * b for a, b in zip(q, self._value)) < 0.0:
            q = tuple(-c for c in q)
        filtered = super().__call__(q, t)
        norm = math.sqrt(sum(c * c for c in filtered)) or 1.0
        self._value = tuple(c / norm for c in filtered)
        return self._value
