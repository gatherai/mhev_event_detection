"""
MHE-V (Material Handling Equipment Vision) pallet pick/drop event detector.
Uses edge detection on depth signal for robust event detection.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class DepthStats:
    """Per-frame depth statistics within the ROI."""
    median_depth_m: float
    variance_m: float
    valid_pixel_pct: float


@dataclass
class PalletEvent:
    """Output event when a pick-up or drop-off is detected."""
    event_type: Optional[str]  # "PICK_UP" | "DROP_OFF" | None
    confidence: float          # 0.0 - 1.0
    edge_strength: float       # Magnitude of depth change
    depth_m: float             # Current median depth
    frame_idx: int


@dataclass
class DetectorConfig:
    """All tunable detection parameters."""
    # ROI definition (x, y, width, height in pixels)
    roi_x: int = 60
    roi_y: int = 80
    roi_width: int = 80
    roi_height: int = 80

    # Depth filtering
    min_depth_m: float = 0.3
    max_depth_m: float = 8.0

    # Edge detection parameters
    window_size: int = 10              # Frames for before/after comparison (1 sec at 10Hz)
    min_edge_strength_m: float = 0.25  # Minimum depth change to trigger event
    smoothing_window: int = 5          # Frames for smoothing before edge detection

    # Event filtering
    min_event_gap_frames: int = 30     # Minimum frames between events (3 sec)
    min_valid_pixels_pct: float = 10.0 # Minimum valid pixels to use frame

    frame_rate_hz: float = 10.0

    def to_dict(self) -> dict:
        """Export config to dictionary for JSON serialization."""
        return {
            "roi": [self.roi_x, self.roi_y, self.roi_width, self.roi_height],
            "min_depth_m": self.min_depth_m,
            "max_depth_m": self.max_depth_m,
            "window_size": self.window_size,
            "min_edge_strength_m": self.min_edge_strength_m,
            "smoothing_window": self.smoothing_window,
            "min_event_gap_frames": self.min_event_gap_frames,
            "min_valid_pixels_pct": self.min_valid_pixels_pct,
            "frame_rate_hz": self.frame_rate_hz,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DetectorConfig":
        """Load config from dictionary."""
        config = cls()
        if "roi" in d:
            config.roi_x, config.roi_y, config.roi_width, config.roi_height = d["roi"]

        for key in ["min_depth_m", "max_depth_m", "window_size", "min_edge_strength_m",
                    "smoothing_window", "min_event_gap_frames", "min_valid_pixels_pct",
                    "frame_rate_hz"]:
            if key in d:
                setattr(config, key, d[key])

        return config


class EdgeDetector:
    """
    Detects pallet pick-up and drop-off events using edge detection.

    Algorithm:
    1. Extract median depth from ROI for each frame
    2. Smooth the depth signal
    3. Compute edge signal: difference between after-window and before-window
    4. Find peaks in edge signal that exceed threshold
    5. Classify: negative edge = PICK_UP, positive edge = DROP_OFF
    """

    def __init__(self, config: DetectorConfig = None):
        self.config = config or DetectorConfig()

    def extract_roi(self, depth_image: np.ndarray) -> np.ndarray:
        """Extract the region of interest from the depth image."""
        c = self.config
        y1 = max(0, c.roi_y)
        y2 = min(depth_image.shape[0], c.roi_y + c.roi_height)
        x1 = max(0, c.roi_x)
        x2 = min(depth_image.shape[1], c.roi_x + c.roi_width)
        return depth_image[y1:y2, x1:x2]

    def compute_frame_stats(self, depth_image: np.ndarray) -> DepthStats:
        """Compute depth statistics for a single frame."""
        c = self.config
        roi = self.extract_roi(depth_image)

        # Filter valid pixels
        valid_mask = (
            (roi > c.min_depth_m) &
            (roi < c.max_depth_m) &
            np.isfinite(roi)
        )
        valid_pixels = roi[valid_mask]

        total_pixels = roi.size
        valid_pct = (len(valid_pixels) / total_pixels * 100) if total_pixels > 0 else 0

        if len(valid_pixels) == 0:
            return DepthStats(
                median_depth_m=np.nan,
                variance_m=np.nan,
                valid_pixel_pct=0.0
            )

        return DepthStats(
            median_depth_m=float(np.median(valid_pixels)),
            variance_m=float(np.var(valid_pixels)),
            valid_pixel_pct=valid_pct
        )

    def process_batch(self, depth_frames: List[np.ndarray]) -> Tuple[List[float], List[PalletEvent]]:
        """
        Process a batch of depth frames and detect all events.

        Args:
            depth_frames: List of depth images (numpy arrays)

        Returns:
            Tuple of (depth_history, detected_events)
        """
        c = self.config
        n_frames = len(depth_frames)

        # Step 1: Extract median depth for each frame
        depths = []
        for frame in depth_frames:
            stats = self.compute_frame_stats(frame)
            if stats.valid_pixel_pct >= c.min_valid_pixels_pct:
                depths.append(stats.median_depth_m)
            else:
                depths.append(np.nan)

        depths = np.array(depths)

        # Step 2: Interpolate NaN values for continuity
        depths_interp = self._interpolate_nans(depths)

        # Step 3: Smooth the depth signal
        if c.smoothing_window > 1:
            kernel = np.ones(c.smoothing_window) / c.smoothing_window
            smoothed = np.convolve(depths_interp, kernel, mode='same')
        else:
            smoothed = depths_interp

        # Step 4: Compute edge signal (after - before)
        edge_signal = np.zeros(n_frames)
        w = c.window_size

        for i in range(w, n_frames - w):
            before = np.nanmean(smoothed[i - w:i])
            after = np.nanmean(smoothed[i:i + w])
            if not np.isnan(before) and not np.isnan(after):
                edge_signal[i] = after - before

        # Step 5: Find events (peaks in edge signal)
        events = self._find_events(edge_signal, smoothed)

        return depths.tolist(), events

    def _interpolate_nans(self, arr: np.ndarray) -> np.ndarray:
        """Interpolate NaN values in array."""
        result = arr.copy()
        nans = np.isnan(result)
        if np.all(nans):
            return result

        # Linear interpolation
        x = np.arange(len(result))
        result[nans] = np.interp(x[nans], x[~nans], result[~nans])
        return result

    def _find_events(self, edge_signal: np.ndarray, smoothed_depths: np.ndarray) -> List[PalletEvent]:
        """Find events from edge signal."""
        c = self.config
        events = []
        n = len(edge_signal)
        w = c.window_size

        # Find frames where edge exceeds threshold
        for i in range(w, n - w):
            edge = edge_signal[i]

            if abs(edge) < c.min_edge_strength_m:
                continue

            # Check if this is a local extremum (peak or trough)
            is_peak = True
            search_range = min(5, w // 2)  # Look +/- 5 frames

            for j in range(max(w, i - search_range), min(n - w, i + search_range + 1)):
                if j == i:
                    continue
                if edge > 0 and edge_signal[j] > edge:  # Looking for max positive
                    is_peak = False
                    break
                if edge < 0 and edge_signal[j] < edge:  # Looking for max negative
                    is_peak = False
                    break

            if not is_peak:
                continue

            # Check minimum gap from previous event
            if events and (i - events[-1].frame_idx) < c.min_event_gap_frames:
                # Keep the stronger event
                if abs(edge) > abs(events[-1].edge_strength):
                    events[-1] = PalletEvent(
                        event_type="DROP_OFF" if edge > 0 else "PICK_UP",
                        confidence=min(1.0, abs(edge) / (c.min_edge_strength_m * 2)),
                        edge_strength=edge,
                        depth_m=smoothed_depths[i],
                        frame_idx=i
                    )
                continue

            # Create event
            event_type = "DROP_OFF" if edge > 0 else "PICK_UP"
            confidence = min(1.0, abs(edge) / (c.min_edge_strength_m * 2))

            events.append(PalletEvent(
                event_type=event_type,
                confidence=confidence,
                edge_strength=edge,
                depth_m=smoothed_depths[i],
                frame_idx=i
            ))

        # Post-process: merge consecutive same-type events (keep strongest)
        events = self._merge_consecutive_same_type(events)

        return events

    def _merge_consecutive_same_type(self, events: List[PalletEvent]) -> List[PalletEvent]:
        """
        Merge consecutive events of the same type if they're close together.

        Logic: You can't pick up twice without dropping off, and vice versa.
        If we see PICK_UP, PICK_UP close together - keep the LAST one (final action).
        If they're far apart - keep both (could be separate real events).
        """
        if len(events) <= 1:
            return events

        # Max gap to consider events as duplicates (frames)
        # Events farther apart than this are considered separate
        max_merge_gap = self.config.min_event_gap_frames * 5  # e.g., 80 * 5 = 400 frames = 40 sec

        merged = [events[0]]

        for event in events[1:]:
            if event.event_type == merged[-1].event_type:
                # Same type as previous
                gap = event.frame_idx - merged[-1].frame_idx

                if gap <= max_merge_gap:
                    # Close together - keep the LAST one (replace previous)
                    merged[-1] = event
                else:
                    # Far apart - keep both (separate events)
                    merged.append(event)
            else:
                # Different type - add to list
                merged.append(event)

        return merged


# Keep old class name for compatibility
PalletEventDetector = EdgeDetector
