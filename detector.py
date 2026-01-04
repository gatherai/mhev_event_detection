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
class FilterStats:
    """Statistics about what each filter rejected."""
    spike_rejected: int = 0           # PICK_UP→DROP_OFF pairs rejected
    dwell_rejected: int = 0           # Events rejected for insufficient dwell time
    variance_rejected: int = 0        # Events rejected for high variance
    baseline_rejected: int = 0        # Events rejected for returning to baseline
    pre_stability_rejected: int = 0   # Events rejected for unstable depth before event
    twoway_baseline_rejected: int = 0 # Events rejected for depth returning to pre-event level


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
    max_event_distance_m: float = 5.0  # Only detect events when depth <= this (forklift must be close)

    frame_rate_hz: float = 10.0

    # Anti-false-positive filters (for transient occlusions like people walking through)
    # These filters are OFF by default - enable and tune for environments with pedestrian traffic
    #
    # Filter 1: Spike detection - reject PICK_UP→DROP_OFF pairs within short window
    # A person walking through creates a brief dip: PICK_UP (depth drops) → DROP_OFF (returns)
    spike_reject_window_frames: int = 30  # ~3 sec at 10Hz - reject if drop follows pick within this
    enable_spike_filter: bool = False     # OFF by default

    # Filter 2: Minimum dwell time - new depth must persist after transition
    # NOTE: Edge detection already handles this via window averaging
    min_dwell_frames: int = 10            # ~1 sec at 10Hz
    dwell_tolerance_m: float = 0.5        # More permissive - depth changes during transitions
    enable_dwell_filter: bool = False     # OFF by default

    # Filter 3: Depth variance check - reject high variance (curved person vs flat pallet)
    # NOTE: DROP_OFF events naturally have higher variance during transition
    max_event_variance_m2: float = 1.0    # Permissive threshold (pallet variance ~0.01-0.02)
    enable_variance_filter: bool = False  # OFF by default

    # Filter 4: Return-to-baseline - reject if depth returns to original
    # If depth returns to pre-event level after the event, likely a transient
    baseline_check_frames: int = 50       # Check this many frames after event (~5 sec)
    baseline_return_threshold_m: float = 0.2  # If depth returns within this of pre-event, reject
    enable_baseline_filter: bool = False  # OFF by default

    # Filter 5: Pre-event stability - reject if depth was unstable before event
    # Real events have stable depth beforehand; people suddenly appearing have no pre-stability
    pre_event_check_frames: int = 30      # Check stability this many frames before event (~3 sec)
    max_pre_event_variance_m2: float = 0.05  # Max variance before event (stable = low variance)
    min_nan_block_for_stability: int = 20  # Long NaN block = stable state (pallet blocking sensor)
    max_state_transitions: int = 3         # Max valid↔NaN transitions (more = flickering/unstable)
    enable_pre_stability_filter: bool = False  # OFF by default

    # Filter 6: Two-way baseline - reject if depth same before AND after event
    # Transient occlusions: depth_before ≈ depth_after (person walks through, depth returns)
    # Real pallets: depth_before ≠ depth_after (permanent change)
    # Uses state comparison: NaN vs valid, or valid depth values
    enable_twoway_baseline_filter: bool = False  # OFF by default

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
            "max_event_distance_m": self.max_event_distance_m,
            "frame_rate_hz": self.frame_rate_hz,
            # Anti-false-positive filter settings
            "spike_reject_window_frames": self.spike_reject_window_frames,
            "enable_spike_filter": self.enable_spike_filter,
            "min_dwell_frames": self.min_dwell_frames,
            "dwell_tolerance_m": self.dwell_tolerance_m,
            "enable_dwell_filter": self.enable_dwell_filter,
            "max_event_variance_m2": self.max_event_variance_m2,
            "enable_variance_filter": self.enable_variance_filter,
            "baseline_check_frames": self.baseline_check_frames,
            "baseline_return_threshold_m": self.baseline_return_threshold_m,
            "enable_baseline_filter": self.enable_baseline_filter,
            "pre_event_check_frames": self.pre_event_check_frames,
            "max_pre_event_variance_m2": self.max_pre_event_variance_m2,
            "min_nan_block_for_stability": self.min_nan_block_for_stability,
            "max_state_transitions": self.max_state_transitions,
            "enable_pre_stability_filter": self.enable_pre_stability_filter,
            "enable_twoway_baseline_filter": self.enable_twoway_baseline_filter,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DetectorConfig":
        """Load config from dictionary."""
        config = cls()
        if "roi" in d:
            config.roi_x, config.roi_y, config.roi_width, config.roi_height = d["roi"]

        for key in ["min_depth_m", "max_depth_m", "window_size", "min_edge_strength_m",
                    "smoothing_window", "min_event_gap_frames", "min_valid_pixels_pct",
                    "max_event_distance_m", "frame_rate_hz",
                    # Anti-false-positive filter settings
                    "spike_reject_window_frames", "enable_spike_filter",
                    "min_dwell_frames", "dwell_tolerance_m", "enable_dwell_filter",
                    "max_event_variance_m2", "enable_variance_filter",
                    "baseline_check_frames", "baseline_return_threshold_m", "enable_baseline_filter",
                    "pre_event_check_frames", "max_pre_event_variance_m2", "min_nan_block_for_stability",
                    "max_state_transitions", "enable_pre_stability_filter", "enable_twoway_baseline_filter"]:
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
    6. Apply anti-false-positive filters (spike, dwell, variance, baseline)
    """

    def __init__(self, config: DetectorConfig = None):
        self.config = config or DetectorConfig()
        self.filter_stats = FilterStats()  # Track what each filter rejected

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

    def process_batch(self, depth_frames: List[np.ndarray], return_diagnostics: bool = False) -> Tuple[List[float], List[PalletEvent], dict]:
        """
        Process a batch of depth frames and detect all events.

        Args:
            depth_frames: List of depth images (numpy arrays)
            return_diagnostics: If True, return edge signal and rejected candidates

        Returns:
            Tuple of (depth_history, detected_events, diagnostics_dict)
            diagnostics_dict contains: edge_signal, rejected_candidates, filter_stats
        """
        c = self.config
        n_frames = len(depth_frames)

        # Reset filter stats for this batch
        self.filter_stats = FilterStats()

        # Step 1: Extract median depth and variance for each frame
        depths = []
        variances = []
        for frame in depth_frames:
            stats = self.compute_frame_stats(frame)
            if stats.valid_pixel_pct >= c.min_valid_pixels_pct:
                depths.append(stats.median_depth_m)
                variances.append(stats.variance_m)
            else:
                depths.append(np.nan)
                variances.append(np.nan)

        depths = np.array(depths)
        variances = np.array(variances)

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
        events_before_filters = self._find_events(edge_signal, smoothed)

        # Step 6: Apply anti-false-positive filters
        events = self._apply_filters(events_before_filters, smoothed, variances)

        # Prepare diagnostics
        diagnostics = {}
        if return_diagnostics:
            # Rejected candidates = events that didn't pass filters
            rejected_candidates = [e for e in events_before_filters if e not in events]

            diagnostics = {
                'edge_signal': edge_signal.tolist(),
                'smoothed_depths': smoothed.tolist(),
                'rejected_candidates': rejected_candidates,
                'filter_stats': self.filter_stats
            }

        return depths.tolist(), events, diagnostics

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

            # Filter: only detect events when forklift is close
            if smoothed_depths[i] > c.max_event_distance_m:
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

    def _apply_filters(self, events: List[PalletEvent], smoothed_depths: np.ndarray,
                       variances: np.ndarray) -> List[PalletEvent]:
        """
        Apply anti-false-positive filters to detected events.

        Filters (applied in order):
        1. Spike filter: Reject PICK_UP→DROP_OFF pairs within short window
        2. Variance filter: Reject events with high depth variance (curved person)
        3. Dwell filter: Reject if new depth doesn't persist
        4. Baseline filter: Reject if depth returns to pre-event level
        5. Pre-stability filter: Reject if depth was unstable before event
        6. Two-way baseline filter: Reject if depth before ≈ depth after (transient)
        """
        if len(events) == 0:
            return events

        c = self.config
        n_frames = len(smoothed_depths)

        # Filter 1: Spike detection (PICK_UP→DROP_OFF pairs within short window)
        if c.enable_spike_filter:
            events = self._filter_spikes(events)

        # Filter 2: Variance check (reject high variance = curved person)
        if c.enable_variance_filter:
            events = self._filter_by_variance(events, variances)

        # Filter 3: Dwell time check (new depth must persist)
        if c.enable_dwell_filter:
            events = self._filter_by_dwell(events, smoothed_depths, n_frames)

        # Filter 4: Baseline return check (reject if depth returns to original)
        if c.enable_baseline_filter:
            events = self._filter_by_baseline_return(events, smoothed_depths, n_frames)

        # Filter 5: Pre-event stability check (reject if depth unstable before event)
        if c.enable_pre_stability_filter:
            events = self._filter_by_pre_stability(events, smoothed_depths, n_frames)

        # Filter 6: Two-way baseline check (reject if depth_before ≈ depth_after)
        if c.enable_twoway_baseline_filter:
            events = self._filter_by_twoway_baseline(events, smoothed_depths, n_frames)

        return events

    def _filter_spikes(self, events: List[PalletEvent]) -> List[PalletEvent]:
        """
        Filter 1: Reject PICK_UP→DROP_OFF pairs that occur within a short window.

        A person walking through creates: PICK_UP (depth drops) → DROP_OFF (depth returns)
        If these happen within spike_reject_window_frames, reject both.
        """
        if len(events) < 2:
            return events

        c = self.config
        filtered = []
        i = 0

        while i < len(events):
            event = events[i]

            # Look ahead to see if this is part of a spike
            if (i + 1 < len(events) and
                event.event_type == "PICK_UP" and
                events[i + 1].event_type == "DROP_OFF"):

                gap = events[i + 1].frame_idx - event.frame_idx

                if gap <= c.spike_reject_window_frames:
                    # This is a spike (quick PICK_UP→DROP_OFF) - reject both
                    self.filter_stats.spike_rejected += 2
                    i += 2  # Skip both events
                    continue

            filtered.append(event)
            i += 1

        return filtered

    def _filter_by_variance(self, events: List[PalletEvent],
                            variances: np.ndarray) -> List[PalletEvent]:
        """
        Filter 2: Reject events where depth variance is too high.

        A flat pallet has low variance, a curved person has high variance.
        Check variance in a window around the event.
        """
        c = self.config
        filtered = []
        w = c.window_size

        for event in events:
            idx = event.frame_idx
            # Get variance in window around event
            start = max(0, idx - w)
            end = min(len(variances), idx + w)
            window_variances = variances[start:end]
            valid_vars = window_variances[~np.isnan(window_variances)]

            if len(valid_vars) == 0:
                # No valid variance data, keep the event
                filtered.append(event)
                continue

            mean_variance = np.mean(valid_vars)

            if mean_variance <= c.max_event_variance_m2:
                filtered.append(event)
            else:
                self.filter_stats.variance_rejected += 1

        return filtered

    def _filter_by_dwell(self, events: List[PalletEvent],
                         smoothed_depths: np.ndarray, n_frames: int) -> List[PalletEvent]:
        """
        Filter 3: Reject events where the new depth doesn't persist (dwell).

        After an event, the depth should stay at the new level for min_dwell_frames.
        This rejects transient occlusions.
        """
        c = self.config
        filtered = []

        for event in events:
            idx = event.frame_idx
            event_depth = event.depth_m

            # Check if depth stays stable after event
            dwell_end = min(n_frames, idx + c.min_dwell_frames)
            if dwell_end <= idx:
                # Not enough frames after event, keep it
                filtered.append(event)
                continue

            post_depths = smoothed_depths[idx:dwell_end]
            valid_depths = post_depths[~np.isnan(post_depths)]

            if len(valid_depths) == 0:
                filtered.append(event)
                continue

            # Check if most frames stay within tolerance of event depth
            within_tolerance = np.abs(valid_depths - event_depth) <= c.dwell_tolerance_m
            dwell_ratio = np.sum(within_tolerance) / len(valid_depths)

            if dwell_ratio >= 0.7:  # At least 70% of frames within tolerance
                filtered.append(event)
            else:
                self.filter_stats.dwell_rejected += 1

        return filtered

    def _filter_by_baseline_return(self, events: List[PalletEvent],
                                   smoothed_depths: np.ndarray, n_frames: int) -> List[PalletEvent]:
        """
        Filter 4: Reject events where depth returns to pre-event baseline.

        If after an event the depth returns to what it was before, this was
        likely a transient occlusion (person walking through), not a real event.
        """
        c = self.config
        w = c.window_size
        filtered = []

        for event in events:
            idx = event.frame_idx

            # Get baseline (pre-event) depth
            baseline_start = max(0, idx - w - c.window_size)
            baseline_end = max(0, idx - w)
            if baseline_end <= baseline_start:
                filtered.append(event)
                continue

            baseline_depths = smoothed_depths[baseline_start:baseline_end]
            valid_baseline = baseline_depths[~np.isnan(baseline_depths)]

            if len(valid_baseline) == 0:
                filtered.append(event)
                continue

            pre_event_depth = np.mean(valid_baseline)

            # Check depth after the check window
            check_start = idx + c.baseline_check_frames
            check_end = min(n_frames, check_start + w)

            if check_end <= check_start:
                # Not enough frames to check, keep event
                filtered.append(event)
                continue

            post_depths = smoothed_depths[check_start:check_end]
            valid_post = post_depths[~np.isnan(post_depths)]

            if len(valid_post) == 0:
                filtered.append(event)
                continue

            post_event_depth = np.mean(valid_post)

            # If depth returned to baseline, reject the event
            returned_to_baseline = abs(post_event_depth - pre_event_depth) <= c.baseline_return_threshold_m

            if not returned_to_baseline:
                filtered.append(event)
            else:
                self.filter_stats.baseline_rejected += 1

        return filtered

    def _is_window_stable(self, window: np.ndarray, config) -> bool:
        """
        Check if a window represents a stable state.
        Stable = either low depth variance OR long NaN block.
        """
        nan_mask = np.isnan(window)
        nan_ratio = np.sum(nan_mask) / len(window)

        # Case 1: Mostly NaN (>70%)
        if nan_ratio > 0.7:
            # Check if it's a consistent NaN block (not flickering)
            nan_block_length = self._longest_consecutive_block(nan_mask)
            return nan_block_length >= config.min_nan_block_for_stability

        # Case 2: Mostly valid depth (>70%)
        elif nan_ratio < 0.3:
            valid_values = window[~nan_mask]
            if len(valid_values) < 5:
                return False
            variance = np.var(valid_values)
            return variance <= config.max_pre_event_variance_m2

        # Case 3: Mixed (30-70% NaN) - check for flickering
        else:
            # Count transitions between valid and NaN
            transitions = self._count_state_transitions(nan_mask)
            return transitions <= config.max_state_transitions

    def _longest_consecutive_block(self, mask: np.ndarray) -> int:
        """Find the longest consecutive True block in a boolean array."""
        if len(mask) == 0:
            return 0

        max_length = 0
        current_length = 0

        for val in mask:
            if val:
                current_length += 1
                max_length = max(max_length, current_length)
            else:
                current_length = 0

        return max_length

    def _count_state_transitions(self, mask: np.ndarray) -> int:
        """Count transitions between True and False in a boolean array."""
        if len(mask) <= 1:
            return 0

        transitions = 0
        for i in range(1, len(mask)):
            if mask[i] != mask[i-1]:
                transitions += 1

        return transitions

    def _filter_by_pre_stability(self, events: List[PalletEvent],
                                  smoothed_depths: np.ndarray, n_frames: int) -> List[PalletEvent]:
        """
        Filter 5: Reject events where depth was unstable before the event.

        Real pallet events have stable depth beforehand (empty floor, then forklift approaches).
        Transient occlusions (people) appear suddenly with no pre-stability.

        Considers both valid depth stability AND long NaN blocks as stable states.
        """
        c = self.config
        filtered = []

        for event in events:
            idx = event.frame_idx

            # Get pre-event window (avoid the edge detection window itself)
            pre_start = max(0, idx - c.pre_event_check_frames - c.window_size)
            pre_end = max(0, idx - c.window_size)

            if pre_end <= pre_start:
                # Not enough frames before event, keep it
                filtered.append(event)
                continue

            pre_window = smoothed_depths[pre_start:pre_end]

            # Check if pre-event state was stable (valid depth OR long NaN block)
            is_stable = self._is_window_stable(pre_window, c)

            if is_stable:
                filtered.append(event)
            else:
                self.filter_stats.pre_stability_rejected += 1

        return filtered

    def _classify_window_state(self, window: np.ndarray) -> tuple:
        """
        Classify window as NaN-state or valid-state with mean depth.
        Returns: (is_nan_state, mean_depth)
        """
        nan_mask = np.isnan(window)
        nan_ratio = np.sum(nan_mask) / len(window)

        # Mostly NaN (>70%) = NaN state
        if nan_ratio > 0.7:
            return (True, np.nan)

        # Mostly valid (>70%) = valid state with mean depth
        elif nan_ratio < 0.3:
            valid_values = window[~nan_mask]
            if len(valid_values) >= 5:
                return (False, np.mean(valid_values))

        # Mixed/uncertain - treat as valid if possible
        valid_values = window[~nan_mask]
        if len(valid_values) >= 5:
            return (False, np.mean(valid_values))

        return (True, np.nan)

    def _filter_by_twoway_baseline(self, events: List[PalletEvent],
                                    smoothed_depths: np.ndarray, n_frames: int) -> List[PalletEvent]:
        """
        Filter 6: Reject events where state before ≈ state after (transient occlusion).

        For real pallet events: state changes permanently (before ≠ after)
        For transient occlusions (people): state returns to original (before ≈ after)

        Compares states: NaN vs valid, or valid depth values.
        """
        c = self.config
        w = c.window_size
        filtered = []

        for event in events:
            idx = event.frame_idx

            # Get baseline BEFORE event (well before the edge window)
            before_start = max(0, idx - c.baseline_check_frames - w)
            before_end = max(0, idx - w - 10)  # 10 frame buffer from edge

            # Get baseline AFTER event (well after the edge window)
            after_start = min(n_frames, idx + w + 10)  # 10 frame buffer
            after_end = min(n_frames, idx + c.baseline_check_frames + w)

            # Need enough frames on both sides
            if before_end <= before_start or after_end <= after_start:
                filtered.append(event)
                continue

            before_window = smoothed_depths[before_start:before_end]
            after_window = smoothed_depths[after_start:after_end]

            # Classify states
            before_is_nan, before_depth = self._classify_window_state(before_window)
            after_is_nan, after_depth = self._classify_window_state(after_window)

            # Compare states
            state_returned = False

            if before_is_nan and after_is_nan:
                # Both NaN states → no change → transient
                state_returned = True

            elif not before_is_nan and not after_is_nan:
                # Both valid → compare depth values
                if abs(after_depth - before_depth) <= c.baseline_return_threshold_m:
                    state_returned = True

            # else: one NaN, one valid → state changed → real event

            if state_returned:
                self.filter_stats.twoway_baseline_rejected += 1
            else:
                filtered.append(event)

        return filtered


# Keep old class name for compatibility
PalletEventDetector = EdgeDetector
