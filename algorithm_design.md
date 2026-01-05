# MHE-V Pallet Detection Algorithm Design

## Problem Statement

Detect when a forklift picks up or drops off a pallet using a forward-facing depth camera mounted to view the fork tips.

**Constraints:**
- PMD depth sensor at 10Hz, 180x240 pixels
- Runs on Snapdragon processor (compute-limited)
- Must reject false positives from people walking through
- Must handle noisy depth data with 20-80% valid pixels

---

## Approach 1: Threshold-Based (Simple)

```
IF depth < THRESHOLD → pallet present
IF depth > THRESHOLD → no pallet
State change = event
```

**Pros:**
- Simple, fast
- Easy to understand

**Cons:**
- Many false positives when depth fluctuates around threshold
- Sensitive to threshold value
- Triggers on any object crossing threshold (people, other forklifts)

---

## Approach 2: State Machine (Attempted)

```
1. Track BASELINE depth when signal is stable
2. Detect when depth changes significantly from BASELINE
3. Wait for new depth to stabilize
4. Classify based on change direction:
   - Depth decreased → PICK_UP
   - Depth increased → DROP_OFF
5. Update BASELINE to new stable depth
```

**Pros:**
- Ignores gradual drift
- Requires stability (rejects transient objects)

**Cons:**
- Baseline updates continuously, can miss slow events
- Misses events where depth is noisy before/after
- No absolute reference - relies entirely on relative change

**Performance on test data:**
- Precision: 67%
- Recall: 44%

---

## Approach 3: Edge Detection (Current Implementation)

Sliding window edge detection - simple and highly effective.

### Algorithm

```
1. Extract median depth from ROI for each frame
2. Smooth depth signal (moving average)
3. Compute edge signal at each frame:
   edge[i] = mean(depth[i:i+window]) - mean(depth[i-window:i])
   (difference between "after" and "before" windows)
4. Find local maxima/minima in edge signal that exceed threshold
5. Classify:
   - Negative edge (depth decreased) → PICK_UP
   - Positive edge (depth increased) → DROP_OFF
6. Post-process: merge consecutive same-type events (keep last)
```

### Key Insight

This is essentially an **edge detector** on the depth signal. It finds step changes by comparing the average depth before vs after each point.

### Post-Processing: Duplicate Removal

**Problem:** Sometimes two events of the same type are detected close together (e.g., forklift approaching then engaging pallet = two PICK_UP edges).

**Solution:** Merge consecutive same-type events:
- If two PICK_UPs (or two DROP_OFFs) occur within `max_merge_gap` frames
- Keep the **last** one (the final action is the real event)
- If they're far apart, keep both (separate real events)

```python
def merge_consecutive_same_type(events):
    merged = [events[0]]
    for event in events[1:]:
        if event.type == merged[-1].type and gap < max_merge_gap:
            merged[-1] = event  # Keep last
        else:
            merged.append(event)
    return merged
```

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `min_edge_strength_m` | 0.20 | Minimum depth change to trigger |
| `window_size` | 10 | Frames for before/after comparison (1 sec at 10Hz) |
| `smoothing_window` | 5 | Frames for smoothing before edge detection |
| `min_event_gap_frames` | 80 | Minimum frames between events (8 sec) |
| `max_merge_gap` | 400 | Max frames to merge same-type events (40 sec) |

### Performance

| Metric | Value |
|--------|-------|
| Precision | **100%** |
| Recall | **100%** |
| F1 Score | **1.0** |

### Why It Works

1. **Sliding window comparison** is robust to noise (averages multiple frames)
2. **Edge detection** finds step changes regardless of absolute depth values
3. **Smoothing** reduces false edges from frame-to-frame noise
4. **Same-type merging** eliminates duplicates (can't pick up twice without dropping)
5. **Keep last** strategy captures the final action (actual engagement, not approach)

### Spurious Reading Filter (Pre-Processing)

**Added:** Filters out brief valid depth readings between NaN blocks before edge detection.

**Problem:** NaN is signal (often means pallet blocking sensor). Brief valid depth readings (1-4 frames) between NaN blocks create false events.

**Solution:**
```python
# Remove valid depth blocks shorter than threshold
if valid_block_length < min_consecutive_valid_frames:
    replace_with_nan(valid_block)

# Optionally interpolate brief NaN gaps between valid readings
if nan_block_length < min_consecutive_nan_frames:
    interpolate_across_gap(nan_block)
```

**Parameters:**
```
min_consecutive_valid_frames: 5  # Remove valid readings shorter than 0.5 sec
min_consecutive_nan_frames: 5    # Fill NaN gaps shorter than 0.5 sec
```

**Benefits:**
- Eliminates false events from sensor noise/glitches
- Preserves NaN signal (pallet blocking view)
- Only processes sustained depth changes (real events)

### Anti-False-Positive Filters (Optional)

For environments with pedestrian traffic (people walking in front of camera), additional filters are available. These are **OFF by default** since edge detection already handles most cases well.

All filters track per-event pass/fail status for diagnostics.

#### Filter 1: Spike Detection
Rejects PICK_UP→DROP_OFF pairs that occur within a short window.

**Rationale:** A person walking through creates:
1. PICK_UP event (depth drops as person enters)
2. DROP_OFF event (depth returns as person leaves)

If these happen within ~3 seconds, reject both as transient occlusion.

```
enable_spike_filter: false
spike_reject_window_frames: 30  # ~3 sec at 10Hz
```

#### Filter 2: Variance Check
Rejects events with high depth variance in the ROI.

**Rationale:** Flat pallet surface has low variance (~0.01-0.02 m²). Curved person has higher variance.

**Note:** Less reliable because DROP_OFF events naturally have higher variance during transition.

```
enable_variance_filter: false
max_event_variance_m2: 1.0  # Permissive to avoid false rejections
```

#### Filter 3: Dwell Time Check
Requires depth to stay stable after event.

**Note:** Less effective with edge detection since the edge point is in the middle of the transition, not at the stable end state.

```
enable_dwell_filter: false
min_dwell_frames: 10        # ~1 sec at 10Hz
dwell_tolerance_m: 0.5      # Permissive tolerance
```

#### Filter 4: Baseline Return Check
Rejects events where depth returns to pre-event level after the event.

**Rationale:** Real pick-up/drop-off changes the depth permanently. If depth returns to original, it was a transient occlusion.

```
enable_baseline_filter: false
baseline_check_frames: 50        # Check 5 sec after event
baseline_return_threshold_m: 0.2 # If within 0.2m of original, reject
```

#### Filter 5: Pre-Stability Check (NEW)
Rejects events where depth was unstable before the event.

**Rationale:** Real pallet events have stable depth beforehand (empty floor, then forklift approaches). People appearing suddenly have no pre-stability.

Considers both valid depth stability AND long NaN blocks as stable states.

```
enable_pre_stability_filter: false
pre_event_check_frames: 30          # Check ~3 sec before event
max_pre_event_variance_m2: 0.05     # Max variance for stability
min_nan_block_for_stability: 20     # Long NaN = stable state
max_state_transitions: 3            # Max valid↔NaN transitions
```

#### Filter 6: Two-Way Baseline Check (NEW)
Rejects events where state before ≈ state after (transient occlusion).

**Rationale:**
- Real pallet events: state changes permanently (before ≠ after)
- Transient occlusions: state returns to original (before ≈ after)

Compares states using both NaN vs valid and depth values.

```
enable_twoway_baseline_filter: false
baseline_check_frames: 50            # Check window size
baseline_return_threshold_m: 0.2     # Tolerance for "same depth"
```

#### Recommended Filter Configuration

For environments with pedestrian traffic:

| Filter | Recommended | Notes |
|--------|-------------|-------|
| Spike | **Enable** | Most effective for person walk-through |
| Pre-Stability | **Enable** | Catches sudden appearances |
| Two-Way Baseline | **Enable** | Robust transient check |
| Baseline Return | Enable | Good secondary check |
| Variance | Disable | High false rejection rate |
| Dwell | Disable | Not compatible with edge detection |

---

## Approach 4: Hybrid (Proposed for Future)

Combine threshold zones with change detection.

### Depth Zones

Define three zones based on typical depths observed:

```
CLOSE_ZONE:  depth < 1.0m   (pallet surface visible)
MIDDLE_ZONE: 1.0m ≤ depth < 1.5m  (ambiguous)
FAR_ZONE:    depth ≥ 1.5m   (empty forks / distant objects)
```

### State Machine

```
States:
  EMPTY      - Forks are empty (depth in FAR_ZONE)
  LOADED     - Pallet on forks (depth in CLOSE_ZONE)
  TRANSITION - Depth changing between zones

Transitions:
  EMPTY → TRANSITION:
    - Depth enters CLOSE_ZONE or MIDDLE_ZONE
    - Start tracking for potential PICK_UP

  TRANSITION → LOADED (emit PICK_UP):
    - Depth stabilizes in CLOSE_ZONE
    - AND depth change from pre-transition > min_change (0.25m)
    - AND stable for min_duration (1.5s)

  TRANSITION → EMPTY (abort):
    - Depth returns to FAR_ZONE
    - OR timeout exceeded
    - (This was a transient object, not a pick-up)

  LOADED → TRANSITION:
    - Depth leaves CLOSE_ZONE
    - Start tracking for potential DROP_OFF

  TRANSITION → EMPTY (emit DROP_OFF):
    - Depth stabilizes in FAR_ZONE
    - AND depth change from pre-transition > min_change (0.25m)
    - AND stable for min_duration (1.5s)

  TRANSITION → LOADED (abort):
    - Depth returns to CLOSE_ZONE
    - (False alarm, still have pallet)
```

### Detection Logic

```python
def process_frame(depth):
    zone = get_zone(depth)  # CLOSE, MIDDLE, or FAR

    if state == EMPTY:
        if zone in [CLOSE, MIDDLE]:
            state = TRANSITION
            transition_start_depth = baseline_depth
            transition_start_time = now

    elif state == LOADED:
        if zone in [MIDDLE, FAR]:
            state = TRANSITION
            transition_start_depth = baseline_depth
            transition_start_time = now

    elif state == TRANSITION:
        if is_stable(depth):
            stable_frames += 1
        else:
            stable_frames = 0

        if stable_frames >= min_stable_frames:
            depth_change = abs(depth - transition_start_depth)

            if depth_change >= min_depth_change:
                if zone == CLOSE:
                    emit("PICK_UP")
                    state = LOADED
                elif zone == FAR:
                    emit("DROP_OFF")
                    state = EMPTY
            else:
                # Change too small, return to previous state
                state = previous_state

        elif (now - transition_start_time) > timeout:
            # Timeout, return to appropriate state based on current zone
            state = LOADED if zone == CLOSE else EMPTY

    # Update baseline when stable
    if state != TRANSITION and is_stable(depth):
        baseline_depth = depth
```

### Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `close_threshold_m` | 1.0 | Below this = CLOSE_ZONE |
| `far_threshold_m` | 1.5 | Above this = FAR_ZONE |
| `min_depth_change_m` | 0.25 | Minimum change to confirm event |
| `stability_threshold_m` | 0.12 | Max std dev to be "stable" |
| `stable_duration_s` | 1.5 | Seconds of stability to confirm |
| `transition_timeout_s` | 8.0 | Max time in transition state |

### Why This Should Work Better

1. **Absolute zones** provide reference points (unlike pure change-based)
2. **Change requirement** filters out small fluctuations (unlike pure threshold)
3. **Stability requirement** rejects transient objects (people walking through)
4. **Timeout** prevents stuck states
5. **Hysteresis** (different thresholds for CLOSE vs FAR) prevents oscillation

### Expected Improvements

| Metric | Change-Only | Hybrid (Expected) |
|--------|-------------|-------------------|
| Precision | 67% | 80%+ |
| Recall | 44% | 70%+ |

The hybrid approach should:
- Catch more DROP_OFF events (using absolute FAR_ZONE threshold)
- Reduce false positives (requiring both zone change AND significant depth change)
- Handle noisy data better (zone-based is more robust than pure baseline tracking)

---

## Implementation Notes

### Tunable Parameters for UI

Add sliders for:
- `close_threshold_m` (0.5 - 1.5)
- `far_threshold_m` (1.0 - 3.0)
- `min_depth_change_m` (0.1 - 0.5)
- `stability_threshold_m` (0.05 - 0.3)
- `stable_duration_s` (0.5 - 3.0)

### Visualization

Show on timeline:
- Horizontal lines at close_threshold and far_threshold
- Color-coded background for zones
- Predicted events as vertical lines
- Labels as triangle markers

### Calibration Process

1. Load recording with labeled events
2. Adjust zone thresholds to separate "pallet" vs "no pallet" depths
3. Adjust stability/change thresholds to maximize precision/recall
4. Save config for production deployment
