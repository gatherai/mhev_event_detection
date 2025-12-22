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

## Approach 2: Change-Based (Current Implementation)

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

## Approach 3: Hybrid (Proposed)

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
