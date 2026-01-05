# MHE-V Event Detection Workflow Guide

Step-by-step guide for labeling ground truth, tuning parameters, and debugging detection issues.

---

## 1. Initial Setup

### Load Data
```bash
python tuner.py bags/<depth_file>.mcap --rotate 90
```

### Set ROI (Region of Interest)
1. Click and drag on the **depth image** to select the area where pallets appear
2. ROI should cover the fork tips area
3. ROI is saved automatically with config

### Initial Parameters
Start with default parameters:
- `EdgeStr`: 0.20m
- `Window`: 10 frames
- `Smooth`: 5 frames
- All filters: OFF (disabled)

---

## 2. Labeling Ground Truth

### Quick Labeling Workflow

**Goal:** Mark all real PICK-UP and DROP-OFF events in the data.

1. **Play at 5x-10x speed** to scan through data quickly
   - Press `5` for 5x speed, `0` for 10x speed
   - Press `Space` to pause when you see an event

2. **Use auto-find for labeling**
   - When at 5x+ speed, press `p` or `d` to auto-find nearest event
   - Tuner will automatically find the strongest edge near current frame
   - Label is placed at the detected event peak

3. **Fine-tune label position**
   - Use `Up/Down` arrows to move label ±1 frame
   - Use `-/+` to move label ±10 frames
   - Watch the edge signal plot to see event strength

4. **Manual labeling** (when auto-find isn't working)
   - Press `P` (Shift+P) or `D` (Shift+D) for manual placement
   - Places label at exact current frame
   - Use when edge is weak or detector isn't configured yet

### Jumping Between Events

**Navigate labeled events:**
- **`,` (comma)**: Jump to **previous** labeled event
- **`.` (period)**: Jump to **next** labeled event
- These skip through your ground truth labels quickly

**Navigate detected events:**
- Use `.` repeatedly at 5x speed to jump through detections
- Compare with your labels to check accuracy

**Quick navigation:**
- **`[` / `]`**: Jump ±50 frames (5 seconds)
- **`{` / `}`**: Jump ±200 frames (20 seconds)
- **Left/Right arrows**: Step ±1 frame (precise positioning)

### Reviewing Labels

1. Press `,` to jump to first labeled event
2. Check if detection (dashed line) aligns with label (triangle)
3. Press `.` to jump to next label
4. Repeat until all labels reviewed

**Status panel shows:**
```
Frame: 1234 [LABEL: PICK_UP] [PRED: PICK_UP]
```

---

## 3. Parameter Tuning

### Check Current Performance

Look at status panel:
```
=== Predictions vs Labels ===
Predictions: 45 | Labels: 50
True Pos: 40 | False Pos: 5 | Missed: 10
Precision: 89% | Recall: 80%
```

**Precision low?** → Too many false positives → Increase `EdgeStr` or enable filters
**Recall low?** → Missing real events → Decrease `EdgeStr` or adjust `Window`

### Core Parameter Tuning

#### Edge Strength (`EdgeStr`)
- **Too high**: Misses subtle events (low recall)
- **Too low**: Detects noise as events (low precision)
- Start at 0.20m, adjust in 0.05m increments
- **Use edge signal plot** to see event magnitudes

#### Window Size (`Window`)
- Larger window = smoother detection, slower response
- Smaller window = more sensitive, more noise
- Typical: 5-15 frames (0.5-1.5 seconds)

#### Smoothing (`Smooth`)
- More smoothing = fewer noise spikes
- Less smoothing = faster response
- Typical: 3-7 frames

### Using Edge Signal Plot

The **middle plot** shows why events were detected:

1. **Purple line**: Edge signal (after - before depth)
2. **Red dashed lines**: Edge strength threshold (±EdgeStr)
3. **Green/Orange vertical lines**: Detected events
4. **Numbers on events**: Edge strength value (e.g., "0.35m")
5. **Gray X markers**: Rejected candidates with reason

**Example interpretation:**
```
Edge signal crosses threshold at frame 1234 with 0.35m
→ Event detected (above 0.20m threshold)
→ All filters passed (✓✓✓)
→ Classified as PICK_UP (negative edge)
```

---

## 4. Debugging False Positives

### Identify False Positives

Look for predictions (dashed lines) **without** nearby labels (triangles).

**Navigate to false positive:**
1. Play at 2x-5x speed
2. Press `.` to jump to next detected event
3. Check if there's a label nearby
4. If no label → false positive

### Understand Why It Was Detected

**Check edge signal plot:**
- Is the edge strong enough to cross threshold?
- Was it a real depth change or noise?

**Check filter status:**
```
=== Detected Event Filters ===
Edge: ✓ | Spike: ✓ | Var: ✓ | Dwell: - | Base: - | PreStab: - | 2Way: -
```
- `✓` = Filter passed
- `-` = Filter not enabled
- All filters passed → event wasn't filtered out

### Enable Appropriate Filters

**For pedestrian traffic (people walking through):**

1. Enable **Spike filter**:
   - Set `Spike: 1` (enable)
   - Catches PICK_UP→DROP_OFF pairs within 3 seconds
   - Most effective anti-FP filter

2. Enable **Pre-Stability filter**:
   - Set `PreStab: 1` (enable)
   - Rejects events with no stable depth beforehand
   - Catches sudden appearances

3. Enable **Two-Way Baseline filter**:
   - Set `2WayBase: 1` (enable)
   - Rejects if depth returns to original level
   - Catches transient occlusions

**After enabling filters:**
- Check **Overall Filter Stats**:
  ```
  Total Rejected: 12
    Spike: 8 | Dwell: 0
    Variance: 0 | Baseline: 0
    Pre-Stab: 4 | 2Way: 0
  ```
- Spike filter caught 8 false positives!

### Verify Fix

1. Jump back to the false positive (use `{` to jump back)
2. Check if it's now rejected (gray X marker)
3. Look for rejection reason annotation (e.g., "SPIK")

---

## 5. Debugging Missed Detections

### Identify Missed Events

Look for labels (triangles) **without** nearby predictions (dashed lines).

**Navigate to missed event:**
1. Press `,` to jump to labeled events
2. Check if there's a prediction (dashed line) within ±50 frames
3. If no prediction nearby → missed detection

### Understand Why It Was Missed

Status panel shows diagnostics for ground truth labels:

**Case 1: Event detected nearby**
```
=== Ground Truth Label ===
Nearest Detection: +3 frames away
Edge: ✓ | Spike: ✓ | Var: ✓ | Dwell: - | Base: - | PreStab: - | 2Way: -
```
→ Detection is 3 frames off, but all filters passed
→ Event WAS detected, just slightly offset
→ This is acceptable (within tolerance)

**Case 2: Event was rejected**
```
=== Ground Truth Label ===
Event REJECTED (-5 frames) - Failed: VARIANCE
Edge: ✓ | Spike: ✓ | Var: ✗ | Dwell: ✓ | Base: - | PreStab: - | 2Way: -
```
→ Event WAS detected at frame -5
→ BUT variance filter rejected it
→ Look for gray X marker on edge signal plot
→ **Solution**: Disable variance filter or increase `MaxVar` threshold

**Case 3: No event detected at all**
```
=== Ground Truth Label ===
NO EVENT DETECTED within 50 frames
Possible reasons: edge too weak, or all filters rejected it
```
→ Edge signal didn't cross threshold
→ Check edge signal plot - is there a visible edge?
→ **Solutions**:
  - Decrease `EdgeStr` threshold
  - Decrease `Window` size (more sensitive)
  - Check if spurious filter removed valid readings (`MinValid` too high)

### Using Edge Signal for Debugging

1. Navigate to missed event (press `,` or `.`)
2. Look at **edge signal plot** (middle plot)
3. Is there a visible edge near the label?
   - **Yes, strong edge**: Check filter stats (something rejected it)
   - **Yes, weak edge**: Decrease `EdgeStr` or adjust `Window`
   - **No visible edge**: Data issue or ROI problem

### Tune Filters to Reduce Rejections

If filters are rejecting too many real events:

- **Variance filter too strict**: Increase `MaxVar` from 1.0 to 2.0
- **Dwell filter too strict**: Decrease `DwellFr` or increase `DwellTol`
- **Baseline filter too strict**: Increase `BaseThr` threshold
- **Pre-stability too strict**: Increase `PreVar` or decrease `PreStabFr`

**Best practice**: Start with all filters OFF, enable only when needed.

---

## 6. Advanced Debugging

### Inspect Rejected Candidates

**Gray X markers** on edge signal plot show rejected events:

1. Navigate to the X marker
2. Read the annotation (e.g., "SPIK", "DWELL", "VARI")
3. This shows which filter rejected it
4. Check if it was a real event or false positive
5. Adjust filter parameters if needed

### Filter Rejection Codes

- **EDGE**: Below edge strength threshold
- **SPIK**: Spike filter (paired with opposite event)
- **VARI**: Variance filter (high depth variance)
- **DWEL**: Dwell filter (depth didn't persist)
- **BASE**: Baseline filter (depth returned to original)
- **PRE_**: Pre-stability filter (unstable before event)
- **TWOW**: Two-way baseline filter (transient occlusion)

### Timeline Zoom/Pan

For detailed inspection:

1. **Scroll** over timeline to zoom in
2. **Drag** timeline to pan left/right
3. **Press `z`** to reset zoom
4. Zoomed view shows edge signal details more clearly

### Info Display

Press **`i`** to show detailed frame info:
- Current frame number
- Depth value
- Nearby labels (within ±50 frames)
- Nearby detections
- Filter results

---

## 7. Spurious Reading Filter

### When to Use

If you see brief "flickers" in depth timeline:
- Valid depth readings for 1-4 frames between NaN blocks
- These create false PICK_UP/DROP_OFF pairs

### Parameters

- **MinValid**: Minimum consecutive valid frames to keep (default: 5)
- **MinNaN**: Minimum consecutive NaN frames to keep (default: 5)

**Example:**
```
Before filtering: NaN NaN [2.0m 2.1m] NaN NaN → false event
After filtering:  NaN NaN [NaN NaN] NaN NaN → no event
```

### Adjusting

- **Too aggressive**: Increase `MinValid` (e.g., 10 frames = 1 second)
- **Too permissive**: Decrease `MinValid` (e.g., 3 frames = 0.3 seconds)
- Set `MinValid: 0` to disable

---

## 8. Saving and Deployment

### Save Configuration

Press **`s`** to save:
- `<filename>_config.json` - All parameters and filter settings
- `<filename>_labels.json` - All ground truth labels

### Load Configuration

```bash
python tuner.py bags/<file>.mcap --config <file>_config.json
```

### Deploy to Production

1. Verify performance: Precision ≥90%, Recall ≥90%
2. Copy `config.json` to production system
3. Use same ROI and parameters in deployment code

---

## Quick Reference

### Navigation Shortcuts

| Action | Key |
|--------|-----|
| Play/Pause | `Space` |
| Speed: 1x/2x/5x/10x | `1` / `2` / `5` / `0` |
| Step frame | `←` / `→` |
| Jump ±5 sec | `[` / `]` |
| Jump ±20 sec | `{` / `}` |
| Previous label | `,` |
| Next label | `.` |
| Frame info | `i` |

### Labeling Shortcuts

| Action | Key |
|--------|-----|
| Pick-up (auto) | `p` |
| Pick-up (manual) | `P` (Shift+P) |
| Drop-off (auto) | `d` |
| Drop-off (manual) | `D` (Shift+D) |
| Move label ±1 frame | `↑` / `↓` |
| Move label ±10 frames | `-` / `+` |
| Clear label | `c` |
| Clear all | `C` (Shift+C) |

### Workflow Checklist

- [ ] Load MCAP file and set ROI
- [ ] Play through at 5x-10x speed
- [ ] Label all PICK_UP events (press `p`)
- [ ] Label all DROP-OFF events (press `d`)
- [ ] Review labels with `,` and `.`
- [ ] Check precision/recall in status panel
- [ ] Tune `EdgeStr` for balance
- [ ] Enable filters if false positives exist
- [ ] Debug missed detections using edge signal
- [ ] Verify all filters with gray X markers
- [ ] Press `s` to save config and labels
- [ ] Test on additional data files
