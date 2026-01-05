# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MHE-V (Material Handling Equipment Vision) pallet pick/drop event detection using depth camera data. Runs on VOXL (Snapdragon) with PMD depth sensor at 10Hz.

**Status:** Algorithm under active development.

## Documentation

- **[WORKFLOW.md](WORKFLOW.md)** - Step-by-step guide for labeling, tuning, and debugging ⭐
- **[README.md](README.md)** - Project overview and quick start
- **[algorithm_design.md](algorithm_design.md)** - Algorithm design notes

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive tuner
python tuner.py bags/<depth_file>.mcap --rotate 90

# Load with existing config
python tuner.py bags/<file>.mcap --config <file>_config.json
```

## Tuner Controls

### Playback
- **Space**: Play/Pause (auto-pauses when labeling)
- **1/2/5/0**: Set playback speed (1x/2x/5x/10x)
- **Left/Right**: Step frame backward/forward
- **[ / ]**: Jump back/forward 50 frames
- **{ / }**: Jump back/forward 200 frames
- **, / .**: Jump to previous/next labeled event

### Labeling
- **p**: Mark PICK-UP (auto-finds nearest event at 5x+ speed)
- **P** (Shift+P): Mark PICK-UP manually (no auto-find)
- **d**: Mark DROP-OFF (auto-finds nearest event at 5x+ speed)
- **D** (Shift+D): Mark DROP-OFF manually (no auto-find)
- **Up/Down**: Move label backward/forward 1 frame
- **- / +**: Move label backward/forward 10 frames
- **c**: Clear label at current frame
- **C** (Shift+C): Clear ALL labels

### Other
- **i**: Show info about current frame and nearby labels
- **R** (Shift+R): Refresh display (fixes squished image)
- **Scroll**: Zoom in/out on timeline (when mouse over timeline)
- **Drag**: Pan timeline (click and drag on timeline)
- **z**: Reset timeline zoom
- **r**: Reset detector state
- **s**: Save config to JSON
- **q**: Quit
- **Mouse drag on depth image**: Select ROI

## Architecture

### detector.py
Edge detection algorithm with anti-false-positive filters:
- `EdgeDetector.process_batch()`: Processes all frames at once, returns diagnostics
- Filters spurious depth readings (brief valid readings between NaNs)
- Computes "edge signal" by comparing before/after windows
- Finds peaks in edge signal that exceed threshold
- Applies 6 optional anti-false-positive filters
- Tracks which filters each event passed/failed
- Classifies: negative edge = PICK_UP, positive edge = DROP_OFF

### tuner.py
Interactive matplotlib GUI with diagnostic visualization:
- `DepthMcapReader`: Loads ROS2 depth images from MCAP, supports rotation
- `InteractiveTuner`: Compact text input parameters (22 params), ROI selection, event labeling
- **3-plot layout**: Depth image, depth timeline, edge signal plot
- Shows edge signal with threshold lines and rejection annotations
- Displays per-event filter results (✓/✗/-) for detected events and ground truth labels
- Real-time precision/recall metrics as you adjust parameters
- Auto-find nearest event when labeling at high speeds
- Timeline zoom/pan support (scroll to zoom, drag to pan)
- Labels saved to `*_labels.json`, config to `*_config.json`

## Current Algorithm: Edge Detection with Anti-FP Filters

```
1. Extract median depth from ROI for each frame
2. Filter spurious readings (remove brief valid readings between NaNs)
3. Interpolate NaN values for continuity
4. Smooth depth signal (moving average)
5. Compute edge signal: after_window_avg - before_window_avg
6. Find local maxima/minima in edge signal
7. Apply filters:
   - Edge detection: Minimum edge strength, minimum gap between events
   - Spike filter (optional): Reject PICK_UP→DROP_OFF pairs within short window
   - Variance filter (optional): Reject high variance (curved person vs flat pallet)
   - Dwell filter (optional): Reject if new depth doesn't persist
   - Baseline filter (optional): Reject if depth returns to pre-event level
   - Pre-stability filter (optional): Reject if depth was unstable before event
   - Two-way baseline filter (optional): Reject if depth same before AND after
8. Classify:
   - Negative edge (depth decreased) → PICK_UP
   - Positive edge (depth increased) → DROP_OFF
9. Track which filters each event passed/failed for diagnostics
```

### Core Parameters
```
min_edge_strength_m: 0.20           # Minimum depth change to trigger
window_size: 10                     # Frames for before/after comparison (1 sec)
smoothing_window: 5                 # Smoothing before edge detection
min_event_gap_frames: 80            # Minimum 8 sec between events
min_valid_pixels_pct: 10.0          # Minimum valid pixels in ROI
max_event_distance_m: 5.0           # Only detect when forklift is close
min_consecutive_valid_frames: 5     # Remove valid readings shorter than this
min_consecutive_nan_frames: 5       # Remove NaN blocks shorter than this
```

### Anti-False-Positive Filters (All OFF by default)
Enable and tune for environments with pedestrian traffic:
```
enable_spike_filter: false          # Reject PICK_UP→DROP_OFF pairs
spike_reject_window_frames: 30      # ~3 sec window

enable_variance_filter: false       # Reject high variance events
max_event_variance_m2: 1.0          # Variance threshold

enable_dwell_filter: false          # Require new depth to persist
min_dwell_frames: 10                # ~1 sec dwell time

enable_baseline_filter: false       # Reject if returns to baseline
baseline_check_frames: 50           # Check ~5 sec after event

enable_pre_stability_filter: false  # Require stable depth before
pre_event_check_frames: 30          # Check ~3 sec before

enable_twoway_baseline_filter: false # Reject transient occlusions
```

### Current Performance (on test data)
- Precision: ~90%
- Recall: ~100%
- F1: ~0.95
- With filters enabled: Fewer false positives in pedestrian environments

## Data Format

- MCAP files with `/depth` topic, `sensor_msgs/Image` (16UC1 encoding, mm)
- PMD sensor: 180x240 pixels after 90° rotation
- Floor appears at bottom of rotated image

## Algorithm Design Notes

See `algorithm_design.md` for detailed notes on different approaches considered:
- Threshold-based (simple but noisy)
- State machine (complex, baseline tracking issues)
- Edge detection (current - simple and effective)
- Hybrid approach (proposed for future improvement)
