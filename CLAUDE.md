# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MHE-V (Material Handling Equipment Vision) pallet pick/drop event detection using depth camera data. Runs on VOXL (Snapdragon) with PMD depth sensor at 10Hz.

**Status:** Algorithm under active development.

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

- **Space**: Play/Pause
- **Left/Right arrows**: Step frame
- **p**: Mark current frame as PICK-UP event
- **d**: Mark current frame as DROP-OFF event
- **c**: Clear label at current frame
- **s**: Save config to JSON
- **q**: Quit
- **Mouse drag on image**: Select ROI

## Architecture

### detector.py
Edge detection algorithm (current approach):
- `EdgeDetector.process_batch()`: Processes all frames at once
- Computes "edge signal" by comparing before/after windows
- Finds peaks in edge signal that exceed threshold
- Classifies: negative edge = PICK_UP, positive edge = DROP_OFF

### tuner.py
Interactive matplotlib GUI:
- `DepthMcapReader`: Loads ROS2 depth images from MCAP, supports rotation
- `InteractiveTuner`: Parameter sliders, ROI selection, event labeling
- Shows precision/recall metrics in real-time as you adjust parameters
- Labels saved to `*_labels.json`, config to `*_config.json`

## Current Algorithm: Edge Detection

```
1. Extract median depth from ROI for each frame
2. Smooth depth signal (moving average)
3. Compute edge signal: after_window_avg - before_window_avg
4. Find local maxima/minima in edge signal
5. Filter by:
   - Minimum edge strength (depth change threshold)
   - Minimum gap between events
6. Classify:
   - Negative edge (depth decreased) → PICK_UP
   - Positive edge (depth increased) → DROP_OFF
```

### Current Best Parameters
```
min_edge_strength_m: 0.20    # Minimum depth change to trigger
window_size: 10              # Frames for before/after comparison (1 sec)
smoothing_window: 5          # Smoothing before edge detection
min_event_gap_frames: 80     # Minimum 8 sec between events
min_valid_pixels_pct: 10.0   # Minimum valid pixels in ROI
```

### Current Performance (on test data)
- Precision: 90%
- Recall: 100%
- F1: 0.95

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
