# MHE-V Pallet Event Detection

Automated pallet pick-up and drop-off detection using depth camera data for material handling equipment (forklifts).

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run interactive tuner
python tuner.py bags/<depth_file>.mcap --rotate 90

# Load with existing configuration
python tuner.py bags/<file>.mcap --config <file>_config.json
```

## Documentation

- **[WORKFLOW.md](WORKFLOW.md)** - Step-by-step guide for labeling, tuning, and debugging
- **[CLAUDE.md](CLAUDE.md)** - Technical reference for Claude Code (architecture, parameters, controls)
- **[algorithm_design.md](algorithm_design.md)** - Algorithm design notes and filter descriptions

## Overview

### Algorithm
Edge detection on depth signal with optional anti-false-positive filters:
1. Extract median depth from ROI
2. Filter spurious readings (brief valid readings between NaNs)
3. Smooth depth signal
4. Compute edge signal (after - before comparison)
5. Find peaks exceeding threshold
6. Apply optional filters (spike, variance, dwell, baseline, etc.)
7. Classify: negative edge = PICK_UP, positive edge = DROP_OFF

### Performance
- **Precision**: ~90%
- **Recall**: ~100%
- **F1**: ~0.95

### Features
- **Interactive tuner** with real-time parameter adjustment
- **3-plot visualization**: Depth image, depth timeline, edge signal
- **Filter diagnostics**: See which filters each event passed/failed (✓/✗/-)
- **Ground truth labeling** with auto-find and manual modes
- **Rejection annotations**: Gray X markers show why events were rejected

## Getting Started

1. **Read the [Workflow Guide](WORKFLOW.md)** for step-by-step instructions
2. Load your MCAP file in the tuner
3. Label ground truth events
4. Tune parameters to maximize precision/recall
5. Save configuration for deployment

## Key Files

- `detector.py` - Core edge detection algorithm with filters
- `tuner.py` - Interactive matplotlib GUI for labeling and tuning
- `requirements.txt` - Python dependencies

## Hardware

- **Sensor**: PMD ToF depth camera (180x240 pixels, 10Hz)
- **Platform**: VOXL (Snapdragon processor)
- **Mount**: Forward-facing, viewing fork tips
- **Data format**: ROS2 MCAP files with `/depth` topic

## Repository

https://github.com/gatherai/mhev_event_detection
