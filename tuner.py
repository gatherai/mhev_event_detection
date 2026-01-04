#!/usr/bin/env python3
"""
MHE-V (Material Handling Equipment Vision) interactive tuning tool.
Replay MCAP files and adjust pallet pick/drop detection parameters in real-time.

Usage:
    python tuner.py bags/your_depth_file.mcap
    python tuner.py bags/your_depth_file.mcap --rotate 90  # Floor at bottom

Controls:
    Space      - Play/Pause (auto-pauses when labeling)
    1/2/5/0    - Set playback speed (1x/2x/5x/10x)
    Left/Right - Step frame backward/forward (or move label if no auto-find)
    [ / ]      - Jump back/forward 50 frames (quick rewind)
    { / }      - Jump back/forward 200 frames (big rewind)
    , / .      - Jump to previous/next labeled event

    Labeling:
    p          - Mark PICK-UP (auto-finds nearest event at 5x+ speed)
    P          - Mark PICK-UP manually (Shift+P, no auto-find)
    d          - Mark DROP-OFF (auto-finds nearest event at 5x+ speed)
    D          - Mark DROP-OFF manually (Shift+D, no auto-find)
    Up/Down    - Move label backward/forward 1 frame
    - / +      - Move label backward/forward 10 frames (Mac-friendly)
    c          - Clear label at current frame (exact frame match)
    C          - Clear ALL labels (Shift+C)

    Other:
    i          - Show info about current frame and nearby labels
    R          - Refresh display (fixes squished image) (Shift+R)
    Scroll     - Zoom in/out on timeline (when mouse over timeline)
    Drag       - Pan timeline (click and drag on timeline)
    z          - Reset timeline zoom
    r          - Reset detector state
    s          - Save config to JSON
    q          - Quit
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RectangleSelector, TextBox
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

from detector import EdgeDetector, DetectorConfig, PalletEvent


class DepthMcapReader:
    """Read depth images from an MCAP file."""

    def __init__(self, mcap_path: str, rotate: int = 0):
        """
        Args:
            mcap_path: Path to MCAP file
            rotate: Rotation in degrees (0, 90, 180, 270).
                    Use 90 to put floor at bottom (rotates CW).
        """
        self.mcap_path = mcap_path
        self.rotate = rotate
        self.frames = []
        self.timestamps = []
        self._load_mcap()

    def _load_mcap(self):
        """Load all depth frames from MCAP file."""
        print(f"Loading MCAP file: {self.mcap_path}")

        decoder_factory = DecoderFactory()

        with open(self.mcap_path, "rb") as f:
            reader = make_reader(f, decoder_factories=[decoder_factory])

            for schema, channel, message, decoded_msg in reader.iter_decoded_messages():
                # Look for depth image topics
                topic = channel.topic
                if "depth" in topic.lower() or "image" in topic.lower():
                    try:
                        depth_image = self._decode_depth_message(decoded_msg, channel.topic)
                        if depth_image is not None:
                            timestamp = message.log_time / 1e9  # Convert ns to seconds
                            self.frames.append(depth_image)
                            self.timestamps.append(timestamp)
                    except Exception as e:
                        # Skip frames that can't be decoded
                        pass

        print(f"Loaded {len(self.frames)} depth frames")

        if self.frames:
            print(f"Frame shape: {self.frames[0].shape}")
            print(f"Depth range: {np.nanmin(self.frames[0]):.2f} - {np.nanmax(self.frames[0]):.2f} m")

    def _decode_depth_message(self, msg, topic: str) -> Optional[np.ndarray]:
        """Decode a ROS2 depth image message to numpy array."""
        # Handle sensor_msgs/Image
        if hasattr(msg, 'data') and hasattr(msg, 'encoding'):
            width = msg.width
            height = msg.height
            encoding = msg.encoding

            data = np.frombuffer(bytes(msg.data), dtype=self._get_dtype(encoding))

            if '16UC1' in encoding or 'mono16' in encoding:
                # 16-bit depth in mm, convert to meters
                depth = data.reshape((height, width)).astype(np.float32) / 1000.0
            elif '32FC1' in encoding:
                # 32-bit float depth in meters
                depth = data.reshape((height, width))
            elif '8UC1' in encoding or 'mono8' in encoding:
                # 8-bit - could be normalized depth
                depth = data.reshape((height, width)).astype(np.float32) / 255.0 * 5.0
            else:
                return None

            # Replace invalid values
            depth[depth <= 0] = np.nan
            depth[depth > 10] = np.nan

            # Apply rotation if specified
            # Floor is on left edge, rotate to put it at bottom
            if self.rotate == 90:
                depth = np.rot90(depth, k=1)  # Counter-clockwise 90 (left edge -> bottom)
            elif self.rotate == 180:
                depth = np.rot90(depth, k=2)
            elif self.rotate == 270:
                depth = np.rot90(depth, k=-1)  # Clockwise 90

            return depth

        return None

    def _get_dtype(self, encoding: str) -> np.dtype:
        """Get numpy dtype from ROS encoding string."""
        if '16UC1' in encoding or 'mono16' in encoding:
            return np.uint16
        elif '32FC1' in encoding:
            return np.float32
        elif '8UC1' in encoding or 'mono8' in encoding:
            return np.uint8
        else:
            return np.uint8

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        return self.frames[idx], self.timestamps[idx]


class InteractiveTuner:
    """Interactive GUI for tuning detection parameters."""

    def __init__(self, mcap_reader: DepthMcapReader, config_path: Optional[str] = None):
        self.reader = mcap_reader
        self.config = DetectorConfig()

        # Load existing config if provided
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                self.config = DetectorConfig.from_dict(json.load(f))

        self.detector = EdgeDetector(self.config)

        # Playback state
        self.current_frame = 0
        self.playing = False
        self.play_speed = 1.0

        # History for timeline
        self.depth_history = []
        self.event_markers = []  # [(frame_idx, event_type), ...] - detected events

        # Manual labels for ground truth
        self.labels = []  # [(frame_idx, event_type), ...]
        self._load_labels()

        # ROI selection
        self.roi_rect = None

        # Setup figure
        self._setup_figure()

        # Pre-process all frames for visualization
        self._process_all_frames()

    def _setup_figure(self):
        """Setup matplotlib figure with all widgets."""
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.canvas.manager.set_window_title('MHE-V Pallet Event Detection Tuner')

        # Main axes layout
        # Top left: Depth image with ROI
        self.ax_depth = self.fig.add_axes([0.05, 0.45, 0.45, 0.5])
        self.ax_depth.set_title('Depth Image (draw ROI)')

        # Top right: Timeline (with zoom/pan support)
        self.ax_timeline = self.fig.add_axes([0.55, 0.45, 0.4, 0.5])
        self.ax_timeline.set_title('Median Depth in ROI (scroll to zoom, drag to pan)')
        self.ax_timeline.set_xlabel('Frame')
        self.ax_timeline.set_ylabel('Depth (m)')

        # Enable interactive zoom/pan on timeline
        self.timeline_xlim = None  # Track manual zoom state
        self.pan_start = None  # Track panning state
        self.auto_pan = True  # Auto-pan timeline to follow playhead

        # Debouncing for label moves (prevent keyboard repeat spam)
        self.last_label_move_time = 0
        self.label_move_cooldown = 0.15  # 150ms minimum between moves

        # Slider area - two columns to fit all parameters
        slider_height = 0.018
        slider_spacing = 0.028

        # Left column - main detection parameters
        left_col_x = 0.08
        left_col_width = 0.18
        left_sliders = [
            ('min_edge_strength_m', 'Edge Strength (m)', 0.1, 1.0, None),
            ('window_size', 'Window (frames)', 5, 30, None),
            ('smoothing_window', 'Smoothing', 1, 15, None),
            ('min_event_gap_frames', 'Event Gap (f)', 10, 100, None),
            ('min_valid_pixels_pct', 'Valid Px (%)', 5, 50, None),
            ('max_event_distance_m', 'Max Dist (m)', 0.5, 5.0, None),
        ]

        # Right column - filter parameters
        right_col_x = 0.30
        right_col_width = 0.18
        right_sliders = [
            ('enable_spike_filter', 'Spike Filter', 0, 1, 1),
            ('spike_reject_window_frames', 'Spike Win (f)', 10, 100, None),
            ('enable_dwell_filter', 'Dwell Filter', 0, 1, 1),
            ('min_dwell_frames', 'Dwell (f)', 5, 30, None),
            ('dwell_tolerance_m', 'Dwell Tol (m)', 0.1, 2.0, None),
            ('enable_variance_filter', 'Var Filter', 0, 1, 1),
            ('max_event_variance_m2', 'Max Var (m²)', 0.1, 5.0, None),
            ('enable_baseline_filter', 'Base Filter', 0, 1, 1),
            ('baseline_check_frames', 'Base Chk (f)', 20, 100, None),
            ('baseline_return_threshold_m', 'Base Thr (m)', 0.1, 1.0, None),
            ('enable_pre_stability_filter', 'Pre-Stab Filter', 0, 1, 1),
            ('pre_event_check_frames', 'Pre-Stab (f)', 10, 100, None),
            ('max_pre_event_variance_m2', 'Pre-Var (m²)', 0.01, 0.5, None),
            ('min_nan_block_for_stability', 'NaN Block (f)', 5, 50, None),
            ('max_state_transitions', 'Max Transit', 1, 10, None),
            ('enable_twoway_baseline_filter', 'TwoWay Base', 0, 1, 1),
        ]

        self.sliders = {}

        # Create left column sliders
        for i, (param, label, vmin, vmax, valstep) in enumerate(left_sliders):
            ax = self.fig.add_axes([left_col_x, 0.40 - i * slider_spacing, left_col_width, slider_height])
            initial_val = getattr(self.config, param, vmin)
            if isinstance(initial_val, bool):
                initial_val = 1 if initial_val else 0
            slider = Slider(ax, label, vmin, vmax, valinit=initial_val, valstep=valstep)
            slider.on_changed(self._make_slider_callback(param))
            self.sliders[param] = slider

        # Create right column sliders
        for i, (param, label, vmin, vmax, valstep) in enumerate(right_sliders):
            ax = self.fig.add_axes([right_col_x, 0.40 - i * slider_spacing, right_col_width, slider_height])
            initial_val = getattr(self.config, param, vmin)
            if isinstance(initial_val, bool):
                initial_val = 1 if initial_val else 0
            slider = Slider(ax, label, vmin, vmax, valinit=initial_val, valstep=valstep)
            slider.on_changed(self._make_slider_callback(param))
            self.sliders[param] = slider

        # Frame slider and textbox
        ax_frame = self.fig.add_axes([0.55, 0.35, 0.25, slider_height])
        self.frame_slider = Slider(ax_frame, 'Frame', 0, max(1, len(self.reader) - 1),
                                   valinit=0, valstep=1)
        self.frame_slider.on_changed(self._on_frame_change)

        # Frame number text input
        ax_frame_text = self.fig.add_axes([0.81, 0.35, 0.08, slider_height])
        total_frames = len(self.reader)
        self.frame_textbox = TextBox(ax_frame_text, '', initial=f'0 / {total_frames-1}')
        self.frame_textbox.on_submit(self._on_frame_text_submit)

        # Buttons
        ax_play = self.fig.add_axes([0.55, 0.28, 0.08, 0.04])
        self.btn_play = Button(ax_play, 'Play (1x)')
        self.btn_play.on_clicked(self._toggle_play)

        # Speed buttons
        ax_speed_1x = self.fig.add_axes([0.64, 0.28, 0.04, 0.04])
        self.btn_speed_1x = Button(ax_speed_1x, '1x', color='lightgreen')
        self.btn_speed_1x.on_clicked(lambda e: self._set_speed(1))

        ax_speed_2x = self.fig.add_axes([0.685, 0.28, 0.04, 0.04])
        self.btn_speed_2x = Button(ax_speed_2x, '2x')
        self.btn_speed_2x.on_clicked(lambda e: self._set_speed(2))

        ax_speed_5x = self.fig.add_axes([0.73, 0.28, 0.04, 0.04])
        self.btn_speed_5x = Button(ax_speed_5x, '5x')
        self.btn_speed_5x.on_clicked(lambda e: self._set_speed(5))

        ax_speed_10x = self.fig.add_axes([0.775, 0.28, 0.05, 0.04])
        self.btn_speed_10x = Button(ax_speed_10x, '10x')
        self.btn_speed_10x.on_clicked(lambda e: self._set_speed(10))

        ax_reset = self.fig.add_axes([0.83, 0.28, 0.06, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._reset_detector)

        # Move save button down
        ax_save = self.fig.add_axes([0.83, 0.22, 0.06, 0.04])
        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(self._save_config)

        # Event navigation buttons
        ax_prev_event = self.fig.add_axes([0.55, 0.16, 0.13, 0.04])
        self.btn_prev_event = Button(ax_prev_event, '<< Prev Event [,]', color='lightblue')
        self.btn_prev_event.on_clicked(self._goto_prev_event)

        ax_next_event = self.fig.add_axes([0.69, 0.16, 0.13, 0.04])
        self.btn_next_event = Button(ax_next_event, 'Next Event [.] >>', color='lightblue')
        self.btn_next_event.on_clicked(self._goto_next_event)

        # Labeling buttons
        ax_pickup = self.fig.add_axes([0.55, 0.22, 0.1, 0.04])
        self.btn_pickup = Button(ax_pickup, 'Pick-up [p]', color='lightgreen')
        self.btn_pickup.on_clicked(self._mark_pickup)

        ax_dropoff = self.fig.add_axes([0.66, 0.22, 0.1, 0.04])
        self.btn_dropoff = Button(ax_dropoff, 'Drop-off [d]', color='lightsalmon')
        self.btn_dropoff.on_clicked(self._mark_dropoff)

        ax_clear = self.fig.add_axes([0.77, 0.22, 0.1, 0.04])
        self.btn_clear = Button(ax_clear, 'Clear [c]', color='lightgray')
        self.btn_clear.on_clicked(self._clear_label)

        # Status text (bottom right, below buttons)
        self.ax_status = self.fig.add_axes([0.55, 0.02, 0.4, 0.12])
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(0, 0.95, '', fontsize=8,
                                                verticalalignment='top',
                                                family='monospace')

        # Initialize depth image display
        if len(self.reader) > 0:
            depth, _ = self.reader[0]
            self.depth_im = self.ax_depth.imshow(depth, cmap='viridis',
                                                  vmin=0, vmax=5, aspect='auto')
            self.ax_depth.set_aspect('auto')  # Prevent squishing
            self.fig.colorbar(self.depth_im, ax=self.ax_depth, label='Depth (m)')

            # ROI rectangle
            self._draw_roi()

        # Setup ROI selector
        self.roi_selector = RectangleSelector(
            self.ax_depth, self._on_roi_select,
            useblit=True,
            button=[1],
            minspanx=5, minspany=5,
            spancoords='pixels',
            interactive=True
        )

        # Connect keyboard and mouse events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        self.fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)

    def _make_slider_callback(self, param: str):
        """Create a callback function for a slider."""
        def callback(val):
            # Integer parameters
            if param in ['smoothing_window', 'window_size', 'min_event_gap_frames',
                        'spike_reject_window_frames', 'min_dwell_frames', 'baseline_check_frames',
                        'pre_event_check_frames', 'min_nan_block_for_stability', 'max_state_transitions']:
                val = int(val)
            # Boolean parameters (0=False, 1=True)
            elif param in ['enable_spike_filter', 'enable_dwell_filter',
                          'enable_variance_filter', 'enable_baseline_filter',
                          'enable_pre_stability_filter', 'enable_twoway_baseline_filter']:
                val = bool(int(val))
            setattr(self.config, param, val)
            self._process_all_frames()
            self._update_display()
        return callback

    def _on_frame_change(self, val):
        """Handle frame slider change."""
        self.current_frame = int(val)
        # Update textbox to show current frame
        total_frames = len(self.reader)
        self.frame_textbox.set_val(f'{self.current_frame} / {total_frames-1}')
        self._update_display()

    def _on_frame_text_submit(self, text):
        """Handle frame number text input."""
        try:
            # Extract frame number from "current / total" format
            if '/' in text:
                frame_str = text.split('/')[0].strip()
            else:
                frame_str = text.strip()

            frame_num = int(frame_str)
            # Clamp to valid range
            frame_num = max(0, min(len(self.reader) - 1, frame_num))
            self.current_frame = frame_num
            self.auto_pan = True  # Re-enable auto-pan
            self.frame_slider.set_val(frame_num)
        except ValueError:
            # Invalid input, reset to current frame
            total_frames = len(self.reader)
            self.frame_textbox.set_val(f'{self.current_frame} / {total_frames-1}')

    def _on_roi_select(self, eclick, erelease):
        """Handle ROI selection."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        # Save old ROI in case new one is invalid
        old_roi = (self.config.roi_x, self.config.roi_y,
                   self.config.roi_width, self.config.roi_height)

        self.config.roi_x = min(x1, x2)
        self.config.roi_y = min(y1, y2)
        self.config.roi_width = abs(x2 - x1)
        self.config.roi_height = abs(y2 - y1)

        # Test if new ROI has valid data
        test_frame = self.reader[self.current_frame][0]
        y1, y2 = self.config.roi_y, self.config.roi_y + self.config.roi_height
        x1, x2 = self.config.roi_x, self.config.roi_x + self.config.roi_width
        roi = test_frame[y1:y2, x1:x2]

        valid_mask = (roi > 0.1) & (roi < 10.0) & np.isfinite(roi)
        valid_pct = (np.sum(valid_mask) / roi.size * 100) if roi.size > 0 else 0

        if valid_pct < 5.0:
            print(f"⚠️  Warning: New ROI has only {valid_pct:.1f}% valid pixels!")
            print(f"   ROI might be in invalid area. Select a region with visible depth data.")
            print(f"   Reverting to previous ROI.")
            # Revert to old ROI
            self.config.roi_x, self.config.roi_y = old_roi[0], old_roi[1]
            self.config.roi_width, self.config.roi_height = old_roi[2], old_roi[3]
            self._update_display()
            return

        print(f"✓ ROI updated: ({self.config.roi_x}, {self.config.roi_y}, "
              f"{self.config.roi_width}, {self.config.roi_height}) - {valid_pct:.1f}% valid pixels")

        self.detector = EdgeDetector(self.config)
        self._process_all_frames()
        self._update_display()

    def _draw_roi(self):
        """Draw ROI rectangle on depth image."""
        if self.roi_rect is not None:
            self.roi_rect.remove()

        self.roi_rect = Rectangle(
            (self.config.roi_x, self.config.roi_y),
            self.config.roi_width, self.config.roi_height,
            fill=False, edgecolor='red', linewidth=2
        )
        self.ax_depth.add_patch(self.roi_rect)

    def _toggle_play(self, event=None):
        """Toggle playback."""
        self.playing = not self.playing
        label = 'Pause' if self.playing else f'Play ({self.play_speed}x)'
        self.btn_play.label.set_text(label)

    def _set_speed(self, speed):
        """Set playback speed."""
        self.play_speed = speed

        # Update button colors to show active speed
        self.btn_speed_1x.color = 'lightgreen' if speed == 1 else '0.85'
        self.btn_speed_2x.color = 'lightgreen' if speed == 2 else '0.85'
        self.btn_speed_5x.color = 'lightgreen' if speed == 5 else '0.85'
        self.btn_speed_10x.color = 'lightgreen' if speed == 10 else '0.85'

        # Update play button label
        if self.playing:
            self.btn_play.label.set_text('Pause')
        else:
            self.btn_play.label.set_text(f'Play ({speed}x)')

        self.fig.canvas.draw_idle()
        print(f"Playback speed set to {speed}x")

    def _reset_detector(self, event=None):
        """Reset detector state and reprocess."""
        self.detector = EdgeDetector(self.config)
        self._process_all_frames()
        self._update_display()
        print("✓ Detector reset and frames reprocessed")

    def _save_config(self, event=None):
        """Save current config to JSON file, including rotation."""
        mcap_path = Path(self.reader.mcap_path)
        config_path = mcap_path.parent / (mcap_path.stem + '_config.json')
        config_dict = self.config.to_dict()
        config_dict['rotate'] = self.reader.rotate  # Save rotation with config
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Config saved to {config_path}")

    def _get_labels_path(self) -> str:
        """Get path for labels JSON file."""
        mcap_path = Path(self.reader.mcap_path)
        return str(mcap_path.parent / (mcap_path.stem + '_labels.json'))

    def _load_labels(self):
        """Load labels from JSON file if exists."""
        labels_path = self._get_labels_path()
        if Path(labels_path).exists():
            with open(labels_path) as f:
                self.labels = [tuple(x) for x in json.load(f)]
            print(f"Loaded {len(self.labels)} labels from {labels_path}")

    def _save_labels(self):
        """Save labels to JSON file."""
        labels_path = self._get_labels_path()
        with open(labels_path, 'w') as f:
            json.dump(self.labels, f, indent=2)
        print(f"Saved {len(self.labels)} labels to {labels_path}")

    def _find_nearest_event_frame(self, lookback_frames=100):
        """Find the frame with the largest depth change in recent history.

        This helps when labeling at high speed - finds the actual event
        frame near where you pressed the key.
        """
        if not self.depth_history:
            return self.current_frame

        # Look back from current frame
        start_frame = max(0, self.current_frame - lookback_frames)
        end_frame = min(len(self.depth_history) - 1, self.current_frame)

        if end_frame - start_frame < 10:
            return self.current_frame

        # Find frame with largest absolute depth change
        max_change = 0
        best_frame = self.current_frame

        for i in range(start_frame + 5, end_frame - 5):
            # Compute depth change at this frame
            before = np.nanmean(self.depth_history[i-5:i])
            after = np.nanmean(self.depth_history[i:i+5])

            if not np.isnan(before) and not np.isnan(after):
                change = abs(after - before)
                if change > max_change:
                    max_change = change
                    best_frame = i

        return best_frame

    def _mark_pickup(self, event=None, auto_find=True):
        """Mark current frame as a pick-up event."""
        # Auto-pause when labeling
        if self.playing:
            self.playing = False
            self.btn_play.label.set_text(f'Play ({self.play_speed}x)')
            print("Auto-paused for labeling")

        # Find nearest event if at high speed
        target_frame = self.current_frame
        if auto_find and self.play_speed >= 5:
            target_frame = self._find_nearest_event_frame()
            if target_frame != self.current_frame:
                print(f"Auto-found event at frame {target_frame} (was at {self.current_frame})")
                self.current_frame = target_frame
                # Slider will be updated by _add_label -> _update_display

        self._add_label("PICK_UP")

    def _mark_dropoff(self, event=None, auto_find=True):
        """Mark current frame as a drop-off event."""
        # Auto-pause when labeling
        if self.playing:
            self.playing = False
            self.btn_play.label.set_text(f'Play ({self.play_speed}x)')
            print("Auto-paused for labeling")

        # Find nearest event if at high speed
        target_frame = self.current_frame
        if auto_find and self.play_speed >= 5:
            target_frame = self._find_nearest_event_frame()
            if target_frame != self.current_frame:
                print(f"Auto-found event at frame {target_frame} (was at {self.current_frame})")
                self.current_frame = target_frame
                # Slider will be updated by _add_label -> _update_display

        self._add_label("DROP_OFF")

    def _add_label(self, event_type: str):
        """Add a label at current frame."""
        # Remove any existing label at this frame
        self.labels = [(f, t) for f, t in self.labels if f != self.current_frame]
        # Add new label
        self.labels.append((self.current_frame, event_type))
        self.labels.sort(key=lambda x: x[0])
        self._save_labels()
        self._update_display()
        print(f"Marked frame {self.current_frame} as {event_type}")

    def _move_label(self, delta_frames):
        """Move a label at current frame by delta_frames.

        Args:
            delta_frames: Number of frames to move (+/- integer)
        """
        # Debounce to prevent keyboard repeat spam
        import time
        current_time = time.time()
        if current_time - self.last_label_move_time < self.label_move_cooldown:
            return  # Ignore rapid repeated keypresses
        self.last_label_move_time = current_time

        # Find label at current frame
        label_at_current = None
        for i, (f, t) in enumerate(self.labels):
            if f == self.current_frame:
                label_at_current = (i, f, t)
                break

        if label_at_current is None:
            print(f"⚠️  No label at frame {self.current_frame} to move")
            # Show nearby labels
            nearby = [(f, t) for f, t in self.labels if abs(f - self.current_frame) < 50]
            if nearby:
                print(f"   Nearby labels: {nearby[:3]}")
            return

        idx, old_frame, event_type = label_at_current
        new_frame = max(0, min(len(self.reader) - 1, old_frame + delta_frames))

        if new_frame == old_frame:
            print(f"⚠️  Can't move label - at boundary")
            return

        # Remove old label and add at new position
        self.labels = [(f, t) for f, t in self.labels if f != old_frame]
        # Check if there's already a label at new position
        existing = next((t for f, t in self.labels if f == new_frame), None)
        if existing:
            print(f"⚠️  Label already exists at frame {new_frame} ({existing})")
            # Keep the old label
            self.labels.append((old_frame, event_type))
            self.labels.sort(key=lambda x: x[0])
            return

        # Add label at new position
        self.labels.append((new_frame, event_type))
        self.labels.sort(key=lambda x: x[0])

        # Move playhead to follow the label
        self.current_frame = new_frame
        self.frame_slider.set_val(self.current_frame)

        self._save_labels()
        self._update_display()

        direction = "forward" if delta_frames > 0 else "backward"
        print(f"✓ Moved {event_type} label {direction} {abs(delta_frames)} frames: {old_frame} → {new_frame}")

    def _clear_label(self, event=None):
        """Clear label at current frame."""
        old_len = len(self.labels)
        self.labels = [(f, t) for f, t in self.labels if f != self.current_frame]
        if len(self.labels) < old_len:
            self._save_labels()
            self._update_display()
            print(f"✓ Cleared label at frame {self.current_frame}")
        else:
            print(f"⚠️  No label at frame {self.current_frame} to clear")
            # Show nearby labels
            nearby = [(f, t) for f, t in self.labels if abs(f - self.current_frame) < 100]
            if nearby:
                print(f"   Nearby labels: {nearby[:5]}")

    def _goto_prev_event(self, event=None):
        """Jump to previous labeled event."""
        if not self.labels:
            print("No labels to navigate to")
            return

        # Find the previous label before current frame
        prev_events = [f for f, _ in self.labels if f < self.current_frame]
        if prev_events:
            self.current_frame = max(prev_events)
            self.auto_pan = True  # Re-enable auto-pan for navigation
            self.frame_slider.set_val(self.current_frame)
            print(f"Jumped to event at frame {self.current_frame}")
        else:
            # Wrap around to last event
            self.current_frame = self.labels[-1][0]
            self.auto_pan = True  # Re-enable auto-pan for navigation
            self.frame_slider.set_val(self.current_frame)
            print(f"Wrapped to last event at frame {self.current_frame}")

    def _goto_next_event(self, event=None):
        """Jump to next labeled event."""
        if not self.labels:
            print("No labels to navigate to")
            return

        # Find the next label after current frame
        next_events = [f for f, _ in self.labels if f > self.current_frame]
        if next_events:
            self.current_frame = min(next_events)
            self.auto_pan = True  # Re-enable auto-pan for navigation
            self.frame_slider.set_val(self.current_frame)
            print(f"Jumped to event at frame {self.current_frame}")
        else:
            # Wrap around to first event
            self.current_frame = self.labels[0][0]
            self.auto_pan = True  # Re-enable auto-pan for navigation
            self.frame_slider.set_val(self.current_frame)
            print(f"Wrapped to first event at frame {self.current_frame}")

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == ' ':
            self._toggle_play()
            self.auto_pan = True  # Re-enable auto-pan for playback
        elif event.key == 'left':
            self.current_frame = max(0, self.current_frame - 1)
            self.auto_pan = True  # Re-enable auto-pan for navigation
            self.frame_slider.set_val(self.current_frame)
        elif event.key == 'right':
            self.current_frame = min(len(self.reader) - 1, self.current_frame + 1)
            self.auto_pan = True  # Re-enable auto-pan for navigation
            self.frame_slider.set_val(self.current_frame)
        elif event.key == 'r':
            self._reset_detector()
        elif event.key == 's':
            self._save_config()
        elif event.key == 'p':
            self._mark_pickup(auto_find=True)
        elif event.key == 'P':  # Shift+P = manual placement (no auto-find)
            self._mark_pickup(auto_find=False)
            print("Manual placement (auto-find disabled)")
        elif event.key == 'd':
            self._mark_dropoff(auto_find=True)
        elif event.key == 'D':  # Shift+D = manual placement (no auto-find)
            self._mark_dropoff(auto_find=False)
            print("Manual placement (auto-find disabled)")
        elif event.key == 'up':
            # Move label at current frame backward by 1 frame
            self._move_label(-1)
        elif event.key == 'down':
            # Move label at current frame forward by 1 frame
            self._move_label(1)
        elif event.key == 'pageup' or event.key == 'alt+up':
            # Move label at current frame backward by 10 frames
            self._move_label(-10)
        elif event.key == 'pagedown' or event.key == 'alt+down':
            # Move label at current frame forward by 10 frames
            self._move_label(10)
        elif event.key == '-':
            # Alternative: move label backward 10 frames (Mac-friendly)
            self._move_label(-10)
        elif event.key == '=' or event.key == '+':
            # Alternative: move label forward 10 frames (Mac-friendly)
            self._move_label(10)
        elif event.key == 'c':
            self._clear_label()
        elif event.key == 'C':  # Shift+C = Clear ALL labels
            if self.labels:
                count = len(self.labels)
                self.labels = []
                self._save_labels()
                self._update_display()
                print(f"✓ Cleared ALL {count} labels")
            else:
                print("⚠️  No labels to clear")
        elif event.key == ',':
            self._goto_prev_event()
        elif event.key == '.':
            self._goto_next_event()
        elif event.key == 'i':
            # Show info about current state
            print("\n" + "="*60)
            print(f"Current frame: {self.current_frame}")
            print(f"Total labels: {len(self.labels)}")
            # Show label at current frame
            current_label = next((t for f, t in self.labels if f == self.current_frame), None)
            if current_label:
                print(f"Label at this frame: {current_label}")
            else:
                print("No label at this frame")
                # Show nearest labels
                before = [(f, t) for f, t in self.labels if f < self.current_frame]
                after = [(f, t) for f, t in self.labels if f > self.current_frame]
                if before:
                    last = before[-1]
                    print(f"Last label: frame {last[0]} ({last[1]}) - {self.current_frame - last[0]} frames ago")
                if after:
                    next_label = after[0]
                    print(f"Next label: frame {next_label[0]} ({next_label[1]}) - in {next_label[0] - self.current_frame} frames")
            print("="*60 + "\n")
        elif event.key == 'R':  # Shift+R = Refresh display
            print("Refreshing display...")
            self.ax_depth.set_aspect('auto')
            self._update_display()
            self.fig.canvas.draw()
            print("✓ Display refreshed")
        elif event.key == 'q':
            plt.close(self.fig)
        elif event.key == 'z':
            # Reset zoom
            self.timeline_xlim = None
            self._update_display()
        elif event.key == '1':
            self._set_speed(1)
        elif event.key == '2':
            self._set_speed(2)
        elif event.key == '5':
            self._set_speed(5)
        elif event.key == '0':
            self._set_speed(10)
        elif event.key == '[':
            # Jump back 50 frames (quick rewind)
            self.current_frame = max(0, self.current_frame - 50)
            self.auto_pan = True
            self.frame_slider.set_val(self.current_frame)
            print(f"Rewound 50 frames to {self.current_frame}")
        elif event.key == ']':
            # Jump forward 50 frames
            self.current_frame = min(len(self.reader) - 1, self.current_frame + 50)
            self.auto_pan = True
            self.frame_slider.set_val(self.current_frame)
            print(f"Advanced 50 frames to {self.current_frame}")
        elif event.key == '{':
            # Jump back 200 frames (big rewind)
            self.current_frame = max(0, self.current_frame - 200)
            self.auto_pan = True
            self.frame_slider.set_val(self.current_frame)
            print(f"Rewound 200 frames to {self.current_frame}")
        elif event.key == '}':
            # Jump forward 200 frames
            self.current_frame = min(len(self.reader) - 1, self.current_frame + 200)
            self.auto_pan = True
            self.frame_slider.set_val(self.current_frame)
            print(f"Advanced 200 frames to {self.current_frame}")

    def _on_scroll(self, event):
        """Handle scroll wheel for zoom on timeline."""
        if event.inaxes != self.ax_timeline:
            return

        # Get current xlim
        if self.timeline_xlim is None:
            self.timeline_xlim = list(self.ax_timeline.get_xlim())

        xmin, xmax = self.timeline_xlim
        xrange = xmax - xmin

        # Zoom factor
        zoom_factor = 1.5 if event.button == 'down' else 1 / 1.5

        # Zoom around mouse position
        if event.xdata is not None:
            mouse_x = event.xdata
            left_fraction = (mouse_x - xmin) / xrange
            right_fraction = (xmax - mouse_x) / xrange

            new_range = xrange * zoom_factor
            new_xmin = mouse_x - new_range * left_fraction
            new_xmax = mouse_x + new_range * right_fraction
        else:
            # Zoom around center
            center = (xmin + xmax) / 2
            new_range = xrange * zoom_factor
            new_xmin = center - new_range / 2
            new_xmax = center + new_range / 2

        # Clamp to valid range
        total_frames = len(self.reader)
        new_xmin = max(0, new_xmin)
        new_xmax = min(total_frames - 1, new_xmax)

        self.timeline_xlim = [new_xmin, new_xmax]
        self._update_display()

    def _on_mouse_press(self, event):
        """Handle mouse button press for panning."""
        if event.inaxes == self.ax_timeline and event.button == 1:  # Left click
            self.pan_start = event.xdata

    def _on_mouse_release(self, event):
        """Handle mouse button release."""
        if event.button == 1:  # Left click
            self.pan_start = None

    def _on_mouse_move(self, event):
        """Handle mouse motion for panning."""
        if self.pan_start is None or event.inaxes != self.ax_timeline:
            return

        if event.xdata is None:
            return

        # Initialize xlim if needed
        if self.timeline_xlim is None:
            self.timeline_xlim = list(self.ax_timeline.get_xlim())

        # Calculate pan offset
        dx = self.pan_start - event.xdata
        xmin, xmax = self.timeline_xlim

        # Apply pan
        new_xmin = xmin + dx
        new_xmax = xmax + dx

        # Clamp to valid range
        total_frames = len(self.reader)
        if new_xmin < 0:
            new_xmax -= new_xmin
            new_xmin = 0
        if new_xmax > total_frames - 1:
            new_xmin -= (new_xmax - (total_frames - 1))
            new_xmax = total_frames - 1

        self.timeline_xlim = [new_xmin, new_xmax]
        self.pan_start = event.xdata - dx  # Update pan start for continuous dragging
        self.auto_pan = False  # Disable auto-pan when user manually pans
        self._update_display()

    def _process_all_frames(self):
        """Process all frames with current detector config."""
        self.detector = EdgeDetector(self.config)

        # Get all depth frames
        depth_frames = [self.reader[i][0] for i in range(len(self.reader))]

        # Batch process
        self.depth_history, events = self.detector.process_batch(depth_frames)

        # Convert events to (frame_idx, event_type) tuples
        self.event_markers = [(e.frame_idx, e.event_type) for e in events]

        print(f"Detected {len(self.event_markers)} events")

    def _update_display(self):
        """Update all display elements."""
        if len(self.reader) == 0:
            return

        # Update depth image
        depth, timestamp = self.reader[self.current_frame]
        self.depth_im.set_data(depth)

        # Update ROI rectangle
        self._draw_roi()

        # Update timeline
        self.ax_timeline.clear()
        self.ax_timeline.set_title('Median Depth in ROI (scroll=zoom, z=reset)')
        self.ax_timeline.set_xlabel('Frame')
        self.ax_timeline.set_ylabel('Depth (m)')

        if self.depth_history:
            frames = range(len(self.depth_history))
            self.ax_timeline.plot(frames, self.depth_history, 'b-', label='Depth')

            # Mark predicted events (dotted vertical lines)
            pickup_plotted = False
            dropoff_plotted = False
            for frame_idx, event_type in self.event_markers:
                if event_type == 'PICK_UP':
                    self.ax_timeline.axvline(x=frame_idx, color='green', linestyle='--',
                                              alpha=0.8, linewidth=1.5)
                    pickup_plotted = True
                else:
                    self.ax_timeline.axvline(x=frame_idx, color='orange', linestyle='--',
                                              alpha=0.8, linewidth=1.5)
                    dropoff_plotted = True

            # Mark manual labels (bold triangles on the depth line)
            for frame_idx, event_type in self.labels:
                color = 'green' if event_type == 'PICK_UP' else 'red'
                marker = '^' if event_type == 'PICK_UP' else 'v'
                depth_at_frame = self.depth_history[frame_idx] if frame_idx < len(self.depth_history) else 0
                self.ax_timeline.plot(frame_idx, depth_at_frame, marker, color=color,
                                      markersize=14, markeredgecolor='black', markeredgewidth=1.5, zorder=5)

            # Current frame marker
            self.ax_timeline.axvline(x=self.current_frame, color='black',
                                      linestyle='-', linewidth=2, alpha=0.7)

            # Legend with all elements
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='b', linewidth=2, label='Depth'),
                Line2D([0], [0], color='green', linestyle='--', linewidth=1.5, label=f'Pred: Pick-up ({sum(1 for _, t in self.event_markers if t == "PICK_UP")})'),
                Line2D([0], [0], color='orange', linestyle='--', linewidth=1.5, label=f'Pred: Drop-off ({sum(1 for _, t in self.event_markers if t == "DROP_OFF")})'),
                Line2D([0], [0], marker='^', color='green', linestyle='', markersize=10, label=f'Label: Pick-up ({sum(1 for _, t in self.labels if t == "PICK_UP")})'),
                Line2D([0], [0], marker='v', color='red', linestyle='', markersize=10, label=f'Label: Drop-off ({sum(1 for _, t in self.labels if t == "DROP_OFF")})'),
            ]
            self.ax_timeline.legend(handles=legend_elements, loc='upper right', fontsize=7)

            # Initialize zoom to show 2000 frames (default view)
            if self.timeline_xlim is None:
                # Auto-zoom to show 2000 frames total
                window = 2000
                xmin = max(0, self.current_frame - window // 2)
                xmax = min(len(self.reader) - 1, xmin + window)
                # Adjust if we hit the end
                if xmax == len(self.reader) - 1 and xmax - xmin < window:
                    xmin = max(0, xmax - window)
                self.timeline_xlim = [xmin, xmax]

            # Auto-pan timeline to follow playhead if enabled
            if self.timeline_xlim is not None and self.auto_pan:
                xmin, xmax = self.timeline_xlim
                # Check if current frame is outside visible range
                margin = (xmax - xmin) * 0.1  # 10% margin
                if self.current_frame < xmin + margin or self.current_frame > xmax - margin:
                    # Pan to center on current frame
                    window = xmax - xmin
                    new_xmin = max(0, self.current_frame - window // 2)
                    new_xmax = min(len(self.reader) - 1, new_xmin + window)
                    # Adjust if we hit the end
                    if new_xmax == len(self.reader) - 1:
                        new_xmin = max(0, new_xmax - window)
                    self.timeline_xlim = [new_xmin, new_xmax]

            # Apply the xlim
            if self.timeline_xlim is not None:
                self.ax_timeline.set_xlim(self.timeline_xlim)

        # Update status text
        if self.current_frame < len(self.depth_history):
            depth_val = self.depth_history[self.current_frame]

            # Check if current frame has a label or prediction
            current_label = next((t for f, t in self.labels if f == self.current_frame), None)
            current_pred = next((t for f, t in self.event_markers if f == self.current_frame), None)

            frame_info = f"Frame: {self.current_frame} / {len(self.reader) - 1}"
            if current_label:
                frame_info += f" [LABEL: {current_label}]"
            if current_pred:
                frame_info += f" [PRED: {current_pred}]"

            # Compute accuracy metrics
            tolerance = 50  # frames
            true_positives = 0
            matched_labels = set()
            for det_frame, det_type in self.event_markers:
                for label_frame, label_type in self.labels:
                    if label_type == det_type and abs(det_frame - label_frame) <= tolerance:
                        if label_frame not in matched_labels:
                            matched_labels.add(label_frame)
                            true_positives += 1
                            break

            n_predictions = len(self.event_markers)
            n_labels = len(self.labels)
            false_positives = n_predictions - true_positives
            missed = n_labels - true_positives

            precision = (true_positives / n_predictions * 100) if n_predictions > 0 else 0
            recall = (true_positives / n_labels * 100) if n_labels > 0 else 0

            status = f"""{frame_info}
Depth: {depth_val:.2f}m

=== Predictions vs Labels ===
Predictions: {n_predictions} | Labels: {n_labels}
True Pos: {true_positives} | False Pos: {false_positives} | Missed: {missed}
Precision: {precision:.0f}% | Recall: {recall:.0f}%

Controls: p=pick-up, d=drop-off, c=clear, s=save
"""
            self.status_text.set_text(status)

        self.fig.canvas.draw_idle()

    def _animate(self, frame):
        """Animation callback for playback."""
        if self.playing and len(self.reader) > 0:
            # Advance by play_speed frames
            self.current_frame = (self.current_frame + self.play_speed) % len(self.reader)
            self.frame_slider.set_val(self.current_frame)
        return []

    def run(self):
        """Run the interactive tuner."""
        if len(self.reader) == 0:
            print("No depth frames found in MCAP file!")
            print("Make sure the file contains depth image messages.")
            return

        self._update_display()

        # Setup animation for playback
        self.anim = animation.FuncAnimation(
            self.fig, self._animate,
            interval=100,  # 10 Hz update
            blit=True
        )

        plt.show()


def main():
    parser = argparse.ArgumentParser(description='MHE-V interactive pallet event detection tuner')
    parser.add_argument('mcap_file', help='Path to MCAP file with depth images')
    parser.add_argument('--config', '-c', help='Path to existing config JSON file')
    parser.add_argument('--rotate', '-r', type=int, default=0, choices=[0, 90, 180, 270],
                        help='Rotate image (degrees). Use 90 to put floor at bottom.')

    args = parser.parse_args()

    if not Path(args.mcap_file).exists():
        print(f"Error: MCAP file not found: {args.mcap_file}")
        sys.exit(1)

    # Determine rotation: command line arg takes priority, then config file
    rotate = args.rotate
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config_data = json.load(f)
            if 'rotate' in config_data and args.rotate == 0:
                rotate = config_data['rotate']
                print(f"Using rotation from config: {rotate}°")

    reader = DepthMcapReader(args.mcap_file, rotate=rotate)
    tuner = InteractiveTuner(reader, args.config)
    tuner.run()


if __name__ == '__main__':
    main()
