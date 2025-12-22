#!/usr/bin/env python3
"""
MHE-V (Material Handling Equipment Vision) interactive tuning tool.
Replay MCAP files and adjust pallet pick/drop detection parameters in real-time.

Usage:
    python tuner.py bags/your_depth_file.mcap
    python tuner.py bags/your_depth_file.mcap --rotate 90  # Floor at bottom

Controls:
    Space      - Play/Pause
    Left/Right - Step frame backward/forward
    p          - Mark current frame as PICK-UP event
    d          - Mark current frame as DROP-OFF event
    c          - Clear label at current frame
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
from matplotlib.widgets import Slider, Button, RectangleSelector
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

        # Top right: Timeline
        self.ax_timeline = self.fig.add_axes([0.55, 0.45, 0.4, 0.5])
        self.ax_timeline.set_title('Median Depth in ROI')
        self.ax_timeline.set_xlabel('Frame')
        self.ax_timeline.set_ylabel('Depth (m)')

        # Slider area
        slider_left = 0.15
        slider_width = 0.3
        slider_height = 0.02
        slider_spacing = 0.032

        # Create sliders - matching edge detector config
        sliders_config = [
            ('min_edge_strength_m', 'Min Edge Strength (m)', 0.1, 1.0),
            ('window_size', 'Window Size (frames)', 5, 30),
            ('smoothing_window', 'Smoothing Window', 1, 15),
            ('min_event_gap_frames', 'Min Event Gap (frames)', 10, 100),
            ('min_valid_pixels_pct', 'Min Valid Pixels (%)', 5, 50),
        ]

        self.sliders = {}
        for i, (param, label, vmin, vmax) in enumerate(sliders_config):
            ax = self.fig.add_axes([slider_left, 0.35 - i * slider_spacing, slider_width, slider_height])
            initial_val = getattr(self.config, param, vmin)
            slider = Slider(ax, label, vmin, vmax, valinit=initial_val)
            slider.on_changed(self._make_slider_callback(param))
            self.sliders[param] = slider

        # Frame slider
        ax_frame = self.fig.add_axes([0.55, 0.35, 0.35, slider_height])
        self.frame_slider = Slider(ax_frame, 'Frame', 0, max(1, len(self.reader) - 1),
                                   valinit=0, valstep=1)
        self.frame_slider.on_changed(self._on_frame_change)

        # Buttons
        ax_play = self.fig.add_axes([0.55, 0.28, 0.08, 0.04])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self._toggle_play)

        ax_reset = self.fig.add_axes([0.65, 0.28, 0.08, 0.04])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self._reset_detector)

        ax_save = self.fig.add_axes([0.75, 0.28, 0.08, 0.04])
        self.btn_save = Button(ax_save, 'Save')
        self.btn_save.on_clicked(self._save_config)

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

        # Status text
        self.ax_status = self.fig.add_axes([0.55, 0.02, 0.4, 0.18])
        self.ax_status.axis('off')
        self.status_text = self.ax_status.text(0, 0.9, '', fontsize=10,
                                                verticalalignment='top',
                                                family='monospace')

        # Initialize depth image display
        if len(self.reader) > 0:
            depth, _ = self.reader[0]
            self.depth_im = self.ax_depth.imshow(depth, cmap='viridis',
                                                  vmin=0, vmax=5)
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

        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _make_slider_callback(self, param: str):
        """Create a callback function for a slider."""
        def callback(val):
            # Integer parameters
            if param in ['smoothing_window', 'window_size', 'min_event_gap_frames']:
                val = int(val)
            setattr(self.config, param, val)
            self._process_all_frames()
            self._update_display()
        return callback

    def _on_frame_change(self, val):
        """Handle frame slider change."""
        self.current_frame = int(val)
        self._update_display()

    def _on_roi_select(self, eclick, erelease):
        """Handle ROI selection."""
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)

        self.config.roi_x = min(x1, x2)
        self.config.roi_y = min(y1, y2)
        self.config.roi_width = abs(x2 - x1)
        self.config.roi_height = abs(y2 - y1)

        self.detector = PalletEventDetector(self.config)
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
        self.btn_play.label.set_text('Pause' if self.playing else 'Play')

    def _reset_detector(self, event=None):
        """Reset detector state and reprocess."""
        self.detector.reset()
        self._process_all_frames()
        self._update_display()

    def _save_config(self, event=None):
        """Save current config to JSON file, including rotation."""
        config_path = Path(self.reader.mcap_path).stem + '_config.json'
        config_dict = self.config.to_dict()
        config_dict['rotate'] = self.reader.rotate  # Save rotation with config
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        print(f"Config saved to {config_path}")

    def _get_labels_path(self) -> str:
        """Get path for labels JSON file."""
        return Path(self.reader.mcap_path).stem + '_labels.json'

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

    def _mark_pickup(self, event=None):
        """Mark current frame as a pick-up event."""
        self._add_label("PICK_UP")

    def _mark_dropoff(self, event=None):
        """Mark current frame as a drop-off event."""
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

    def _clear_label(self, event=None):
        """Clear label at current frame."""
        old_len = len(self.labels)
        self.labels = [(f, t) for f, t in self.labels if f != self.current_frame]
        if len(self.labels) < old_len:
            self._save_labels()
            self._update_display()
            print(f"Cleared label at frame {self.current_frame}")

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == ' ':
            self._toggle_play()
        elif event.key == 'left':
            self.current_frame = max(0, self.current_frame - 1)
            self.frame_slider.set_val(self.current_frame)
        elif event.key == 'right':
            self.current_frame = min(len(self.reader) - 1, self.current_frame + 1)
            self.frame_slider.set_val(self.current_frame)
        elif event.key == 'r':
            self._reset_detector()
        elif event.key == 's':
            self._save_config()
        elif event.key == 'p':
            self._mark_pickup()
        elif event.key == 'd':
            self._mark_dropoff()
        elif event.key == 'c':
            self._clear_label()
        elif event.key == 'q':
            plt.close(self.fig)

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
        self.ax_timeline.set_title('Median Depth in ROI')
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
            self.current_frame = (self.current_frame + 1) % len(self.reader)
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
