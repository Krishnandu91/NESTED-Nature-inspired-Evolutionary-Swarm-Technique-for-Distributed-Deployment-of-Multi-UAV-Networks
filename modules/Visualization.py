import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle
import matplotlib.cm as cm
import os
from PIL import Image


class Visualizer:
    def __init__(self, config, grid_manager):
        self.config = config
        self.grid_manager = grid_manager
        self.output_dir = "visualizations"
        os.makedirs(self.output_dir, exist_ok=True)

        self.cmap_users = cm.get_cmap('viridis')
        self.cmap_uavs = cm.get_cmap('plasma')

        self.initial_uav_count = config.NO_UAV
        print(f"Visualizer initialized. Output directory: {self.output_dir}")

    # ---------------------------------------------------------
    # Plot UAV and user distribution
    # ---------------------------------------------------------
    def _plot_uav_user_distribution(
        self, ax, time_step, uavs, users, drone_positions, access_links
    ):

        ax.set_xlabel('X-Coordinate (meters)', fontsize=18)
        ax.set_ylabel('Y-Coordinate (meters)', fontsize=18)

        ax.set_xlim(0, self.config.FIELD_SIZE)
        ax.set_ylim(0, self.config.FIELD_SIZE)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.15)

        ax.tick_params(axis='x', labelsize=14, rotation=45, pad=6)
        ax.tick_params(axis='y', labelsize=14, pad=8)

        # Align rotated X labels to the right so they don't overlap
        for label in ax.get_xticklabels():
            label.set_ha('right')

        # ---------------------------------------------------------
        # Users
        # ---------------------------------------------------------
        connected_users = [u for u in users if u.connected_drone is not None]
        disconnected_users = [u for u in users if u.connected_drone is None]

        if connected_users:
            ax.scatter(
                [u.position[0] for u in connected_users],
                [u.position[1] for u in connected_users],
                c='green',
                s=95,
                edgecolors='black',
                linewidth=1.5,
                marker='o',
                label='Covered Users'
            )

        if disconnected_users:
            ax.scatter(
                [u.position[0] for u in disconnected_users],
                [u.position[1] for u in disconnected_users],
                c='red',
                s=200,
                edgecolors='darkred',
                linewidth=2,
                marker='x',
                label='Uncovered Users'
            )

        # ---------------------------------------------------------
        # UAV Classification
        # ---------------------------------------------------------
        initial_uavs = {'x': [], 'y': []}
        new_uavs = {'x': [], 'y': []}
        dead_uavs = {'x': [], 'y': []}
        stochastic_failed_uavs = {'x': [], 'y': []}

        for i, (uav, pos) in enumerate(zip(uavs, drone_positions)):

            if uav.failed:
                if getattr(uav, 'stochastic_failure_happened', False):
                    stochastic_failed_uavs['x'].append(pos[0])
                    stochastic_failed_uavs['y'].append(pos[1])
                else:
                    dead_uavs['x'].append(pos[0])
                    dead_uavs['y'].append(pos[1])

            elif i >= self.initial_uav_count:
                new_uavs['x'].append(pos[0])
                new_uavs['y'].append(pos[1])

            else:
                initial_uavs['x'].append(pos[0])
                initial_uavs['y'].append(pos[1])

        # ---------------------------------------------------------
        # Plot UAV Types
        # ---------------------------------------------------------
        if initial_uavs['x']:
            ax.scatter(
                initial_uavs['x'], initial_uavs['y'],
                c='blue',
                s=300,
                edgecolors='black',
                linewidth=3,
                marker='o',
                label='Initial UAVs'
            )

        if new_uavs['x']:
            ax.scatter(
                new_uavs['x'], new_uavs['y'],
                c='lime',
                s=450,
                edgecolors='darkgreen',
                linewidth=3,
                marker='*',
                label='New UAVs'
            )

        if dead_uavs['x']:
            ax.scatter(
                dead_uavs['x'], dead_uavs['y'],
                c='black',
                s=700,
                edgecolors='orange',
                linewidth=4,
                marker='X',
                label='Dead UAVs (Energy)'
            )

        if stochastic_failed_uavs['x']:
            ax.scatter(
                stochastic_failed_uavs['x'], stochastic_failed_uavs['y'],
                c='red',
                s=800,
                edgecolors='darkred',
                linewidth=5,
                marker='X',
                label='Failed UAVs (Stochastic)'
            )

            for x, y in zip(stochastic_failed_uavs['x'], stochastic_failed_uavs['y']):
                ax.add_patch(
                    Circle(
                        (x, y),
                        1000,
                        fill=False,
                        edgecolor='red',
                        linestyle='--',
                        linewidth=3,
                        alpha=0.7
                    )
                )

        # ---------------------------------------------------------
        # Access Links
        # ---------------------------------------------------------
        for u_idx, d_idx in access_links:
            if d_idx < len(drone_positions) and u_idx < len(users):
                if not uavs[d_idx].failed:
                    up = users[u_idx].position
                    dp = drone_positions[d_idx]
                    ax.plot(
                        [up[0], dp[0]],
                        [up[1], dp[1]],
                        'g-', alpha=0.4, linewidth=1
                    )

        # Coverage circles
        for i, pos in enumerate(drone_positions):
            if not uavs[i].failed:
                ax.add_patch(
                    Circle(
                        (pos[0], pos[1]),
                        self.config.ACCESS_LINK_RANGE,
                        fill=False,
                        linestyle='--',
                        color='blue',
                        linewidth=1.5,
                        alpha=0.2
                    )
                )

        # ---------------------------------------------------------
        # Legend Outside (Stable Version)
        # ---------------------------------------------------------
        ax.legend(
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            fontsize=14,
            borderaxespad=0
        )

        # Make room for legend and rotated x-tick labels
        plt.subplots_adjust(right=0.78, bottom=0.18)

    # ---------------------------------------------------------
    # Save Frame
    # ---------------------------------------------------------
    def plot_and_save_distribution_frame(
        self, time_step, uavs, users, drone_positions, access_links
    ):
        # Reduced figure size and DPI to keep GIF under 25 MB
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        self._plot_uav_user_distribution(
            ax, time_step, uavs, users, drone_positions, access_links
        )

        frame_dir = os.path.join(self.output_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        frame_path = os.path.join(frame_dir, f"frame_{time_step:04d}.png")

        # Lowered DPI from 120 → 72 to reduce per-frame file size
        fig.savefig(frame_path, dpi=72, bbox_inches='tight')
        plt.close(fig)

        print(f"Saved frame: {frame_path}")
        return frame_path

    # ---------------------------------------------------------
    # Create Animation
    # ---------------------------------------------------------
    def create_animation(self, interval=500):

        frame_dir = os.path.join(self.output_dir, "frames")

        if not os.path.exists(frame_dir):
            print("Frames directory not found.")
            return None

        frame_files = sorted([
            f for f in os.listdir(frame_dir)
            if f.startswith("frame_") and f.endswith(".png")
        ])

        if not frame_files:
            print("No frame files found.")
            return None

        print(f"Creating animation from {len(frame_files)} frames...")

        # -----------------------------------------------------------
        # MP4 — unchanged logic, still uses matplotlib ArtistAnimation
        # -----------------------------------------------------------
        fig = plt.figure(figsize=(12, 12))
        plt.axis('off')

        ims = []
        for frame_file in frame_files:
            filepath = os.path.join(frame_dir, frame_file)
            img = plt.imread(filepath)
            im = plt.imshow(img, animated=True)
            ims.append([im])

        ani = animation.ArtistAnimation(
            fig, ims,
            interval=interval,
            blit=True,
            repeat_delay=2000
        )

        mp4_path = os.path.join(self.output_dir, "simulation_animation.mp4")
        try:
            ani.save(mp4_path, writer='ffmpeg', fps=2, dpi=100)
            print(f"MP4 saved: {mp4_path}")
        except Exception as e:
            print(f"MP4 saving failed: {e}")

        plt.close(fig)

        # -----------------------------------------------------------
        # GIF — built manually with Pillow for maximum size control
        # -----------------------------------------------------------
        gif_path = os.path.join(self.output_dir, "simulation_animation.gif")
        self._save_optimized_gif(frame_files, frame_dir, gif_path, fps=2)

        return ani

    # ---------------------------------------------------------
    # Optimized GIF builder using Pillow
    # ---------------------------------------------------------
    def _save_optimized_gif(
        self,
        frame_files,
        frame_dir,
        gif_path,
        fps=2,
        max_width=640,        # resize width — increase if you have headroom
        colors=128,           # GIF palette size (max 256); lower = smaller file
        target_mb=25,
    ):
        """
        Build a GIF frame-by-frame with Pillow.
        Applies resizing + palette quantization to stay under target_mb.
        """
        duration_ms = int(1000 / fps)

        pil_frames = []
        for frame_file in frame_files:
            filepath = os.path.join(frame_dir, frame_file)
            img = Image.open(filepath).convert("RGB")

            # --- Resize to max_width, keep aspect ratio ---
            w, h = img.size
            if w > max_width:
                new_h = int(h * max_width / w)
                img = img.resize((max_width, new_h), Image.LANCZOS)

            # --- Quantize to a limited palette (reduces file size a lot) ---
            img = img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)

            pil_frames.append(img)

        if not pil_frames:
            print("No frames to save.")
            return

        try:
            pil_frames[0].save(
                gif_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=duration_ms,
                loop=0,
                optimize=True,       # enables LZW compression pass
            )

            size_mb = os.path.getsize(gif_path) / (1024 * 1024)
            print(f"GIF saved: {gif_path}  ({size_mb:.1f} MB)")

            # --- If still too large, shrink width and retry once ---
            if size_mb > target_mb:
                print(
                    f"GIF is {size_mb:.1f} MB > {target_mb} MB target. "
                    "Retrying with smaller dimensions and fewer colors…"
                )
                self._save_optimized_gif(
                    frame_files,
                    frame_dir,
                    gif_path,
                    fps=fps,
                    max_width=max(320, max_width - 160),
                    colors=max(64, colors - 32),
                    target_mb=target_mb,
                )

        except Exception as e:
            print(f"GIF saving failed: {e}")