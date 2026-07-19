import os
import shlex
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa as lr
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
                                  quaternion_multiply)
from tqdm import tqdm

smpl_joints = [
    "root",  # 0
    "lhip",  # 1
    "rhip",  # 2
    "belly", # 3
    "lknee", # 4
    "rknee", # 5
    "spine", # 6
    "lankle",# 7
    "rankle",# 8
    "chest", # 9
    "ltoes", # 10
    "rtoes", # 11
    "neck",  # 12
    "linshoulder", # 13
    "rinshoulder", # 14
    "head", # 15
    "lshoulder", # 16
    "rshoulder",  # 17
    "lelbow", # 18
    "relbow",  # 19
    "lwrist", # 20
    "rwrist", # 21
    "lhand", # 22
    "rhand", # 23
]

smpl_parents = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
]

smpl_offsets = [
    [0.0, 0.0, 0.0],
    [0.05858135, -0.08228004, -0.01766408],
    [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522],
    [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304],
    [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428],
    [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329],
    [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817],
    [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987],
    [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207],
    [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873],
    [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676],
    [0.08669055, -0.01063603, -0.01559429],
    [-0.0887537, -0.00865157, -0.01010708],
]


def set_line_data_3d(line, x):
    line.set_data(x[:, :2].T)
    line.set_3d_properties(x[:, 2])


def set_scatter_data_3d(scat, x, c):
    scat.set_offsets(x[:, :2])
    scat.set_3d_properties(x[:, 2], "z")
    scat.set_facecolors([c])


def get_axrange(poses):
    pose = poses[0]
    x_min = pose[:, 0].min()
    x_max = pose[:, 0].max()

    y_min = pose[:, 1].min()
    y_max = pose[:, 1].max()

    z_min = pose[:, 2].min()
    z_max = pose[:, 2].max()

    xdiff = x_max - x_min
    ydiff = y_max - y_min
    zdiff = z_max - z_min

    biggestdiff = max([xdiff, ydiff, zdiff])
    return biggestdiff


def audio_output_stem(name):
    actual_name = name[0] if isinstance(name, (list, tuple)) else name
    base_name = os.path.splitext(os.path.basename(str(actual_name)))[0]
    if not base_name:
        return "sample"

    parts = base_name.split("_")
    if len(parts) > 1 and parts[-1].startswith("slice"):
        return "_".join(parts[:-1]) or base_name
    return base_name


def smooth_sequence(sequence, window=5):
    if window <= 1 or len(sequence) < 3:
        return sequence

    window = min(window, len(sequence))
    if window % 2 == 0:
        window = max(1, window - 1)
    if window <= 1:
        return sequence

    pad = window // 2
    kernel = np.ones(window, dtype=np.float32) / window

    original_shape = sequence.shape
    flattened = sequence.reshape(sequence.shape[0], -1)
    padded = np.pad(flattened, ((pad, pad), (0, 0)), mode="edge")

    smoothed = np.empty_like(flattened, dtype=np.float32)
    for i in range(flattened.shape[1]):
        smoothed[:, i] = np.convolve(padded[:, i], kernel, mode="valid")

    return smoothed.reshape(original_shape).astype(sequence.dtype, copy=False)


def compute_camera_track(vis_poses, mode="follow"):
    mode = (mode or "follow").lower()
    if mode == "fixed" and os.getenv("EDGE_RENDER_FIXED_BOUNDS", "0") == "1":
        def _bounds(name, default):
            raw = os.getenv(name, default)
            parts = [float(x.strip()) for x in raw.split(",") if x.strip()]
            if len(parts) != 2 or parts[0] >= parts[1]:
                raise ValueError(f"{name} must be 'min,max', got {raw!r}")
            return parts[0], parts[1]

        xlim = _bounds("EDGE_RENDER_XLIM", "-1.8,1.8")
        ylim = _bounds("EDGE_RENDER_YLIM", "-1.8,1.8")
        zlim = _bounds("EDGE_RENDER_ZLIM", "-0.05,2.25")
        center = np.asarray(
            [(xlim[0] + xlim[1]) * 0.5, (ylim[0] + ylim[1]) * 0.5],
            dtype=vis_poses.dtype,
        )
        camera_radius = float(max(xlim[1] - xlim[0], ylim[1] - ylim[0]) * 0.5)
        frame_centers = np.repeat(center[None, :], vis_poses.shape[0], axis=0)
        return frame_centers, camera_radius, zlim

    # 用根节点(骨盆)作为跟拍中心，比包围盒中心更稳定，不会因为甩手/抬腿把人“挤”到坐标盒边缘。
    root_centers = vis_poses[:, 0, :2]
    if mode == "fixed":
        xy = vis_poses[:, :, :2].reshape(-1, 2)
        xy_min = np.percentile(xy, 1, axis=0)
        xy_max = np.percentile(xy, 99, axis=0)
        xy_min = np.minimum(xy_min, root_centers.min(axis=0))
        xy_max = np.maximum(xy_max, root_centers.max(axis=0))

        center = ((xy_min + xy_max) * 0.5).astype(vis_poses.dtype, copy=False)
        half_span = float(np.max((xy_max - xy_min) * 0.5))
        camera_radius = float(np.clip(half_span * 1.15 + 0.25, 1.35, 20.0))
        frame_centers = np.repeat(center[None, :], vis_poses.shape[0], axis=0)
    else:
        frame_centers = smooth_sequence(root_centers, window=31)

        horizontal_offsets = np.abs(vis_poses[:, :, :2] - frame_centers[:, None, :])
        frame_half_span = horizontal_offsets.max(axis=1).max(axis=1)
        body_radius = float(np.percentile(frame_half_span, 98) * 1.2 + 0.15)
        camera_radius = float(np.clip(body_radius, 1.35, 2.6))

    z_min = float(min(-0.2, np.percentile(vis_poses[:, :, 2], 1) - 0.1))
    z_max = float(max(2.2, np.percentile(vis_poses[:, :, 2], 99) + 0.2))
    return frame_centers, camera_radius, (z_min, z_max)


def plot_single_pose(num, poses, lines, ax, camera_centers, camera_radius, z_limits, scat, contact):
    pose = poses[num]
    static = contact[num]
    indices = [7, 8, 10, 11]

    for i, (point, idx) in enumerate(zip(scat, indices)):
        position = pose[idx : idx + 1]
        color = "r" if static[i] else "g"
        set_scatter_data_3d(point, position, color)

    for i, (p, line) in enumerate(zip(smpl_parents, lines)):
        if i == 0:
            continue
        data = np.stack((pose[i], pose[p]), axis=0)
        set_line_data_3d(line, data)

    xcenter, ycenter = camera_centers[num]
    ax.set_xlim(xcenter - camera_radius, xcenter + camera_radius)
    ax.set_ylim(ycenter - camera_radius, ycenter + camera_radius)
    ax.set_zlim(*z_limits)


def skeleton_render(
    poses,
    epoch=0,
    out="renders",
    name="",
    sound=True,
    stitch=False,
    sound_folder="ood_sliced",
    contact=None,
    render=True,
    camera_mode="follow",
    output_path=None,
    render_smooth_window=9,
):
    if render:
        Path(out).mkdir(parents=True, exist_ok=True)
        num_steps = poses.shape[0]

        fig = plt.figure(figsize=(6.5, 6.5), dpi=140)
        ax = fig.add_subplot(projection="3d")
        if hasattr(ax, "set_proj_type"):
            ax.set_proj_type("ortho")

        # 定义地板平面：由于 vis_poses 中 Y 和 Z 已交换，地面位于 z=0
        grid_size = 50.0
        xx, yy = np.meshgrid(np.linspace(-grid_size, grid_size, 2), np.linspace(-grid_size, grid_size, 2))
        z_plane = np.zeros_like(xx)
        ax.plot_surface(xx, yy, z_plane, zorder=-11, cmap=cm.twilight, alpha=0.2)

        ax.view_init(elev=20, azim=45)

        lines = [
            ax.plot([], [], [], zorder=10, linewidth=1.5)[0]
            for _ in smpl_parents
        ]
        scat = [
            ax.scatter([], [], [], zorder=10, s=0, cmap=ListedColormap(["r", "g", "b"]))
            for _ in range(4)
        ]

        feet = poses[:, (7, 8, 10, 11)]
        feetv = np.zeros(feet.shape[:2])
        feetv[:-1] = np.linalg.norm(feet[1:] - feet[:-1], axis=-1)
        if contact is None:
            contact = feetv < 0.01
        else:
            contact = contact > 0.95

        # 对渲染序列做轻量时域平滑，改善观感上的抖动和僵硬感。
        render_poses = smooth_sequence(
            poses.copy(),
            window=max(1, int(render_smooth_window)),
        )

        vis_poses = render_poses.copy()
        vis_poses[:, :, 1] = -render_poses[:, :, 2]  # Depth
        vis_poses[:, :, 2] = render_poses[:, :, 1]   # Height

        camera_centers, camera_radius, z_limits = compute_camera_track(vis_poses, mode=camera_mode)
        ax.set_box_aspect((2.0 * camera_radius, 2.0 * camera_radius, z_limits[1] - z_limits[0]))

        anim = animation.FuncAnimation(
            fig,
            plot_single_pose,
            num_steps,
            fargs=(vis_poses, lines, ax, camera_centers, camera_radius, z_limits, scat, contact),
            interval=1000 // 30,
        )

    if sound:
        if render:
            Path(out).mkdir(parents=True, exist_ok=True)
            temp_dir = TemporaryDirectory(dir=out)
            videoname = os.path.join(temp_dir.name, f"{epoch}.mp4")
            writer = animation.FFMpegWriter(
                fps=30,
                bitrate=4000,
                codec="libx264",
                extra_args=["-pix_fmt", "yuv420p"],
            )
            anim.save(videoname, writer=writer)

        if stitch:
            assert isinstance(name, (list, tuple)), "For stitching, name must be a list or tuple"
            output_stem = audio_output_stem(name)
            name_ = [os.path.splitext(x)[0] + ".wav" for x in name]
            audio, sr = lr.load(name_[0], sr=None)
            ll, half = len(audio), len(audio) // 2
            total_wav = np.zeros(ll + half * (len(name_) - 1))
            total_wav[:ll] = audio
            idx = ll
            for n_ in name_[1:]:
                audio, sr = lr.load(n_, sr=None)
                total_wav[idx : idx + half] = audio[half:]
                idx += half
            audioname = f"{temp_dir.name}/tempsound.wav" if render else os.path.join(out, f"{epoch}_{output_stem}.wav")
            sf.write(audioname, total_wav, sr)
            outname = output_path or os.path.join(
                out,
                f"{epoch}_{output_stem}.mp4",
            )
        else:
            actual_name = name[0] if isinstance(name, (list, tuple)) else name
            assert isinstance(actual_name, str) and actual_name != "", "Must provide an audio filename"
            audioname = actual_name
            outname = output_path or os.path.join(
                out, f"{epoch}_{os.path.splitext(os.path.basename(actual_name))[0]}.mp4"
            )

        if render:
            Path(os.path.dirname(outname) or ".").mkdir(parents=True, exist_ok=True)
            out_cmd = os.system(
                "ffmpeg -loglevel error -y "
                f"-i {shlex.quote(videoname)} "
                f"-i {shlex.quote(audioname)} "
                f"-shortest -c:v copy -c:a aac -q:a 4 {shlex.quote(outname)}"
            )
    else:
        if render:
            actual_name = name[0] if isinstance(name, (list, tuple)) else name
            path = os.path.normpath(str(actual_name))
            pathparts = path.split(os.sep)
            base_name = pathparts[-1].replace(".npy", "").replace(".wav", "").replace(".pkl", "")
            gifname = os.path.join(out, f"{base_name}.gif")
            anim.save(gifname, savefig_kwargs={"transparent": True, "facecolor": "none"})

    plt.close()


class SMPLSkeleton:
    def __init__(
        self, device=None,
    ):
        offsets = smpl_offsets
        parents = smpl_parents
        assert len(offsets) == len(parents)

        self._offsets = torch.Tensor(offsets).to(device)
        self._parents = np.array(parents)
        self._compute_metadata()

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

    def forward(self, rotations, root_positions):
        assert len(rotations.shape) == 4
        assert len(root_positions.shape) == 3
        rotations = axis_angle_to_quaternion(rotations)

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(
            rotations.shape[0],
            rotations.shape[1],
            self._offsets.shape[0],
            self._offsets.shape[1],
        )

        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(
                    quaternion_apply(
                        rotations_world[self._parents[i]], expanded_offsets[:, :, i]
                    )
                    + positions_world[self._parents[i]]
                )
                if self._has_children[i]:
                    rotations_world.append(
                        quaternion_multiply(
                            rotations_world[self._parents[i]], rotations[:, :, i]
                        )
                    )
                else:
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)
