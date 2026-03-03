#!/usr/bin/env python3
"""Interactive depth-unprojected pointcloud viewer for artekmed datasets.

Loads per-camera raw depth and color images from an artekmed dataset, unprojects
each depth image to a 3D pointcloud on the GPU using DepthUnprojector (with
per-pixel normals and color camera UV projection), and renders the results as
normal-oriented textured hexagonal surfels.

Controls:
    Left-click + drag          Rotate
    Shift + left-click + drag  Pan
    Scroll wheel               Zoom (shift for fine zoom)
    ESC                        Quit

Usage::

    artekmed-visualizer /path/to/dataset
    artekmed-visualizer /path/to/dataset --frame 5 --sprite-scale 2.0
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from pathlib import Path

import numpy as np
import slangpy as spy

from artekmed_dataset_reader import FrameGroup, open_dataset
from slangpy_renderer import (
    CameraIntrinsics,
    ColorProjectionParameters,
    DepthParameters,
    DepthUnprojector,
)
from slangpy_renderer.controllers import ArcBall
from slangpy_renderer.renderers import PointcloudSurfelRenderer

from .pose_utils import rigid_transform_to_matrix

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Projection helper (same as view_cube.py / offscreen.py)
# ---------------------------------------------------------------------------


def vulkan_rh_zo_perspective(
    fov_y_deg: float, aspect: float, near: float, far: float
) -> np.ndarray:
    """Right-handed Vulkan perspective projection (depth 0..1, row-major)."""
    fovy = math.radians(fov_y_deg)
    f = 1.0 / math.tan(0.5 * fovy)
    A = far / (near - far)
    B = (far * near) / (near - far)

    P = np.zeros((4, 4), dtype=np.float32)
    P[0, 0] = f / aspect
    P[1, 1] = f
    P[2, 2] = A
    P[2, 3] = B
    P[3, 2] = -1.0
    return P


# ---------------------------------------------------------------------------
# Depth camera view (GPU-unprojected pointcloud + color texture)
# ---------------------------------------------------------------------------


class DepthCameraView:
    """Holds GPU-unprojected pointcloud + color texture for one depth camera.

    Duck-types the interface expected by ``PointcloudSurfelRenderer.render()``:
    position_buffer, normal_buffer, uv_buffer, texture, vertices, and the
    has_* properties.
    """

    def __init__(
        self,
        device: spy.Device,
        unprojector: DepthUnprojector,
        color_image: np.ndarray,
        model_matrix: np.ndarray,
        color_params: ColorProjectionParameters,
        depth_params: DepthParameters,
    ) -> None:
        # GPU buffers from the unprojector (stable references)
        self.position_buffer = unprojector.position_buffer
        self.normal_buffer = unprojector.normal_buffer
        self.uv_buffer = unprojector.texcoord_buffer
        self.model_matrix = model_matrix

        # Per-camera rendering parameters
        self.color_params = color_params
        self.depth_fy = depth_params.intrinsics.fy
        self.depth_width = depth_params.width
        self.depth_height = depth_params.height

        # vertices attribute provides vertex count (.size) and depth dims (.shape)
        self.vertices = np.empty(
            (depth_params.height, depth_params.width), dtype=np.uint8
        )

        # Upload color image as RGBA texture
        ch, cw = color_image.shape[:2]
        rgba = np.zeros((ch, cw, 4), dtype=np.uint8)
        rgba[:, :, :3] = color_image
        rgba[:, :, 3] = 255

        self.texture = device.create_texture(
            format=spy.Format.rgba8_unorm_srgb,
            width=cw,
            height=ch,
            usage=spy.TextureUsage.shader_resource,
        )
        self.texture.copy_from_numpy(rgba)

        # Keep unprojector alive (owns the GPU buffers)
        self._unprojector = unprojector

    @property
    def has_vertices(self) -> bool:
        return True

    @property
    def has_normals(self) -> bool:
        return True

    @property
    def has_texcoords(self) -> bool:
        return True

    @property
    def has_texture(self) -> bool:
        return True

    def extra_args(self, sprite_scale: float) -> dict:
        """Build per-camera extra_args dict for surfel rendering."""
        return {
            "color_params": self.color_params,
            "depth_fy": self.depth_fy,
            "sprite_scale": sprite_scale,
            "depthWidth": self.depth_width,
            "depthHeight": self.depth_height,
            "useStaticColor": False,
        }


# ---------------------------------------------------------------------------
# Calibration conversion helpers
# ---------------------------------------------------------------------------


def _camera_params_to_dict(params) -> dict:
    """Convert artekmed CameraParameters to dict for from_calibration()."""
    return {
        "fx": params.fx,
        "fy": params.fy,
        "cx": params.cx,
        "cy": params.cy,
        "width": params.width,
        "height": params.height,
        "radial_distortion": params.radial_distortion,
        "tangential_distortion": params.tangential_distortion,
        "metric_radius": params.metric_radius,
    }


def _rigid_transform_to_dict(transform) -> dict:
    """Convert artekmed RigidTransform to dict for from_calibration()."""
    return {
        "rotation": {
            "x": transform.rotation.x,
            "y": transform.rotation.y,
            "z": transform.rotation.z,
            "w": transform.rotation.w,
        },
        "translation": {
            "x": transform.translation.x,
            "y": transform.translation.y,
            "z": transform.translation.z,
        },
    }


def _build_depth_params(params) -> DepthParameters:
    """Convert artekmed CameraParameters to DepthParameters."""
    return DepthParameters(
        width=params.width,
        height=params.height,
        intrinsics=CameraIntrinsics(
            fx=params.fx,
            fy=params.fy,
            cx=params.cx,
            cy=params.cy,
            radial_distortion=params.radial_distortion,
            tangential_distortion=params.tangential_distortion,
            max_radius=params.metric_radius,
        ),
    )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

# The DepthUnprojector produces points in the Azure Kinect depth camera frame:
#   x-right, y-down, z-forward (positive)
# The camera_pose from calibration transforms points in the E57 convention:
#   x-right, y-up, z-backward (negative)
# Verified by test_depth_unprojection.py: e57 = (cam_x, -cam_y, -cam_z).
# This matrix converts from camera frame to E57 frame before applying camera_pose.
_CAM_TO_E57 = np.diag([1.0, -1.0, -1.0, 1.0]).astype(np.float32)


def load_cameras(
    device: spy.Device,
    frame: FrameGroup,
    camera_filter: list[str] | None,
) -> list[DepthCameraView]:
    """Load depth + color images per camera and run GPU unprojection."""
    cameras = frame.camera_ids
    if camera_filter:
        cameras = [c for c in cameras if c in camera_filter]

    views: list[DepthCameraView] = []

    for cam_id in cameras:
        if not frame.has_depth(cam_id) or not frame.has_color(cam_id):
            log.warning("No depth/color for camera %s, skipping", cam_id)
            continue

        calib = frame.calibration(cam_id)
        if calib is None:
            log.warning("No calibration for camera %s, skipping", cam_id)
            continue

        log.info("Loading depth+color for camera %s ...", cam_id)
        depth_image = frame.depth_image(cam_id)
        color_image = frame.color_image(cam_id)

        depth_params = _build_depth_params(calib.depth_parameters)
        color_params = ColorProjectionParameters.from_calibration(
            _camera_params_to_dict(calib.color_parameters),
            _rigid_transform_to_dict(calib.color2depth_transform),
        )

        log.info(
            "  Depth: %s %s, Color: %s %s",
            depth_image.shape,
            depth_image.dtype,
            color_image.shape,
            color_image.dtype,
        )

        # GPU unprojection: positions, normals, UVs
        unprojector = DepthUnprojector(device, depth_params, color_params)
        unprojector.unproject(depth_image)

        dw, dh = depth_params.width, depth_params.height
        log.info("  Unprojected: %dx%d = %d points", dw, dh, dw * dh)

        # Model matrix: camera_pose expects E57 convention (y-up, z-backward),
        # so convert from depth camera convention (y-down, z-forward) first.
        model = rigid_transform_to_matrix(calib.camera_pose) @ _CAM_TO_E57

        view = DepthCameraView(
            device, unprojector, color_image, model, color_params, depth_params
        )
        views.append(view)

    return views


def compute_scene_center(cameras: list[DepthCameraView]) -> np.ndarray:
    """Compute the scene center from the center pixel of each camera's pointcloud.

    Reads back the center pixel's 3D position from each camera's unprojector,
    transforms it to world space, and returns the mean.  Falls back to the mean
    camera position if no valid center pixels exist.
    """
    if not cameras:
        return np.zeros(3, dtype=np.float32)

    centers = []
    for cam in cameras:
        cy, cx = cam.depth_height // 2, cam.depth_width // 2
        local_pos = cam._unprojector.to_numpy()[cy, cx]
        if local_pos[2] > 0:
            world = cam.model_matrix @ np.append(local_pos, 1.0)
            centers.append(world[:3])

    if not centers:
        # Fallback: mean camera position
        centers = [cam.model_matrix[:3, 3] for cam in cameras]

    return np.mean(centers, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Interactive depth-unprojected pointcloud viewer for artekmed datasets."
    )
    parser.add_argument("dataset_path", help="Path to artekmed dataset directory")
    parser.add_argument("--frame", type=int, default=0, help="Frame index (default: 0)")
    parser.add_argument(
        "--sprite-scale",
        type=float,
        default=1.5,
        help="Surfel size multiplier (default: 1.5)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Window width")
    parser.add_argument("--height", type=int, default=720, help="Window height")
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=None,
        help="Camera IDs to load (default: all)",
    )
    parser.add_argument(
        "--near", type=float, default=0.01, help="Near clip plane (default: 0.01)"
    )
    parser.add_argument(
        "--far", type=float, default=100.0, help="Far clip plane (default: 100.0)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    # --- Open dataset ---
    ds = open_dataset(args.dataset_path)
    log.info("Opened dataset: %d frames", len(ds))

    if args.frame >= len(ds):
        log.error("Frame %d out of range (dataset has %d frames)", args.frame, len(ds))
        sys.exit(1)

    frame = ds[args.frame]
    log.info("Frame %d: cameras %s", args.frame, frame.camera_ids)

    # --- Create window and device ---
    width, height = args.width, args.height
    window = spy.Window(width, height, "ArtekMed Depth Viewer", resizable=True)

    # Resolve shader include path from slangpy_renderer package
    import slangpy_renderer

    asset_root = Path(slangpy_renderer.__file__).parent / "assets"
    device = spy.Device(
        type=spy.DeviceType.vulkan,
        enable_debug_layers=True,
        compiler_options={
            "include_paths": [
                str(asset_root / "shaders"),
                os.path.join(os.path.dirname(spy.__file__), "slang"),
            ],
        },
    )

    surface = device.create_surface(window)
    surface.configure(width, height)
    output_format = surface.config.format

    # --- Load cameras (GPU depth unprojection) ---
    cameras = load_cameras(device, frame, args.cameras)
    if not cameras:
        log.error("No cameras loaded — nothing to render")
        sys.exit(1)

    log.info("Loaded %d cameras", len(cameras))

    # --- Create renderer and depth buffer ---
    renderer = PointcloudSurfelRenderer(device, output_format)

    def create_depth_texture() -> spy.Texture:
        return device.create_texture(
            format=spy.Format.d32_float,
            width=window.width,
            height=window.height,
            usage=spy.TextureUsage.depth_stencil,
        )

    depth_texture = create_depth_texture()

    # --- Set up arcball camera ---
    scene_center = compute_scene_center(cameras)
    camera_distance = 3.0
    camera_pos = scene_center + np.array([0.0, 0.0, camera_distance], dtype=np.float32)
    camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    fov = 60.0

    arcball = ArcBall(camera_pos, scene_center, camera_up, fov, (width, height))

    # --- Mouse/keyboard state ---
    current_button = None
    needs_init = False
    dirty = False
    should_render = True

    def on_mouse_event(event: spy.MouseEvent) -> None:
        nonlocal current_button, needs_init, should_render

        if event.type == spy.MouseEventType.button_down:
            if current_button != event.button:
                needs_init = True
            current_button = event.button

        elif event.type == spy.MouseEventType.button_up:
            current_button = None

        elif event.type == spy.MouseEventType.move:
            pos = (int(event.pos.x), int(event.pos.y))
            if current_button == spy.MouseButton.left:
                if needs_init:
                    needs_init = False
                    arcball.init_transformation(pos)

                if event.mods == spy.KeyModifierFlags.shift:
                    arcball.translate(pos)
                else:
                    arcball.rotate(pos)
                should_render = True

        elif event.type == spy.MouseEventType.scroll:
            delta = event.scroll.y / 5.0
            if event.mods == spy.KeyModifierFlags.shift:
                delta /= 3.0
            arcball.zoom(delta)
            should_render = True

    def on_keyboard_event(event: spy.KeyboardEvent) -> None:
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                window.close()

    def on_resize(w: int, h: int) -> None:
        nonlocal dirty, should_render
        dirty = True
        should_render = True

    window.on_mouse_event = on_mouse_event
    window.on_keyboard_event = on_keyboard_event
    window.on_resize = on_resize

    # --- Render loop ---
    log.info("Entering render loop. ESC to quit.")

    while not window.should_close():
        window.process_events()

        if not should_render:
            continue
        should_render = False

        if dirty:
            del depth_texture
            device.wait()
            surface.configure(window.width, window.height)
            depth_texture = create_depth_texture()
            arcball.reshape((window.width, window.height))
            dirty = False

        arcball.update_transformation()

        surface_texture = surface.acquire_next_image()
        if not surface_texture:
            continue

        view_matrix = arcball.view_matrix()
        aspect = float(window.width) / float(window.height)
        proj_matrix = vulkan_rh_zo_perspective(fov, aspect, args.near, args.far)

        command_encoder = device.create_command_encoder()

        with command_encoder.begin_render_pass(
            {
                "color_attachments": [
                    {
                        "view": surface_texture.create_view(),
                        "clear_value": [0.1, 0.1, 0.1, 1.0],
                        "load_op": spy.LoadOp.clear,
                    }
                ],
                "depth_stencil_attachment": {
                    "view": depth_texture.create_view(),
                    "depth_clear_value": 1.0,
                    "depth_load_op": spy.LoadOp.clear,
                    "depth_store_op": spy.StoreOp.store,
                    "depth_read_only": False,
                },
            }
        ) as pass_encoder:
            for cam in cameras:
                renderer.render(
                    pass_encoder,
                    cam,
                    (window.width, window.height),
                    view_matrix,
                    proj_matrix,
                    cam.model_matrix,
                    cam.extra_args(args.sprite_scale),
                )

        device.submit_command_buffer(command_encoder.finish())
        surface.present()


if __name__ == "__main__":
    main()
