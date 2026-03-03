# ArtekMed SlangPy Visualizer

Interactive viewer for multi-camera E57 pointclouds from artekmed datasets,
rendered as colored sprite billboards using SlangPy's Vulkan backend.

## Prerequisites

- Linux with a Vulkan-capable GPU (NVIDIA recommended)
- Python 3.10+
- The following local packages installed:
  - `slangpy` (GPU rendering framework)
  - `slangpy-renderer` (3D rendering library)
  - `artekmed-dataset-reader[e57]` (dataset reader with E57 support)

## Installation

```bash
# Activate the renderer venv (or your own)
source /home/narvis/develop/rendering/.venv-renderer/bin/activate

# Install dependencies (if not already installed)
pip install -e /home/narvis/develop/rendering/slangpy
pip install -e /home/narvis/develop/rendering/tcn_slangpy_renderer
pip install -e /home/narvis/develop/artekmed/artekmed_dataset_reader[e57]

# Install this visualizer
pip install -e /home/narvis/develop/rendering/artekmed_slangpy_visualizer
```

## Usage

```bash
# Basic usage — loads frame 0 with all cameras
artekmed-visualizer /path/to/dataset

# Specify frame and point size
artekmed-visualizer /path/to/dataset --frame 5 --point-size 0.003

# Filter to specific cameras
artekmed-visualizer /path/to/dataset --cameras cam0 cam1

# Custom window size and clip planes
artekmed-visualizer /path/to/dataset --width 1920 --height 1080 --near 0.01 --far 50.0

# Verbose logging
artekmed-visualizer /path/to/dataset -v
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `dataset_path` | (required) | Path to artekmed dataset directory |
| `--frame` | 0 | Frame index to load |
| `--point-size` | 0.004 | Sprite billboard size in world units |
| `--width` | 1280 | Window width |
| `--height` | 720 | Window height |
| `--cameras` | all | Space-separated camera IDs to load |
| `--near` | 0.01 | Near clip plane |
| `--far` | 100.0 | Far clip plane |
| `-v` | off | Verbose logging |

## Controls

| Input | Action |
|---|---|
| Left-click + drag | Rotate camera |
| Shift + left-click + drag | Pan camera |
| Scroll wheel | Zoom |
| Shift + scroll | Fine zoom |
| ESC | Quit |

## Architecture

### Data Flow

```
artekmed dataset (E57 files)
    |
    v
artekmed_dataset_reader.open_dataset()  -->  FrameGroup
    |
    v
PointCloud (N x 3 float64 positions, N x 3 uint8 colors, RigidTransform pose)
    |
    v
color_packing.pack_colors_to_texture()  -->  (H x W x 4 uint8 RGBA, N x 2 float32 UVs)
    |
    v
SimplePointcloud (GPU buffers: position_buffer, uv_buffer, texture)
    |
    v
PointcloudSpritesRenderer.render()  -->  Vulkan render pass  -->  window
```

### Color Packing

The `PointcloudSpritesRenderer` samples colors from a `Texture2D` using UV coordinates
(it has no per-vertex color attribute). Since E57 pointclouds provide per-point RGB values,
we pack N colors into a roughly-square 2D RGBA texture:

- Texture dimensions: ceil(sqrt(N)) x ceil(N / width)
- Each point index maps to a unique texel
- UVs point to texel centers (half-pixel offset) to avoid filtering artifacts

For a 500K-point cloud, this produces a ~708x708 texture (~2MB) — well within GPU limits.

### Pose Transforms

Each E57 file may contain a `RigidTransform` (quaternion + translation) representing
the camera's pose in world space. This is converted to a 4x4 row-major matrix and
passed as the `model_matrix` to the renderer. Points are transformed in the shader:

```
clip_pos = mul(proj, mul(view, mul(model, vertex)))
```

### SimplePointcloud vs Pointcloud

The library's `Pointcloud` class requires CuPy for GPU buffer uploads (CUDA interop).
`SimplePointcloud` is a lightweight alternative that creates GPU buffers directly from
numpy arrays via `device.create_buffer(data=...)`. It exposes the same interface
(`position_buffer`, `uv_buffer`, `texture`, `has_vertices`, etc.) so it works
directly with `PointcloudSpritesRenderer.render()`.

## Troubleshooting

**"No E57 for camera X"**: The camera has no E57 pointcloud for this frame. Check
that the dataset contains E57 files and try other frame indices.

**Validation errors on startup**: Enable debug layers are on by default. If you see
Vulkan validation errors, ensure your GPU drivers are up to date.

**Points are too small/large**: Adjust `--point-size`. Start with 0.004 and go
up/down. The value is in world units.

**Scene is off-center**: The camera targets the mean centroid of all loaded
pointclouds. If only one camera is loaded and it's far from the origin, this is
expected. Try loading all cameras.

**ImportError for pye57**: Install with E57 support:
`pip install artekmed-dataset-reader[e57]`
