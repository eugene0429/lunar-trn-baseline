"""StereoLunar dataset loader (Grethen et al., ICCV 2025 Workshops; Zenodo record 21133765).

Layout of an extracted archive (``nadir/``, ``oblique/`` or ``dynamic/``)::

    im_XXXXX.jpg   8-bit grey image (stored as 3 identical channels), per-image min-max
                   normalised from the float render, so intensities are not radiometric
    im_XXXXX.npz   ``intrinsics`` (3x3 K) and ``cam2world`` (4x4), both float32
    im_XXXXX.exr   single channel ``Y`` depth map, half float, metres, inf = no hit

Index structure: the ReadMe says images ``6k .. 6k+5`` show one stereo geometry with
``6k+2i`` / ``6k+2i+1`` the left/right images under illumination ``i``. That holds only
within a generation batch: the released folders concatenate batches whose first index is
not always a multiple of 6 (V1 found 177 of 452 sampled blocks straddling two
geometries). Illumination is therefore assigned by grouping images with identical
``cam2world`` (:func:`assign_triplets`): a complete group is three images ``j, j+2, j+4``
and its members get illumination 0, 1, 2 in index order.

The released npz files carry no metadata (no Sun vector, no altitude), so the Sun direction
is reconstructed from the generator code (``dataset-generation/*_trajectory.py``):

    dir = cos(inc) (cos(az) N + sin(az) E) + sin(inc) U

at the target surface point, with ``(az, inc)`` in ``SUN_SETUPS_DEG``. Note that ``inc``
acts as an elevation angle in this formula; e.g. (150, 160) is a Sun at 20 deg elevation and
330 deg azimuth. The interpretation is checked against the images by the V1 render test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import frames
from ..camera import PinholeCamera, orthonormalize

SUN_SETUPS_DEG = ((150.0, 160.0), (250.0, 20.0), (360.0, 165.0))
ILLUMINATION_NAMES = ("side", "overhead", "back")
_INDEX_RE = re.compile(r"im_(\d+)\.npz$")


def generator_sun_direction(az_deg: float, inc_deg: float, p_surf) -> np.ndarray:
    """Unit Sun direction (world frame) as computed by the StereoLunar trajectory generator."""
    enu = frames.enu_basis(p_surf)
    east, north, up = enu[:, 0], enu[:, 1], enu[:, 2]
    az, inc = np.deg2rad(az_deg), np.deg2rad(inc_deg)
    v = np.cos(inc) * (np.cos(az) * north + np.sin(az) * east) + np.sin(inc) * up
    return v / np.linalg.norm(v)


def sun_azimuth_elevation(sun_dir, p) -> tuple[float, float]:
    """Azimuth (deg, from local north towards east) and elevation (deg) of a Sun direction at p."""
    enu = frames.enu_basis(p)
    e, n, u = (np.asarray(sun_dir) @ enu)
    return float(np.rad2deg(np.arctan2(e, n)) % 360.0), float(np.rad2deg(np.arcsin(np.clip(u, -1, 1))))


def load_image(path: str | Path) -> np.ndarray:
    """Grey image as float32 in [0, 1]."""
    import cv2

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img.astype(np.float32) / 255.0


def load_depth(path: str | Path) -> np.ndarray:
    """Depth map (float32 metres, inf where the ray hit nothing)."""
    import OpenEXR

    with OpenEXR.File(str(path)) as ex:
        channels = ex.channels()
        key = "Y" if "Y" in channels else next(iter(channels))
        return np.asarray(channels[key].pixels, dtype=np.float32)


def load_camera(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(K, T_world_cam) as float64; the rotation is re-orthonormalised from float32."""
    with np.load(path) as d:
        K = np.asarray(d["intrinsics"], dtype=np.float64)
        T = np.asarray(d["cam2world"], dtype=np.float64)
    T = T.copy()
    T[:3, :3] = orthonormalize(T[:3, :3])
    return K, T


@dataclass
class Frame:
    """One StereoLunar image with its camera and depth map."""

    directory: Path
    index: int
    trajectory: str
    illumination: int | None = None  # set by assign_triplets()
    geometry_id: int | None = None  # index of the first image of the pose group
    _K: np.ndarray | None = field(default=None, repr=False)
    _T: np.ndarray | None = field(default=None, repr=False)

    @property
    def stem(self) -> str:
        return f"im_{self.index:05d}"

    @property
    def image_path(self) -> Path:
        return self.directory / f"{self.stem}.jpg"

    @property
    def depth_path(self) -> Path:
        return self.directory / f"{self.stem}.exr"

    @property
    def camera_path(self) -> Path:
        return self.directory / f"{self.stem}.npz"

    @property
    def index_block(self) -> int:
        """Naive geometry id ``index // 6`` from the ReadMe (wrong across batch boundaries)."""
        return self.index // 6

    @property
    def index_illumination(self) -> int:
        """Naive illumination ``(index // 2) % 3`` from the ReadMe (wrong across batch boundaries)."""
        return (self.index // 2) % 3

    def _load_cam(self) -> None:
        if self._K is None:
            self._K, self._T = load_camera(self.camera_path)

    @property
    def K(self) -> np.ndarray:
        self._load_cam()
        return self._K

    @property
    def T_world_cam(self) -> np.ndarray:
        self._load_cam()
        return self._T

    def image(self) -> np.ndarray:
        return load_image(self.image_path)

    def depth(self) -> np.ndarray:
        return load_depth(self.depth_path)

    def camera(self, pixel_center: float = 0.5) -> PinholeCamera:
        img_w = int(round(2 * self.K[0, 2]))
        img_h = int(round(2 * self.K[1, 2]))
        return PinholeCamera(self.K, img_w, img_h, pixel_center=pixel_center)

    @property
    def position(self) -> np.ndarray:
        return self.T_world_cam[:3, 3]

    def altitude_above_sphere(self) -> float:
        return float(np.linalg.norm(self.position) - frames.MOON_RADIUS_M)

    def off_nadir_deg(self) -> float:
        """Angle between the optical axis and the local down direction."""
        z = self.T_world_cam[:3, 2]
        down = -self.position / np.linalg.norm(self.position)
        return float(np.rad2deg(np.arccos(np.clip(z @ down, -1, 1))))

    def boresight_point(self) -> np.ndarray:
        """World point where the optical axis meets the terrain (z-depth at the image centre)."""
        d = self.depth()
        h, w = d.shape
        c = d[h // 2 - 2:h // 2 + 2, w // 2 - 2:w // 2 + 2]
        z = float(np.nanmedian(np.where(np.isfinite(c), c, np.nan)))
        return self.position + z * self.T_world_cam[:3, 2]

    def sun_direction(self, reference_point=None) -> np.ndarray:
        """Reconstructed Sun direction (world frame).

        The generator evaluates the Sun's (azimuth, incidence) in the local ENU frame of its
        look-at target. Near the pole that frame rotates quickly with longitude, so the
        reference must be the target, not the point below the camera: for oblique views tens
        of km away the two frames differ by tens of degrees (V1 oblique). The default is the
        boresight ground point from the depth map, which is the target to within the stereo
        baseline offset; for nadir views it is the point below the camera.
        """
        if self.illumination is None:
            raise ValueError(f"{self.stem}: illumination unknown; call assign_triplets() on complete groups")
        az, inc = SUN_SETUPS_DEG[self.illumination]
        p = self.boresight_point() if reference_point is None else reference_point
        return generator_sun_direction(az, inc, p)


def list_frames(directory: str | Path, trajectory: str | None = None) -> list[Frame]:
    """All frames with an npz file in ``directory``, sorted by index."""
    directory = Path(directory)
    traj = trajectory or directory.name
    out = []
    for p in sorted(directory.glob("im_*.npz")):
        m = _INDEX_RE.search(p.name)
        if m:
            out.append(Frame(directory=directory, index=int(m.group(1)), trajectory=traj))
    return out


def assign_triplets(frames_: list[Frame], decimals: int = 2) -> list[list[Frame]]:
    """Group frames by identical pose and label complete illumination triplets in place.

    Returns the pose groups (sorted by first index). Members of a complete group (three
    frames with indices j, j+2, j+4) get ``illumination`` 0, 1, 2 and ``geometry_id`` j;
    members of incomplete groups keep ``illumination = None``.
    """
    groups: dict[bytes, list[Frame]] = {}
    for f in frames_:
        key = np.round(f.T_world_cam, decimals).tobytes()
        groups.setdefault(key, []).append(f)
    out = sorted((sorted(g, key=lambda f: f.index) for g in groups.values()), key=lambda g: g[0].index)
    for g in out:
        idx = [f.index for f in g]
        complete = len(g) == 3 and idx[1] - idx[0] == 2 and idx[2] - idx[1] == 2
        for k, f in enumerate(g):
            f.geometry_id = idx[0]
            f.illumination = k if complete else None
    return out
