from pathlib import Path

import numpy as np
import pytest

from lunar_trn import frames
from lunar_trn.datasets import stereolunar as sl

REPO = Path(__file__).resolve().parents[1]
NADIR_DIR = REPO / "data" / "stereolunar" / "nadir"


def _write_frame(directory: Path, index: int, depth_value=12345.0):
    import cv2
    import OpenEXR

    f = 256 / np.tan(np.deg2rad(22.5))
    K = np.array([[f, 0, 256], [0, f, 256], [0, 0, 1]], dtype=np.float32)
    p = frames.stereo_to_body(1000.0, 2000.0, 10000.0)
    down = -p / np.linalg.norm(p)
    x = np.cross(down, [0, 0, 1.0]); x /= np.linalg.norm(x)
    y = np.cross(down, x)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.column_stack([x, y, down])
    T[:3, 3] = p
    np.savez(directory / f"im_{index:05d}.npz", intrinsics=K, cam2world=T)
    img = np.tile(np.linspace(0, 255, 512, dtype=np.uint8), (512, 1))
    cv2.imwrite(str(directory / f"im_{index:05d}.jpg"), np.dstack([img] * 3))
    depth = np.full((512, 512), depth_value, dtype=np.float16)
    depth[0, 0] = np.inf
    hdr = {"compression": OpenEXR.PIZ_COMPRESSION, "type": OpenEXR.scanlineimage}
    with OpenEXR.File(hdr, {"Y": depth}) as ex:
        ex.write(str(directory / f"im_{index:05d}.exr"))


def test_loader_on_synthetic_files(tmp_path):
    for i in (12, 13, 14):
        _write_frame(tmp_path, i)
    fr = sl.list_frames(tmp_path, trajectory="nadir")
    assert [f.index for f in fr] == [12, 13, 14]
    f = fr[2]
    assert (f.index_block, f.index_illumination) == (2, 1)
    assert f.illumination is None
    img = f.image()
    assert img.shape == (512, 512) and img.dtype == np.float32 and 0.0 <= img.min() <= img.max() <= 1.0
    dep = f.depth()
    assert dep.dtype == np.float32 and np.isinf(dep[0, 0]) and dep[1, 1] == pytest.approx(12345.0, rel=1e-3)
    assert np.allclose(f.T_world_cam[:3, :3].T @ f.T_world_cam[:3, :3], np.eye(3), atol=1e-12)
    assert f.altitude_above_sphere() == pytest.approx(10000.0, abs=0.5)
    assert f.off_nadir_deg() == pytest.approx(0.0, abs=1e-3)
    cam = f.camera()
    assert (cam.width, cam.height) == (512, 512)


def test_generator_sun_direction_matches_formula():
    p = frames.stereo_to_body(-4e4, 2.5e4, 0.0)
    for (az, inc) in sl.SUN_SETUPS_DEG:
        s = sl.generator_sun_direction(az, inc, p)
        az_eff, el = sl.sun_azimuth_elevation(s, p)
        # 'inc' acts as elevation through sin(inc); a negative cos flips the azimuth by 180 deg.
        assert el == pytest.approx(np.rad2deg(np.arcsin(np.sin(np.deg2rad(inc)))), abs=1e-9)
        expect_az = az % 360 if np.cos(np.deg2rad(inc)) > 0 else (az + 180) % 360
        assert (az_eff - expect_az + 180) % 360 - 180 == pytest.approx(0.0, abs=1e-9)


@pytest.mark.data
@pytest.mark.skipif(not NADIR_DIR.exists(), reason="StereoLunar nadir subset not downloaded")
def test_real_nadir_frames_load():
    fr = [f for f in sl.list_frames(NADIR_DIR) if f.depth_path.exists() and f.image_path.exists()][:3]
    if not fr:
        pytest.skip("no complete frames yet")
    for f in fr:
        assert f.image().shape == (512, 512)
        d = f.depth()
        assert d.shape == (512, 512)
        alt = f.altitude_above_sphere()
        assert 2_000 < alt < 40_000
        assert np.nanmedian(d[np.isfinite(d)]) == pytest.approx(alt, rel=0.4)
        assert f.off_nadir_deg() < 5.0


def test_assign_triplets_handles_batch_offsets(tmp_path):
    """Pose groups straddling a 6-block boundary must still get illumination 0, 1, 2."""
    # Batch starting at index 4: images 4, 6, 8 share geometry A; 10, 12, 14 share B.
    for i in (4, 6, 8, 10, 12):
        _write_frame(tmp_path, i)
    fr = sl.list_frames(tmp_path, trajectory="nadir")
    # Give the second geometry a different pose.
    for f in fr:
        if f.index >= 10:
            T = f.T_world_cam.copy()
            T[0, 3] += 500.0
            f._T = T
    groups = sl.assign_triplets(fr)
    assert [[f.index for f in g] for g in groups] == [[4, 6, 8], [10, 12]]
    assert [f.illumination for f in groups[0]] == [0, 1, 2]
    assert [f.illumination for f in groups[1]] == [None, None]
    assert groups[0][2].geometry_id == 4
    with pytest.raises(ValueError):
        groups[1][0].sun_direction()
