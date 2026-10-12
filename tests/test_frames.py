import numpy as np
import pytest

from lunar_trn import frames

MOON2000_SPOLE = (
    'PROJCS["Moon2000_spole",GEOGCS["GCS_Moon",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400,0]],'
    'PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]],PROJECTION["Polar_Stereographic"],'
    'PARAMETER["latitude_of_origin",-90],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],'
    'PARAMETER["false_northing",0],UNIT["metre",1]]'
)
MOON_GEOG = 'GEOGCS["GCS_Moon",DATUM["D_Moon_2000",SPHEROID["Moon_2000_IAU_IAG",1737400,0]],PRIMEM["Reference_Meridian",0],UNIT["degree",0.0174532925199433]]'


def test_stereo_matches_proj():
    """Our closed form must match PROJ for the CRS written in the PGDA GeoTIFF."""
    from rasterio.crs import CRS
    from rasterio.warp import transform

    rng = np.random.default_rng(0)
    lat = rng.uniform(-90, -86, 200)
    lon = rng.uniform(-180, 180, 200)
    xs, ys = transform(CRS.from_wkt(MOON_GEOG), CRS.from_wkt(MOON2000_SPOLE), lon.tolist(), lat.tolist())
    x, y = frames.latlon_to_stereo(lat, lon)
    np.testing.assert_allclose(x, xs, atol=1e-6)
    np.testing.assert_allclose(y, ys, atol=1e-6)


def test_latlon_stereo_round_trip():
    rng = np.random.default_rng(1)
    lat = rng.uniform(-90, -80, 1000)
    lon = rng.uniform(-180, 180, 1000)
    x, y = frames.latlon_to_stereo(lat, lon)
    lat2, lon2 = frames.stereo_to_latlon(x, y)
    np.testing.assert_allclose(lat2, lat, atol=1e-9)
    dlon = (lon2 - lon + 180) % 360 - 180
    np.testing.assert_allclose(dlon, 0, atol=1e-7)


def test_stereo_body_round_trip_and_geometry():
    rng = np.random.default_rng(2)
    x = rng.uniform(-1e5, 1e5, 500)
    y = rng.uniform(-1e5, 1e5, 500)
    h = rng.uniform(-5e3, 4e4, 500)
    p = frames.stereo_to_body(x, y, h)
    np.testing.assert_allclose(np.linalg.norm(p, axis=-1), frames.MOON_RADIUS_M + h, rtol=1e-12)
    x2, y2, h2 = frames.body_to_stereo(p)
    np.testing.assert_allclose(x2, x, atol=1e-6)
    np.testing.assert_allclose(y2, y, atol=1e-6)
    np.testing.assert_allclose(h2, h, atol=1e-6)
    # Consistency with the lat/lon route.
    lat, lon = frames.stereo_to_latlon(x, y)
    lat_b, lon_b, _ = frames.body_to_latlon(p)
    np.testing.assert_allclose(lat_b, lat, atol=1e-9)
    np.testing.assert_allclose((lon_b - lon + 180) % 360 - 180, 0, atol=1e-7)


def test_axes_orientation():
    # lon = 0 lies along +X_body and along +y_map; lon = 90 E along +Y_body and +x_map.
    p0 = frames.stereo_to_body(0.0, 1000.0, 0.0)
    p90 = frames.stereo_to_body(1000.0, 0.0, 0.0)
    assert p0[0] > 0 and abs(p0[1]) < 1e-6
    assert p90[1] > 0 and abs(p90[0]) < 1e-6
    assert p0[2] < 0


def test_enu_basis_right_handed():
    p = frames.stereo_to_body(3e4, -2e4, 0.0)
    enu = frames.enu_basis(p)
    np.testing.assert_allclose(enu.T @ enu, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(enu[:, 0], enu[:, 1]), enu[:, 2], atol=1e-12)
    # North points towards the north pole (+Z) in the tangent plane.
    assert enu[2, 1] > 0


@pytest.mark.parametrize("rho", [0.0, 5e4, 1e5])
def test_scale_factor(rho):
    # Numerical scale factor: ground distance vs map distance along a meridian.
    lat, lon = frames.stereo_to_latlon(0.0, rho)
    d = 1.0
    lat2, _ = frames.stereo_to_latlon(0.0, rho + d)
    ground = np.deg2rad(abs(lat2 - lat)) * frames.MOON_RADIUS_M
    assert frames.stereo_scale_factor(0.0, rho) == pytest.approx(d / ground, rel=1e-6)
