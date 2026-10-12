"""Moon-fixed frames and the south polar stereographic map projection.

Conventions
-----------
* Body frame: Moon-fixed Cartesian, origin at the Moon centre, +Z along the spin axis
  (north), +X towards 0 deg longitude, +Y towards 90 deg E. The LOLA polar products use the
  MOON_ME (mean Earth / polar axis, DE421) realisation of this frame.
* Reference body: sphere of radius ``MOON_RADIUS_M`` (1737.4 km). LOLA heights are radii
  minus this value.
* Map projection: south polar stereographic on that sphere, true scale at the pole,
  central meridian 0. Easting ``x = rho * sin(lon)``, northing ``y = rho * cos(lon)``,
  ``rho = 2 R tan(pi/4 + lat/2)``. This matches the PGDA and PDS LOLA polar GeoTIFF/IMG
  products (verified against PROJ in tests).
* Local level frame (ENU): ``up = p / |p|``, ``east = normalize(z_hat x up)``,
  ``north = up x east``. This is the convention used by the StereoLunar trajectory
  generator to place the Sun.
"""

from __future__ import annotations

import numpy as np

MOON_RADIUS_M = 1_737_400.0


def latlon_to_stereo(lat_deg, lon_deg, radius: float = MOON_RADIUS_M):
    """Planetocentric latitude/longitude (deg) to south polar stereographic x, y (m)."""
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    rho = 2.0 * radius * np.tan(np.pi / 4.0 + lat / 2.0)
    return rho * np.sin(lon), rho * np.cos(lon)


def stereo_to_latlon(x, y, radius: float = MOON_RADIUS_M):
    """South polar stereographic x, y (m) to planetocentric latitude/longitude (deg)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    rho = np.hypot(x, y)
    lat = 2.0 * np.arctan(rho / (2.0 * radius)) - np.pi / 2.0
    lon = np.arctan2(x, y)
    return np.rad2deg(lat), np.rad2deg(lon)


def stereo_to_body(x, y, h, radius: float = MOON_RADIUS_M) -> np.ndarray:
    """Stereographic x, y (m) and height above the reference sphere (m) to body-frame XYZ.

    Returns an array of shape ``(..., 3)``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    rho2 = x * x + y * y
    four_r2 = 4.0 * radius * radius
    # Inverse stereographic projection from the north pole onto the unit sphere, written
    # without trigonometry: the unit vector for (x, y) on a sphere of radius R projected
    # from the antipode (north pole) onto the tangent plane at the south pole.
    denom = four_r2 + rho2
    # Map easting x grows towards lon = 90 deg (+Y_body); map northing y towards lon = 0 (+X_body).
    u_body_x = 4.0 * radius * y / denom
    u_body_y = 4.0 * radius * x / denom
    u_body_z = -(four_r2 - rho2) / denom
    r = radius + h
    return np.stack([u_body_x * r, u_body_y * r, u_body_z * r], axis=-1)


def body_to_stereo(p, radius: float = MOON_RADIUS_M):
    """Body-frame XYZ (..., 3) to stereographic x, y (m) and height above the sphere (m)."""
    p = np.asarray(p, dtype=np.float64)
    norm = np.linalg.norm(p, axis=-1)
    # Projection from the north pole onto the plane tangent at the south pole:
    # horizontal offset = 2R / (1 - sin(lat)) * (cos(lat) cos(lon), cos(lat) sin(lon)).
    scale = 2.0 * radius / (1.0 - p[..., 2] / norm)
    x = scale * p[..., 1] / norm
    y = scale * p[..., 0] / norm
    return x, y, norm - radius


def body_to_latlon(p):
    """Body-frame XYZ to planetocentric latitude, longitude (deg) and radius (m)."""
    p = np.asarray(p, dtype=np.float64)
    r = np.linalg.norm(p, axis=-1)
    lat = np.rad2deg(np.arcsin(p[..., 2] / r))
    lon = np.rad2deg(np.arctan2(p[..., 1], p[..., 0]))
    return lat, lon, r


def enu_basis(p) -> np.ndarray:
    """East, north, up unit vectors at body-frame point ``p``.

    Returns a 3x3 matrix whose columns are east, north and up. Uses the StereoLunar
    generator convention ``east = z_hat x up`` which is undefined exactly at the poles.
    """
    p = np.asarray(p, dtype=np.float64)
    up = p / np.linalg.norm(p)
    east = np.cross([0.0, 0.0, 1.0], up)
    n = np.linalg.norm(east)
    if n < 1e-12:
        east = np.array([0.0, 1.0, 0.0])
    else:
        east = east / n
    north = np.cross(up, east)
    return np.column_stack([east, north, up])


def stereo_scale_factor(x, y, radius: float = MOON_RADIUS_M):
    """Point scale factor k of the polar stereographic projection (map metres per ground metre)."""
    rho2 = np.asarray(x, dtype=np.float64) ** 2 + np.asarray(y, dtype=np.float64) ** 2
    return 1.0 + rho2 / (4.0 * radius * radius)
