"""
Low-level 3D gridding primitives and lightcone↔box orchestration.

Pure functions handle enclosing-box geometry, mass assignment, interlacing,
and Fourier windows. :class:`LightconeGriddingMixin` provides the stateful
sky↔box pipeline used by :class:`~meer21cm.power.PowerSpectrum`.
"""

import inspect
import logging
from collections.abc import Iterable

import healpy as hp
import numpy as np
from astropy import units
from astropy.cosmology import Planck18

from .util import (
    angle_in_range,
    f_21,
    find_ch_id,
    freq_to_redshift,
    get_nd_slicer,
    radec_to_indx,
    real_dtype_from_array,
    redshift_to_freq,
    tagging,
)

logger = logging.getLogger(__name__)

allowed_window_scheme = ("nnb", "cic", "tsc", "pcs")


def minimum_enclosing_box_of_lightcone(
    ra_arr,
    dec_arr,
    freq,
    cosmo=Planck18,
    ang_unit="deg",
    tile=True,
    return_coord=False,
    buffkick=0.0,
    rot_mat=None,
):
    """
    This functions finds a rotational axis to rotate the sky vectors of input coordinates so that the (crude) mean of the coordinates is at (0,0,1), and then finds the enclosing cuboid box for the coordinates. The box is not really minimum but should be quite optimal.

    The function also returns a rotational matrix for rotating the coordinates in the cuboid back to the sky positions. For any point in the box ``pos = np.array([x,y,z])``, you can find its RA and Dec by performing the rotation

    .. highlight:: python
    .. code-block:: python

        vec = inv_rot @ pos
        vec /= np.sqrt(np.sum(vec**2))
        ra_pos, dec_pos = hp.vec2ang(vec,lonlat=True)


    Parameters
    ----------
        ra_arr: ``numpy`` array.
            The RA of the coordinates
        dec_arr: ``numpy`` array.
            The Dec of the coordinates
        freq: ``numpy`` array.
            The frequencies of the coordinates.
        cosmo: :class:`astropy.cosmology.Cosmology` object, default `astropy.cosmology.Planck18`.
            The input cosmology for converting frequencies to los length.
        ang_unit: str or :class:`astropy.units.Unit`
            The unit of the input angular coordinates.
        tile: bool, default True.
            Whether to tile the input cooridnates so that the output is a meshgrid of input angular coordinates and frequencies.
        return_coord: bool, default True.
            If True, also returns the corrosponding (x,y,z) coordinate of the input coordinates.
        buffkick: float, default 0.0.
            The box is extended by ``buffkick`` on each end of each dimension.
        rot_mat: ``numpy`` array, default None.
            If specified, override the rotation matrix calculated from the mean cooridnate.



    Returns
    -------
        x_min: float.
            The origin of the box along x-axis.
        y_min: float.
            The origin of the box along y-axis.
        z_min: float.
            The origin of the box along z-axis.
        x_len: float.
            The length of the box along x-axis.
        y_len: float.
            The length of the box along y-axis.
        z_len: float.
            The length of the box along z-axis.
        inv_rot: ``numpy`` array.
            The rotational matrix to rotate the box back to the sky positions.
        pos_arr: ``numpy'' array.
            Only returns if ``return_coord = True''.
            The Cartesian coordinates of the input ra and dec.

    """
    ra_arr = (ra_arr.ravel() * units.Unit(ang_unit)).to("deg").value
    dec_arr = (dec_arr.ravel() * units.Unit(ang_unit)).to("deg").value
    vec_arr = hp.ang2vec(ra_arr, dec_arr, lonlat=True)
    mean_vec = vec_arr.mean(axis=0)
    if rot_mat is None:
        rot_mat = find_rotation_matrix(mean_vec)
    z_arr = f_21 / freq.ravel() - 1
    # rotate so that centre of field is the line-of-sight [0,0,1]
    vec_arr = np.einsum("ab,ib->ia", rot_mat, vec_arr)
    comov_dist_arr = cosmo.comoving_distance(z_arr).value
    if tile:
        pos_arr = vec_arr[:, None, :] * comov_dist_arr[None, :, None]
    else:
        pos_arr = vec_arr * comov_dist_arr[:, None]
    pos_arr = pos_arr.reshape((-1, 3))
    x_min, y_min, z_min = pos_arr.min(axis=0) - buffkick
    x_max, y_max, z_max = pos_arr.max(axis=0) + buffkick
    inv_rot = np.linalg.inv(rot_mat)
    result = (x_min, y_min, z_min, x_max - x_min, y_max - y_min, z_max - z_min, inv_rot)
    if return_coord:
        result += (pos_arr,)
    return result


def find_rotation_matrix(vec):
    r"""
    find the rotation matrix to rotate the input vector to (0,0,1).

    Note that in 3D space, the rotation is not unique. For simplicity, this function first finds the rotational matrix so that the vector (x,y,z) is first rotated to :math:`(\sqrt{x^2+y^2},0,z)`, and then find another matrix to rotate the vector to (0,0,1).

    Parameters
    ----------
        vec: ``numpy`` array.
            The input unit vector

    Returns
    -------
        rot_mat: ``numpy`` array.
            The rotational matrix so that ``rot_mat @ vec`` is ``np.array([0,0,1])``.
    """
    theta_rot = np.arctan2(vec[1], vec[0])
    rot_mat_1 = np.array(
        [
            [np.cos(-theta_rot), -np.sin(-theta_rot), 0],
            [np.sin(-theta_rot), np.cos(-theta_rot), 0],
            [0, 0, 1],
        ]
    )
    inter_vec = rot_mat_1 @ vec
    phi_rot = -np.arctan2(inter_vec[0], inter_vec[2])
    rot_mat_2 = np.array(
        [
            [np.cos(-phi_rot), 0, -np.sin(-phi_rot)],
            [0, 1, 0],
            [np.sin(-phi_rot), 0, np.cos(-phi_rot)],
        ]
    )
    return rot_mat_2 @ rot_mat_1


def fourier_window_for_assignment(
    num_mesh,
    window="nnb",
):
    r"""
    Calculate the effective window function in Fourier space from mass assignment scheme
    that sample continueous fields to discrete grids.

    The window function can be written as [1]

    .. math::
        W(k_x,k_y,k_z) = \Bigg({\rm sinc}\bigg(\frac{k_x H_x}{2}\bigg)
        {\rm sinc}\bigg(\frac{k_y H_y}{2}\bigg)
        {\rm sinc}\bigg(\frac{k_z H_z}{2}\bigg)\Bigg)^p,


    where :math:`k_{x,y,z}` is the wavenumber of the grid in Fourier space
    and :math:`H_{x,y,z}` is the length of the grid in real space.
    :math:`p` is the power index related to the mass assignment scheme, and
    is equal to [1,2,3,4] for [nnb,cic,tsc,pcs]

    Parameters
    ----------
        num_mesh: list
            The number of grids on each side
        window: str, default "nnb".
            The mass assignment scheme

    Returns
    -------
        window_in_fourier: ``numpy`` array
            The window function in Fourier space

    References
    ----------
    .. [1] Sefusatti, E. et al.,
        "Accurate Estimators of Correlation Functions in Fourier Space",
        https://ui.adsabs.harvard.edu/abs/2016MNRAS.460.3624S.
    """
    p = float(allowed_window_scheme.index(window) + 1)
    wx, wy, wz = [np.sinc(np.fft.fftfreq(num_mesh[i])) for i in range(3)]
    window_in_fourier = (
        wx[:, None, None] * wy[None, :, None] * wz[None, None, : num_mesh[-1] // 2 + 1]
    ) ** p
    return window_in_fourier


def compensate_grid_window_effects(
    field_in_real_space,
    grid_scheme="nnb",
):
    """
    Apply correction to cancel the windowing effects from
    discretization of fields into grids.
    """
    num_mesh = field_in_real_space.shape
    window = fourier_window_for_assignment(
        num_mesh,
        grid_scheme,
    )
    field_in_fourier_space = np.fft.rfftn(field_in_real_space)
    field_in_fourier_space /= window
    field_compensated = np.fft.irfftn(
        field_in_fourier_space, axes=(0, 1, 2), s=field_in_real_space.shape
    )
    return field_compensated


def interlace_two_fields(
    real_field_1,
    real_field_2,
    shift,
):
    """
    Interlacing two fields, where one is the shifted version of the other.

    Parameters
    ----------
        real_field_1: array-like.
            The first field for interlacing
        real_field_2: array-like.
            The second field for interlacing, which should be a shifted version of the first.
        shift: float.
            The shift of the field when performing Fourier transform, in the unit of cell size.

    Returns
    -------
        interlaced_field: array-like.
            The interlaced field.
    """
    box_ndim = real_field_1.shape
    fourier_field_1 = np.fft.fftn(real_field_1)
    fourier_field_2 = np.fft.fftn(real_field_2)
    kH_2 = [np.fft.fftfreq(box_ndim[i]) for i in range(3)]
    kH_2 = np.array(np.meshgrid(*kH_2, indexing="ij"))
    exp_term = np.prod(np.exp(-2 * 1j * shift * kH_2), axis=0)
    fourier_field_1 = (fourier_field_1 + exp_term * fourier_field_2) / 2
    return np.fft.ifftn(fourier_field_1).real


def project_function(
    s_arr,
    grid_scheme="nnb",
):
    """
    Return the weighting function for the given mass assignment scheme and input 1D distance.
    The distance is in the unit of the cell size.

    Parameters
    ----------
        s_arr: float array.
            The distance in the unit of the cell size.
        grid_scheme: str, default 'nnb'
            The mass assignment scheme.

    Returns
    -------
        weight_arr: float array.
            The weighting function.
    """
    s_arr = np.abs(s_arr)
    p = allowed_window_scheme.index(grid_scheme)
    if p == 0:
        return (s_arr <= 0.5).astype("float")
    elif p == 1:
        return (1 - s_arr) * (s_arr <= 1)
    elif p == 2:
        result = (3 / 4 - s_arr**2) * (s_arr <= 0.5) + (0.5 * (1.5 - s_arr) ** 2) * (
            s_arr < 1.5
        ) * (s_arr > 0.5)
        return result
    elif p == 3:
        result = (4 - 6 * s_arr**2 + 3 * s_arr**3) / 6 * (s_arr <= 1) + (
            2 - s_arr
        ) ** 3 / 6 * (s_arr < 2) * (s_arr > 1)
        return result


def particle_to_mesh_distance(
    particle_pos,
    box_len,
    box_ndim,
):
    """
    Calculate the distance between particles and the nearest mesh center.
    The distance is in the unit of the cell size.
    For particles outside the box, the nearest mesh center is the one on the boundary.

    Parameters
    ----------
        particle_pos: array.
            The coordinates of the particles.
        box_len: array.
            The length of the box on each side
        box_ndim: array.
            The number of grids on each side

    Returns
    -------
        dist: array.
            The distance between the particles and the nearest mesh center.
        indx_grid: array.
            The index of the nearest mesh center.
    """
    box_resol = box_len / box_ndim
    mesh_edges = [np.linspace(0, box_len[i], box_ndim[i] + 1) for i in range(3)]
    mesh_cen = [(mesh_edges[i][:-1] + mesh_edges[i][1:]) / 2 for i in range(3)]
    indx_grid = []
    for i in range(3):
        indx_i = np.digitize(particle_pos[:, i], mesh_edges[i]) - 1
        # if particle is outside the box, set to 0 or box_ndim-1 as appropriate
        indx_i[indx_i < 0] = 0
        indx_i[indx_i >= box_ndim[i]] = box_ndim[i] - 1
        indx_grid += [
            indx_i,
        ]
    mesh_cen = [
        (np.linspace(0, box_ndim[i] - 1, box_ndim[i]) + 0.5) * box_resol[i]
        for i in range(3)
    ]
    particle_pos_mesh = [mesh_cen[i][indx_grid[i]] for i in range(3)]
    particle_pos_mesh = np.array(particle_pos_mesh).T
    return (particle_pos - particle_pos_mesh) / box_resol[None, :], indx_grid


def project_particle_to_regular_grid(
    particle_pos,
    box_len,
    box_ndim,
    grid_scheme="nnb",
    particle_mass=None,
    particle_weights=None,
    average=True,
    shift=0.0,
    compensate=False,
):
    """
    Project particles into a regular grid with a certain mass assignment scheme.

    Parameters
    ----------
        particle_pos: array.
            The coordinates of the particles.
        box_len: array.
            The length of the box on each side
        box_ndim: array.
            The number of grids on each side
        grid_scheme: str, default 'nnb'
            The mass assignment scheme.
        particle_mass: array, default None.
            The mass of each particle.
        particle_weights: array, default None.
            The weights of each particle.
        average: bool, default True.
            The grid values are weighted averages of the particles if True
            and weighted sums of the particles if False.
        shift: float, default 0.0.
            Shift the position of the particles by the same amount in all directions,
            in the unit of cell size.

    Returns
    -------
        mesh_mass: array.
            The mass of each grid.
        mesh_weights: array.
            The weights of each grid.
        mesh_counts: array.
            The effective number of particles in each grid.
    """
    p = allowed_window_scheme.index(grid_scheme)
    if particle_mass is None:
        particle_mass = np.ones(len(particle_pos))
    if particle_weights is None:
        particle_weights = np.ones(len(particle_pos))
    box_resol = box_len / box_ndim
    mesh_mass = np.zeros(box_ndim)
    mesh_weights = np.zeros(box_ndim)
    mesh_counts = np.zeros(box_ndim)
    par_pos = particle_pos + shift * box_resol[None, :]
    particle_s, indx_grid = particle_to_mesh_distance(par_pos, box_len, box_ndim)
    indx_grid = np.array(indx_grid).T
    shift_limit = np.floor(p / 2 + 0.5)
    shift_mat = np.meshgrid(
        np.arange(-shift_limit, shift_limit + 1),
        np.arange(-shift_limit, shift_limit + 1),
        np.arange(-shift_limit, shift_limit + 1),
        indexing="ij",
    )
    shift_mat = np.array([shift_mat[i].ravel() for i in range(3)]).T
    for shift in shift_mat:
        s_shift = particle_s + shift[None, :]
        grid_func_shift = project_function(s_shift, grid_scheme)
        indx_shift = (indx_grid - shift[None, :]).astype("int")
        indx_sel = np.prod(indx_shift >= 0, axis=1)
        indx_sel *= np.prod(indx_shift < box_ndim[None, :], axis=1)
        indx_sel = indx_sel.astype("bool")
        np.add.at(
            mesh_mass,
            tuple(indx_shift[indx_sel].T),
            (particle_mass * particle_weights * np.prod(grid_func_shift, axis=1))[
                indx_sel
            ],
        )
        np.add.at(
            mesh_weights,
            tuple(indx_shift[indx_sel].T),
            (particle_weights * np.prod(grid_func_shift, axis=1))[indx_sel],
        )
        np.add.at(
            mesh_counts,
            tuple(indx_shift[indx_sel].T),
            np.prod(grid_func_shift, axis=1)[indx_sel],
        )
    if average:
        with np.errstate(divide="ignore", invalid="ignore"):
            mesh_mass = np.where(mesh_weights > 0, mesh_mass / mesh_weights, 0)
    if compensate:
        mesh_mass = compensate_grid_window_effects(
            mesh_mass,
            grid_scheme,
        )
    return mesh_mass, mesh_weights, mesh_counts


def rotation_matrix_to_radec0(ra, dec):
    """
    Find the rotation matrix to rotate the input point at (ra, dec) to (0, 0), by first
    rotating to (0, dec) and then to (0, 0).
    """
    # step 1: rotate to RA=0
    rot_mat_1 = np.array(
        [
            [np.cos(np.deg2rad(ra)), np.sin(np.deg2rad(ra)), 0],
            [-np.sin(np.deg2rad(ra)), np.cos(np.deg2rad(ra)), 0],
            [0, 0, 1],
        ]
    )
    # step 2: rotate to dec=0
    rot_mat_2 = np.array(
        [
            [np.cos(np.deg2rad(dec)), 0, np.sin(np.deg2rad(dec))],
            [0, 1, 0],
            [-np.sin(np.deg2rad(dec)), 0, np.cos(np.deg2rad(dec))],
        ]
    )
    return rot_mat_2 @ rot_mat_1


def sky_partition_for_radecrange(
    ra_range, dec_range, nside_out=128, nside_in=1024, dec_pad=0
):
    """
    Find a partition of the sky, so that each patch can be rotated to cover the specified RA and Dec range.

    Parameters
    ----------
    ra_range: array_like
        The range of RA to cover.
    dec_range: array_like
        The range of Dec to cover.
    nside_out: int, default 128
        The HEALPix NSIDE of the output map pixel id.
    nside_in: int, default 1024
        The HEALPix NSIDE of the map pixel id for inner calculation.
    dec_pad: int, default 0
        The number of extra rows to pad in Dec.
        Increasing this number will result in patches overlapping with each other.

    Returns
    -------
    pix_id_for_patch_i: list
        The list of pixel id for each patch.
    rot_mat_for_patch_i: list
        The list of rotation matrix for each patch, to rotate the patch back to cover the range.
    """
    npix = hp.nside2npix(nside_in)
    ra_grid, dec_grid = hp.pix2ang(nside_in, np.arange(npix), lonlat=True)
    selection_grid = angle_in_range(ra_grid, ra_range[0], ra_range[1]) * angle_in_range(
        dec_grid, dec_range[0], dec_range[1]
    )
    ra_region = ra_grid[selection_grid]
    dec_region = dec_grid[selection_grid]
    vec_region = hp.ang2vec(ra_region, dec_region, lonlat=True)
    vec_mean = vec_region.mean(axis=0)
    ra_mean, dec_mean = hp.vec2ang(vec_mean, lonlat=True)
    ra_mean = ra_mean[0]
    dec_mean = dec_mean[0]
    # rotate range to ra=0, dec=0
    rot_mat_0 = rotation_matrix_to_radec0(ra_mean, dec_mean)
    vec_region_rot = np.dot(rot_mat_0, vec_region.T)
    pix_region_rot = hp.vec2pix(
        nside_in, vec_region_rot[0], vec_region_rot[1], vec_region_rot[2]
    )
    ra_region_rot, dec_region_rot = hp.pix2ang(nside_in, pix_region_rot, lonlat=True)
    # find the enclosing rectangle
    ra_temp = ra_region_rot.copy()
    ra_temp[ra_temp > 180] -= 360
    ra_range_0 = [-np.abs(ra_temp).max(), np.abs(ra_temp).max()]
    dec_range_0 = [-np.abs(dec_region_rot).max(), np.abs(dec_region_rot).max()]
    delta_dec = dec_range_0[1] - dec_range_0[0]
    delta_ra = ra_range_0[1] - ra_range_0[0]
    ra_range_0 = [-np.abs(ra_temp).max(), np.abs(ra_temp).max()]
    dec_range_0 = [-np.abs(dec_region_rot).max(), np.abs(dec_region_rot).max()]
    selection_grid_0 = angle_in_range(
        ra_grid, ra_range_0[0], ra_range_0[1]
    ) * angle_in_range(dec_grid, dec_range_0[0], dec_range_0[1])
    ra_region_0 = ra_grid[selection_grid_0]
    dec_region_0 = dec_grid[selection_grid_0]
    vec_region_0 = hp.ang2vec(ra_region_0, dec_region_0, lonlat=True)
    dec_loop_num = int(90 * np.cos(np.deg2rad(ra_range_0[0])) // delta_dec) + dec_pad
    delta_dec_loop = 90 / max(dec_loop_num, 1)
    pix_id_for_patch_i = []
    rot_mat_for_patch_i = []
    for j in range(-dec_loop_num, dec_loop_num + 1):
        delta_dec_j = delta_dec_loop * j
        ra_loop_num_j = max(
            int(
                360
                * np.cos(np.deg2rad(np.abs(delta_dec_j) + delta_dec_loop / 2))
                // delta_ra
            ),
            1,
        )
        if ra_loop_num_j == 1 and np.abs(j) != dec_loop_num:
            ra_loop_num_j = 2
        for i in range(0, ra_loop_num_j):
            delta_ra_i = 360 / (ra_loop_num_j) * i
            rot_mat = np.linalg.inv(rotation_matrix_to_radec0(delta_ra_i, delta_dec_j))
            vec_region_rot = np.dot(rot_mat, vec_region_0.T)
            pix_region_rot = hp.vec2pix(
                nside_out, vec_region_rot[0], vec_region_rot[1], vec_region_rot[2]
            )
            pix_id_for_patch_i.append(pix_region_rot)
            rot_mat_for_patch_i.append(
                np.linalg.inv(rot_mat_0) @ np.linalg.inv(rot_mat)
            )
    return pix_id_for_patch_i, rot_mat_for_patch_i


def shot_noise_correction_from_gridding(
    box_ndim,
    grid_scheme,
):
    """
    Calculate the multiplicative correction from gridding to the shot noise.
    Support 'nnb', 'cic' and 'tsc'.
    The correction is taken from Jing (2005), astro-ph/0409240.

    Parameters
    ----------
        box_ndim: array.
            The number of grids on each side.
        grid_scheme: str.
            The mass assignment scheme.

    Returns
    -------
        shot_noise_correction: array.
            The multiplicative correction from gridding to the shot noise.
    """
    p = allowed_window_scheme.index(grid_scheme)
    if p == 0:
        return np.ones([box_ndim[0], box_ndim[1], box_ndim[2] // 2 + 1])
    sinpikiHover2 = [np.sin(np.fft.fftfreq(box_ndim[i]) * np.pi) for i in range(2)]
    sinpikiHover2.append(np.sin(np.fft.rfftfreq(box_ndim[2]) * np.pi))
    if p == 1:
        ci = [1 - 2 / 3 * sinpikiHover2[i] ** 2 for i in range(3)]
        ci = ci[0][:, None, None] * ci[1][None, :, None] * ci[2][None, None, :]
    if p == 2:
        ci = [
            1 - sinpikiHover2[i] ** 2 + 2 / 15 * sinpikiHover2[i] ** 4 for i in range(3)
        ]
        ci = ci[0][:, None, None] * ci[1][None, :, None] * ci[2][None, None, :]
    return ci


def _random_radec_in_radec_rectangles(ra0, dec0, half_ra, half_dec, rng):
    """
    Draw points uniformly in solid angle inside RA–Dec rectangles.

    Each rectangle is centred at ``(ra0, dec0)`` with half-widths
    ``half_ra`` and ``half_dec`` in degrees. Declination is sampled from
    the ``sin(δ)`` measure so the density is uniform on the sphere.
    """
    ra0 = np.asarray(ra0, dtype=float)
    dec0 = np.asarray(dec0, dtype=float)
    dec_lo = np.clip(np.radians(dec0 - half_dec), -0.5 * np.pi, 0.5 * np.pi)
    dec_hi = np.clip(np.radians(dec0 + half_dec), -0.5 * np.pi, 0.5 * np.pi)
    u = rng.uniform(0.0, 1.0, size=ra0.shape)
    sin_dec = np.sin(dec_lo) + u * (np.sin(dec_hi) - np.sin(dec_lo))
    sin_dec = np.clip(sin_dec, -1.0, 1.0)
    dec = np.degrees(np.arcsin(sin_dec))
    ra = ra0 + rng.uniform(-half_ra, half_ra, size=ra0.shape)
    return ra, dec


def _random_radec_in_healpix_pixels(nside, ipix, rng):
    """
    Draw points uniformly in solid angle inside HEALPix pixels.

    Uses nested fine-pixel subdivision (same idea as astrotools
    ``rand_vec_in_pix``): each coarse RING pixel is split into
    ``4**(29 - log2(nside))`` nested subpixels at ``nside=2**29``, and one
    subpixel centre is chosen uniformly.
    """
    nside = int(nside)
    ipix = np.asarray(ipix, dtype=np.int64).ravel()
    n_order = int(round(np.log2(nside)))
    if 2**n_order != nside:
        raise ValueError(f"hp_nside must be a power of 2, got {nside}")
    nest_pix = hp.ring2nest(nside, ipix)
    n_up = 29 - n_order
    n_sub = 4**n_up
    i_up = nest_pix * n_sub + rng.integers(0, n_sub, size=ipix.size)
    lon, lat = hp.pix2ang(2**29, i_up, nest=True, lonlat=True)
    return np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)


class LightconeGriddingMixin:
    """
    Mixin providing lightcone↔rectangular-box gridding for power spectrum objects.

    The :class:`LightconeGriddingMixin` object is only meant to be
    be used as a mixin for the :class:`PowerSpectrum` class.
    """

    @property
    def seed(self):
        """
        Seed value for RNG calls throughout the instance.
        """
        return self._seed

    @seed.setter
    def seed(self, pseed):
        self._seed = pseed
        if "seed_dep_attr" in dir(self):
            self.clean_cache(self.seed_dep_attr)

    @property
    def box_buffkick(self):
        """
        The buffer kick for the box on each side when gridding. In the unit of Mpc.
        """
        return self._box_buffkick

    @box_buffkick.setter
    def box_buffkick(self, value):
        if not isinstance(value, Iterable):
            self._box_buffkick = np.array([value, value, value])
        else:
            self._box_buffkick = np.array(value)
        init_attr = [
            "_box_origin",
            "_counts_in_box",
        ]
        logger.debug(f"cleaning cache of {init_attr} due to resetting box_buffkick")
        for attr in init_attr:
            setattr(self, attr, None)

    @property
    def num_particle_per_pixel(self):
        """
        The number of random sampling particles for each sky map pixel.
        """
        return self._num_particle_per_pixel

    @num_particle_per_pixel.setter
    def num_particle_per_pixel(self, value):
        self._num_particle_per_pixel = int(value)
        init_attr = [
            "_box_origin",
            "_counts_in_box",
        ]
        logger.debug(
            f"cleaning cache of {init_attr} due to resetting num_particle_per_pixel"
        )
        for attr in init_attr:
            setattr(self, attr, None)

    @property
    def interlace_shift(self):
        """
        The length in the unit of grid cell size for
        shifting the gridded field for interlacing.
        0 corresponds to no interlacing.
        """
        return self._interlace_shift

    @interlace_shift.setter
    def interlace_shift(self, value):
        self._interlace_shift = value

    @property
    def downres_factor_transverse(self):
        """
        The down-sampling factor for the transverse direction of the rectangular box for gridding.
        The box resolution is then multiplied by this factor from the resolution of the sky map pixel
        specified by ``pix_resol_in_mpc``.
        For example, if ``pix_resol_in_mpc`` is 0.1 Mpc, and ``downres_factor_transverse`` is 2.0,
        the box resolution will be 0.2 Mpc.
        """
        return self._downres_factor_transverse

    @downres_factor_transverse.setter
    def downres_factor_transverse(self, value):
        self._downres_factor_transverse = value
        # clean cache
        init_attr = [
            "_box_origin",
            "_counts_in_box",
        ]
        logger.debug(
            f"cleaning cache of {init_attr} due to resetting downres_factor_transverse"
        )
        for attr in init_attr:
            setattr(self, attr, None)

    @property
    def downres_factor_radial(self):
        """
        The down-sampling factor for the radial direction of the rectangular box for gridding.
        The box resolution is then multiplied by this factor from the resolution of the frequency channel
        specified by ``los_resol_in_mpc``.
        For example, if ``los_resol_in_mpc`` is 0.1 Mpc, and ``downres_factor_radial`` is 2.0,
        the box resolution will be 0.2 Mpc.
        """
        return self._downres_factor_radial

    @downres_factor_radial.setter
    def downres_factor_radial(self, value):
        self._downres_factor_radial = value
        # clean cache
        init_attr = [
            "_box_origin",
            "_counts_in_box",
        ]
        logger.debug(
            f"cleaning cache of {init_attr} due to resetting downres_factor_radial"
        )
        for attr in init_attr:
            setattr(self, attr, None)

    @property
    def counts_in_box(self):
        """
        The counts of the map cube voxels in the rectangular box.
        """
        if self._counts_in_box is None:
            self._counts_in_box = self.get_counts_in_box()
        return self._counts_in_box

    @property
    def flat_sky(self):
        """
        Whether to use flat sky approximation.
        If True, no proper projection and sky rotation is considered.
        Instead, the sky map cube is assumed to be a rectangular grid
        with equal voxel size specified by ``pix_resol_in_mpc`` and
        ``los_resol_in_mpc``.
        """
        return self._flat_sky

    @flat_sky.setter
    def flat_sky(self, value):
        self._flat_sky = bool(value)
        # clean cache
        init_attr = [
            "_box_origin",
            "_counts_in_box",
        ]
        logger.debug(f"cleaning cache of {init_attr} due to resetting flat_sky")
        for attr in init_attr:
            setattr(self, attr, None)

    @property
    def flat_sky_padding(self):
        """
        Pad the rectangular box in the flat sky approximation.

        The input should be a list of 3 integers, corresponding to number of padding cells along
        each dimension in both directions.
        For example, [1,1,1] will pad 2x2x2 cells.
        """
        return self._flat_sky_padding

    @flat_sky_padding.setter
    def flat_sky_padding(self, value):
        self._flat_sky_padding = value
        # clean cache
        init_attr = [
            "_box_origin",
            "_counts_in_box",
        ]
        logger.debug(f"cleaning cache of {init_attr} due to resetting flat_sky_padding")
        for attr in init_attr:
            setattr(self, attr, None)

    @property
    def box_origin(self):
        """
        The coordinate of the origin of the box in Mpc.
        See :func:`meer21cm.grid.minimum_enclosing_box_of_lightcone`
        for definition.
        """
        return self._box_origin

    @box_origin.setter
    def box_origin(self, value):
        self._box_origin = np.array(value)

    @property
    def rot_mat_sky_to_box(self):
        """
        The rotational matrix from spheircal cooridnate to regular box.

        See :func:`meer21cm.grid.minimum_enclosing_box_of_lightcone`
        for definition.
        """
        return self._rot_mat_sky_to_box

    @property
    def pix_coor_in_cartesian(self):
        """
        The cartesian coordinate of the pixels in Mpc.
        """
        return self._pix_coor_in_cartesian

    @property
    def pix_coor_in_box(self):
        """
        The cartesian coordinate of the pixels in Mpc,
        shifted so that the origin is the origin of the enclosing box.
        """
        return self.pix_coor_in_cartesian - self.box_origin[None, :]

    def use_flat_sky_box(self, flat_sky_padding=None):
        """
        Use flat sky approximation to calculate the box dimensions.

        Parameters
        ----------
        flat_sky_padding: list, default None
            The padding for the flat sky box.
            If None, use the class attribute ``flat_sky_padding``.
        """
        if flat_sky_padding is None:
            flat_sky_padding = self.flat_sky_padding
        logger.debug(f"using flat sky box with padding {flat_sky_padding}")
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: setting self.box_ndim, self.box_len, self.box_origin"
        )
        self.box_ndim = np.array(self.data.shape) + 2 * np.array(flat_sky_padding)
        self.box_len = np.array(self.box_ndim) * np.array(
            [
                self.pix_resol_in_mpc,
                self.pix_resol_in_mpc,
                self.los_resol_in_mpc,
            ]
        )
        # flat sky does not have rotation so there is no box_origin
        self.box_origin = np.array([0, 0, 0])
        if self.model_k_from_field:
            logger.info(
                f"{inspect.currentframe().f_code.co_name}: "
                "setting the model self.kmode and self.mumode to correspond to the field k-modes"
            )
            self.propagate_field_k_to_model()
        self._counts_in_box = None
        nu_ext = np.linspace(
            self.nu.min() - self.freq_resol * flat_sky_padding[2],
            self.nu.max() + self.freq_resol * flat_sky_padding[2],
            len(self.nu) + 2 * flat_sky_padding[2],
        )
        self._box_voxel_redshift = (
            np.ones(self.box_ndim) * freq_to_redshift(nu_ext)[None, None, :]
        )

    def get_enclosing_box(self, rot_mat=None):
        """
        invoke to calculate the box dimensions for enclosing all
        the map pixels.

        Parameters
        ----------
        rot_mat: np.ndarray, default None
            The rotational matrix from the sky to the box.
            If None, calculates the suitable rotation matrix automatically.
        """
        if self.flat_sky:
            self.use_flat_sky_box()
            if self.model_k_from_field:
                logger.info(
                    f"{inspect.currentframe().f_code.co_name}: "
                    "setting the model self.kmode and self.mumode to correspond to the field k-modes"
                )
                self.propagate_field_k_to_model()
            return 1
        ra = self.ra_map.copy()[self.W_HI.sum(-1) > 0]
        dec = self.dec_map.copy()[self.W_HI.sum(-1) > 0]
        logger.debug(f"calculating enclosing box for {len(ra)} particles")
        (
            _x_start,
            _y_start,
            _z_start,
            _x_len,
            _y_len,
            _z_len,
            rot_back,
            pos_arr,
        ) = minimum_enclosing_box_of_lightcone(
            ra,
            dec,
            self.nu,
            cosmo=self.astropy_cosmo_fiducial,
            return_coord=True,
            buffkick=self.box_buffkick,
            rot_mat=rot_mat,
        )
        logger.debug(
            f"{inspect.currentframe().f_code.co_name}: calculated enclosing box with size {_x_len} x {_y_len} x {_z_len}"
        )
        logger.info(
            f"{inspect.currentframe().f_code.co_name}: setting self.box_len, self.box_origin, self.box_ndim"
        )
        self._box_origin = np.array([_x_start, _y_start, _z_start])
        self._box_len = np.array(
            [
                _x_len,
                _y_len,
                _z_len,
            ]
        )
        self._rot_mat_sky_to_box = np.linalg.inv(rot_back)
        # random sample
        num_p = self.num_particle_per_pixel
        ra_sample = [
            ra,
        ] * num_p
        dec_sample = [
            dec,
        ] * num_p
        nu_sample = [
            self.nu,
        ] * num_p
        ra_sample = np.array(ra_sample)
        dec_sample = np.array(dec_sample)
        nu_sample = np.array(nu_sample)
        logger.debug(f"randomly sampled {num_p} particles in each pixel")
        rng = np.random.default_rng(seed=self.seed)
        rand_angle = rng.uniform(
            -self.pix_resol / 2, self.pix_resol / 2, size=(2,) + ra_sample[1:].shape
        )
        rand_nu = rng.uniform(
            -self.freq_resol / 2, self.freq_resol / 2, size=(1,) + nu_sample[1:].shape
        )
        ra_sample[1:] += rand_angle[0]
        dec_sample[1:] += rand_angle[1]
        nu_sample[1:] += rand_nu[0]
        pos_arr = [
            pos_arr,
        ]
        for i in range(1, num_p):
            (_, _, _, _, _, _, _, pos_arr_i) = minimum_enclosing_box_of_lightcone(
                ra_sample[i],
                dec_sample[i],
                nu_sample[i],
                cosmo=self.astropy_cosmo_fiducial,
                return_coord=True,
                buffkick=self.box_buffkick,
                rot_mat=self.rot_mat_sky_to_box,
            )
            pos_arr.append(pos_arr_i)
        pos_arr = np.array(pos_arr)
        pos_arr = pos_arr.reshape((-1, 3))

        self._pix_coor_in_cartesian = pos_arr
        downres = np.array(
            [
                self.downres_factor_transverse,
                self.downres_factor_transverse,
                self.downres_factor_radial,
            ]
        )
        pix_resol_in_mpc = self.pix_resol_in_mpc
        los_resol_in_mpc = self.los_resol_in_mpc
        box_resol = (
            np.array([pix_resol_in_mpc, pix_resol_in_mpc, los_resol_in_mpc]) * downres
        )
        ndim_rg = self.box_len / box_resol
        ndim_rg = ndim_rg.astype("int")
        for i in range(3):
            if ndim_rg[i] % 2 == 0:
                ndim_rg[i] += 1
        box_resol = self.box_len / ndim_rg
        self.box_ndim = ndim_rg
        logger.debug(
            f"calculated box resolution due to downres factor: {box_resol}, {downres}"
        )
        self._counts_in_box = None
        slicer = get_nd_slicer()
        vec = [(self.x_vec[i] + self.box_origin[i])[slicer[i]] for i in range(3)]
        vec_len = np.sqrt(vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2)
        self._box_voxel_redshift = self.z_as_func_of_comov_dist(vec_len)
        if self.model_k_from_field:
            logger.info(
                f"{inspect.currentframe().f_code.co_name}: "
                "setting the model self.kmode and self.mumode to correspond to the field k-modes"
            )
            self.propagate_field_k_to_model()

    def get_counts_in_box(self, partial_sel=None):
        """
        Grid the counts of the map cube voxels into the rectangular box, and return the
        effective counts per rectangular grid.

        Parameters
        ----------
        partial_sel: array, default None
            An additional selection function of the data on top of W_HI.
            Allows hacking for batch processing.

        Returns
        -------
        counts_in_grids: array.
            The counts of the map cube voxels in the rectangular box.
        """
        if self.flat_sky:
            counts_in_grids = self.w_HI
        else:
            pix_coor_orig = self.pix_coor_in_box.reshape(
                (self.num_particle_per_pixel, -1)
            )[0].reshape((-1, 3))
            num_pix = (self.W_HI.sum(-1) > 0).sum()
            if partial_sel is None:
                partial_sel = slice(None)
            pix_coor_orig = pix_coor_orig.reshape((num_pix, self.nu.size, 3))
            pix_coor_orig = pix_coor_orig[partial_sel].reshape((-1, 3))
            counts_in_grids, _, _ = project_particle_to_regular_grid(
                pix_coor_orig,
                self.box_len,
                self.box_ndim,
                grid_scheme=self.grid_scheme,
                particle_mass=self.w_HI[self.W_HI.sum(-1) > 0][partial_sel].ravel(),
                compensate=False,  # compensate should be at model level
                average=False,
            )
        return counts_in_grids

    def grid_data_to_field(self, flat_sky=None, partial_sel=None):
        """
        Grid the stored data map to a rectangular grid field.

        If flat_sky is True, no gridding is performed. Instead, the map cube
        dimensions are taken to be a rectangular grid, with the grid length
        corresponding to the pixel resolution on x/y and los frequency resolution
        as z.

        If flat_sky is False, the data is gridded onto a regular grid using the
        input grid scheme and performing the proper curved sky projection.

        The gridded field is stored as field_1 and the weights are stored as weights_1.

        Parameters
        ----------
        flat_sky: bool, default None
            If True, use flat sky approximation.
        partial_sel: array, default None
            An additional selection function of the data on top of W_HI.
            Allows hacking for batch processing.
        """
        if flat_sky is None:
            flat_sky = self.flat_sky
        if flat_sky:
            self.field_1 = self.data
            self.weights_1 = self.w_HI.astype(float)
            self.use_flat_sky_box(flat_sky_padding=[0, 0, 0])
            self.mean_center_1 = False
            self.unitless_1 = False
            self.include_sky_sampling = [True, False]
            self.compensate = False
            self.include_beam = [True, False]
            return self.field_1, self.weights_1, (self.weights_1 > 0).astype(float)
        if self.box_origin is None:
            self.get_enclosing_box()
        num_pix = (self.W_HI.sum(-1) > 0).sum()
        all_sel = np.arange(num_pix)
        if partial_sel is None:
            selected_pix = all_sel
        else:
            selected_pix = all_sel[partial_sel]
        batch_sel = np.array_split(selected_pix, self.batch_number)
        batch_sel = [sel for sel in batch_sel if sel.size > 0]
        if len(batch_sel) == 0:
            shape = tuple(self.box_ndim.tolist())
            hi_map_rg = np.zeros(shape, dtype=self.real_dtype)
            hi_weights_sum = np.zeros(shape, dtype=self.real_dtype)
            pixel_counts_sum = np.zeros(shape, dtype=self.real_dtype)
            self.field_1 = hi_map_rg
            self.weights_1 = pixel_counts_sum.astype(float)
            self.unitless_1 = False
            include_beam = np.array(self.include_beam)
            include_beam[0] = True
            self.include_beam = include_beam
            include_sky_sampling = np.array(self.include_sky_sampling)
            include_sky_sampling[0] = True
            self.include_sky_sampling = include_sky_sampling
            return hi_map_rg, hi_weights_sum, pixel_counts_sum

        map_weighted_sum = None
        hi_weights_sum = None
        pixel_counts_sum = None
        for sel in batch_sel:
            hi_map_i, hi_weights_i, pixel_counts_i = self._grid_data_to_field(sel)
            if map_weighted_sum is None:
                map_weighted_sum = np.zeros_like(hi_map_i)
                hi_weights_sum = np.zeros_like(hi_weights_i)
                pixel_counts_sum = np.zeros_like(pixel_counts_i)
            map_weighted_sum += hi_map_i * hi_weights_i
            hi_weights_sum += hi_weights_i
            pixel_counts_sum += pixel_counts_i
        with np.errstate(divide="ignore", invalid="ignore"):
            hi_map_rg = np.nan_to_num(map_weighted_sum / hi_weights_sum)
        self.field_1 = hi_map_rg
        self.weights_1 = pixel_counts_sum.astype(float)
        self.unitless_1 = False
        include_beam = np.array(self.include_beam)
        include_beam[0] = True
        self.include_beam = include_beam
        include_sky_sampling = np.array(self.include_sky_sampling)
        include_sky_sampling[0] = True
        self.include_sky_sampling = include_sky_sampling
        return hi_map_rg, hi_weights_sum, pixel_counts_sum

    def _grid_data_to_field(self, partial_sel):
        num_pix = (self.W_HI.sum(-1) > 0).sum()
        data_particle = self.data[self.W_HI.sum(-1) > 0]
        weights_particle = self.w_HI[self.W_HI.sum(-1) > 0]
        num_p = self.num_particle_per_pixel
        data_particle = [data_particle] * num_p
        weights_particle = [weights_particle] * num_p
        real_dtype = real_dtype_from_array(self.data)
        data_particle = np.asarray(data_particle, dtype=real_dtype)[
            :, partial_sel
        ].ravel()
        weights_particle = np.asarray(weights_particle, dtype=real_dtype)[
            :, partial_sel
        ].ravel()
        pix_coor_in_box = self.pix_coor_in_box.reshape(
            (num_p, num_pix, self.nu.size, 3)
        )
        pix_coor_in_box = pix_coor_in_box[:, partial_sel].reshape((-1, 3))
        hi_map_rg, hi_weights_rg, pixel_counts_hi_rg = project_particle_to_regular_grid(
            pix_coor_in_box,
            self.box_len,
            self.box_ndim,
            grid_scheme=self.grid_scheme,
            particle_mass=data_particle,
            particle_weights=weights_particle,
            compensate=False,  # compensate should be at model level
        )
        hi_map_rg2, _, _ = project_particle_to_regular_grid(
            pix_coor_in_box,
            self.box_len,
            self.box_ndim,
            grid_scheme=self.grid_scheme,
            particle_mass=data_particle,
            particle_weights=weights_particle,
            compensate=False,  # compensate should be at model level
            shift=self.interlace_shift,
        )
        hi_map_rg = interlace_two_fields(hi_map_rg, hi_map_rg2, self.interlace_shift)
        hi_map_rg = np.asarray(hi_map_rg, dtype=real_dtype)
        hi_weights_rg = np.asarray(hi_weights_rg, dtype=real_dtype)
        return hi_map_rg, hi_weights_rg, pixel_counts_hi_rg

    def grid_gal_to_field(self, radecfreq=None, flat_sky=None):
        """
        Grid the galaxy catalogue to a rectangular grid field.

        If flat_sky is True, no gridding is performed. Instead, the map cube
        dimensions are taken to be a rectangular grid, with the grid length
        corresponding to the pixel resolution on x/y and los frequency resolution
        as z.

        """
        if self.box_origin is None:
            self.get_enclosing_box()
        if flat_sky is None:
            flat_sky = self.flat_sky
        if radecfreq is None:
            ra_gal = self.ra_gal
            dec_gal = self.dec_gal
            freq_gal = self.freq_gal
        else:
            ra_gal, dec_gal, freq_gal = radecfreq
        real_dtype = self.real_dtype
        if ra_gal.size == 0:
            gal_pos_in_box = np.zeros((0, 3), dtype=real_dtype)
        if flat_sky:
            self.compensate = False
            z_gal = freq_to_redshift(freq_gal)
            self.use_flat_sky_box(flat_sky_padding=[0, 0, 0])
            pos_indx_1, pos_indx_2 = radec_to_indx(
                ra_gal, dec_gal, self.wproj, to_int=False
            )
            if ra_gal.size > 0:
                gal_pos_in_box = np.zeros((ra_gal.size, 3), dtype=real_dtype)
                gal_pos_in_box[:, 0] = pos_indx_1 / self.num_pix_x * self.box_len[0]
                gal_pos_in_box[:, 1] = pos_indx_2 / self.num_pix_y * self.box_len[1]
                gal_pos_in_box[:, 2] = (
                    self.astropy_cosmo_fiducial.comoving_distance(z_gal).value
                    - self.astropy_cosmo_fiducial.comoving_distance(
                        self.z_ch.min()
                    ).value
                )
        else:
            if ra_gal.size > 0:
                (_, _, _, _, _, _, _, gal_pos_arr) = minimum_enclosing_box_of_lightcone(
                    ra_gal,
                    dec_gal,
                    freq_gal,
                    cosmo=self.astropy_cosmo_fiducial,
                    return_coord=True,
                    tile=False,
                    rot_mat=self.rot_mat_sky_to_box,
                )
                gal_pos_in_box = (gal_pos_arr - self.box_origin[None, :]).astype(
                    real_dtype, copy=False
                )
        all_sel = np.arange(gal_pos_in_box.shape[0])
        gal_sel_batches = np.array_split(all_sel, self.batch_number)
        gal_sel_batches = [sel for sel in gal_sel_batches if sel.size > 0]
        if len(gal_sel_batches) == 0:
            shape = tuple(self.box_ndim.tolist())
            gal_map_rg = np.zeros(shape, dtype=real_dtype)
            gal_weights_rg = np.zeros(shape, dtype=real_dtype)
            pixel_counts_gal_rg = np.zeros(shape, dtype=real_dtype)
        else:
            gal_map_rg = None
            gal_weights_rg = None
            pixel_counts_gal_rg = None
            for sel in gal_sel_batches:
                gal_map_i, gal_weights_i, pixel_counts_i = self._grid_gal_to_field(
                    gal_pos_in_box, sel
                )
                if gal_map_rg is None:
                    gal_map_rg = np.zeros_like(gal_map_i)
                    gal_weights_rg = np.zeros_like(gal_weights_i)
                    pixel_counts_gal_rg = np.zeros_like(pixel_counts_i)
                gal_map_rg += gal_map_i
                gal_weights_rg += gal_weights_i
                pixel_counts_gal_rg += pixel_counts_i
        gal_map_rg = np.asarray(gal_map_rg, dtype=real_dtype)
        gal_weights_rg = np.asarray(gal_weights_rg, dtype=real_dtype)
        pixel_counts_gal_rg = np.asarray(pixel_counts_gal_rg, dtype=real_dtype)
        self.field_2 = gal_map_rg
        # only pixels sampled by the lightcone is used
        weights_g = (self.counts_in_box > 0).astype(real_dtype)
        self.weights_field_2 = weights_g
        self.weights_grid_2 = np.ones_like(self.field_2, dtype=real_dtype)
        self.mean_center_2 = True
        self.unitless_2 = True
        include_beam = np.array(self.include_beam)
        include_beam[1] = False
        self.include_beam = include_beam
        include_sky_sampling = np.array(self.include_sky_sampling)
        include_sky_sampling[1] = False
        self.include_sky_sampling = include_sky_sampling

        return gal_map_rg, gal_weights_rg, pixel_counts_gal_rg

    def _grid_gal_to_field(self, gal_pos_in_box, partial_sel):
        gal_pos_i = gal_pos_in_box[partial_sel]
        (
            gal_map_rg,
            gal_weights_rg,
            pixel_counts_gal_rg,
        ) = project_particle_to_regular_grid(
            gal_pos_i,
            self.box_len,
            self.box_ndim,
            grid_scheme=self.grid_scheme,
            compensate=False,  # compensate should be at model level
            average=False,
        )
        gal_map_rg2, _, _ = project_particle_to_regular_grid(
            gal_pos_i,
            self.box_len,
            self.box_ndim,
            grid_scheme=self.grid_scheme,
            compensate=False,  # compensate should be at model level
            average=False,
            shift=self.interlace_shift,
        )
        gal_map_rg = interlace_two_fields(gal_map_rg, gal_map_rg2, self.interlace_shift)
        real_dtype = self.real_dtype
        gal_map_rg = np.asarray(gal_map_rg, dtype=real_dtype)
        gal_weights_rg = np.asarray(gal_weights_rg, dtype=real_dtype)
        pixel_counts_gal_rg = np.asarray(pixel_counts_gal_rg, dtype=real_dtype)
        return gal_map_rg, gal_weights_rg, pixel_counts_gal_rg

    def ra_dec_z_for_coord_in_box(self, pos_in_box):
        """
        Convert the coordinates in the box to ra, dec, z,
        and also return the comoving distance to the observer for each point.

        Parameters
        ----------
        pos_in_box: array.
            The coordinates in the box.

        Returns
        -------
        pos_ra: array.
            The ra of the points.
        pos_dec: array.
            The dec of the points.
        pos_z: array.
            The redshift of the points.
        pos_comov_dist: array.
            The comoving distance to the observer for each point.
        """
        pos_arr = pos_in_box + self.box_origin
        rot_back = np.linalg.inv(self.rot_mat_sky_to_box)
        pos_arr = np.einsum("ij,aj->ai", rot_back, pos_arr)
        pos_comov_dist = np.sqrt(np.sum(pos_arr**2, axis=-1))
        pos_z = self.z_as_func_of_comov_dist(pos_comov_dist)
        pos_ra, pos_dec = hp.vec2ang(pos_arr / pos_comov_dist[:, None], lonlat=True)
        return pos_ra, pos_dec, pos_z, pos_comov_dist

    def _grid_field_to_sky_map_wcs(
        self,
        field,
        average=True,
        mask=True,
        wproj=None,
        num_pix_x=None,
        num_pix_y=None,
        los_sel=None,
    ):
        """
        Grid a box field onto the WCS map cube (raster ``(num_pix_x, num_pix_y, n_ch)``).
        """
        if wproj is None:
            wproj = self.wproj
        if num_pix_x is None:
            num_pix_x = self.num_pix_x
        if num_pix_y is None:
            num_pix_y = self.num_pix_y
        los_sel = (
            np.arange(self.box_ndim[2], dtype=int)
            if los_sel is None
            else np.asarray(los_sel, dtype=int)
        )
        expected_shape = (self.box_ndim[0], self.box_ndim[1], los_sel.size)
        if field.shape != expected_shape:
            raise ValueError(
                f"field shape {field.shape} does not match expected shape "
                f"{expected_shape} for los_sel size {los_sel.size}"
            )
        x_vec = self.x_vec[0]
        y_vec = self.x_vec[1]
        z_vec = self.x_vec[2][los_sel]
        nx = x_vec.size
        ny = y_vec.size
        nz = z_vec.size
        nxyz = nx * ny * nz
        pos_xyz = np.empty((nxyz, 3), dtype=self.real_dtype)
        pos_xyz[:, 0] = np.repeat(x_vec, ny * nz)
        pos_xyz[:, 1] = np.tile(np.repeat(y_vec, nz), nx)
        pos_xyz[:, 2] = np.tile(z_vec, nx * ny)
        pos_ra, pos_dec, pos_z, _ = self.ra_dec_z_for_coord_in_box(pos_xyz)
        pos_indx_1, pos_indx_2 = radec_to_indx(pos_ra, pos_dec, wproj, to_int=False)
        pos_indx_z = redshift_to_freq(pos_z) - self.nu.min()
        pos_indx_array = np.empty((nxyz, 3), dtype=self.real_dtype)
        pos_indx_array[:, 0] = pos_indx_1
        pos_indx_array[:, 1] = pos_indx_2
        pos_indx_array[:, 2] = pos_indx_z
        map_bin, _, count_bin = project_particle_to_regular_grid(
            pos_indx_array,
            np.array([num_pix_x, num_pix_y, self.nu.max() - self.nu.min()]),
            np.array([num_pix_x, num_pix_y, self.nu.size]),
            particle_mass=field.ravel(),
            average=average,
            compensate=False,
            grid_scheme="nnb",
        )
        if mask:
            map_bin *= self.W_HI
        return map_bin, count_bin

    def _grid_field_to_sky_map_healpix(
        self,
        field,
        average=True,
        mask=True,
        los_sel=None,
    ):
        """
        Grid a box field onto the sparse HEALPix map ``(len(pixel_id), n_ch)``.

        Voxel centres are assigned to HEALPix pixels at ``self.hp_nside`` and to
        frequency channels via :func:`~meer21cm.util.find_ch_id`. Only voxels whose
        pixel lies in ``self.pixel_id`` contribute.
        """
        los_sel = (
            np.arange(self.box_ndim[2], dtype=int)
            if los_sel is None
            else np.asarray(los_sel, dtype=int)
        )
        expected_shape = (self.box_ndim[0], self.box_ndim[1], los_sel.size)
        if field.shape != expected_shape:
            raise ValueError(
                f"field shape {field.shape} does not match expected shape "
                f"{expected_shape} for los_sel size {los_sel.size}"
            )
        nside = int(self.hp_nside)
        pixel_id = np.asarray(self.pixel_id, dtype=np.int64)
        n_out = pixel_id.size
        n_ch = int(self.nu.size)
        order = np.argsort(pixel_id, kind="mergesort")
        pix_sorted = pixel_id[order]

        x_vec = self.x_vec[0]
        y_vec = self.x_vec[1]
        z_vec = self.x_vec[2][los_sel]
        nx = x_vec.size
        ny = y_vec.size
        nz = z_vec.size
        nxyz = nx * ny * nz
        pos_xyz = np.empty((nxyz, 3), dtype=self.real_dtype)
        pos_xyz[:, 0] = np.repeat(x_vec, ny * nz)
        pos_xyz[:, 1] = np.tile(np.repeat(y_vec, nz), nx)
        pos_xyz[:, 2] = np.tile(z_vec, nx * ny)
        pos_ra, pos_dec, pos_z, _ = self.ra_dec_z_for_coord_in_box(pos_xyz)
        hpix = hp.ang2pix(nside, pos_ra, pos_dec, lonlat=True).astype(np.int64)
        pos_nu = np.asarray(redshift_to_freq(pos_z), dtype=np.float64)
        ch_idx = find_ch_id(pos_nu, self.nu)
        valid_ch = (ch_idx >= 0) & (ch_idx < n_ch)
        hpix = hpix[valid_ch]
        ch_idx = ch_idx[valid_ch]
        mass = np.asarray(field, dtype=self.real_dtype).ravel()[valid_ch]

        row_s = np.searchsorted(pix_sorted, hpix)
        # Do not index pix_sorted[row_s] when row_s == n_out (past end of searchsorted).
        in_bounds = row_s < n_out
        in_survey = np.zeros(hpix.shape, dtype=bool)
        in_survey[in_bounds] = pix_sorted[row_s[in_bounds]] == hpix[in_bounds]
        row = order[row_s[in_survey]]
        ch_idx = ch_idx[in_survey]
        mass = mass[in_survey]

        map_sum = np.zeros((n_out, n_ch), dtype=self.real_dtype)
        cnt = np.zeros((n_out, n_ch), dtype=self.real_dtype)
        np.add.at(map_sum, (row, ch_idx), mass)
        np.add.at(cnt, (row, ch_idx), 1.0)
        if average:
            with np.errstate(divide="ignore", invalid="ignore"):
                map_bin = np.where(cnt > 0, map_sum / cnt, 0.0)
        else:
            map_bin = map_sum
        count_bin = cnt
        if mask:
            map_bin *= self.W_HI
        return map_bin, count_bin

    def grid_field_to_sky_map(
        self,
        field,
        average=True,
        mask=True,
        wproj=None,
        num_pix_x=None,
        num_pix_y=None,
        los_sel=None,
    ):
        """
        Grid a field in the rectangular box onto the sky.

        Parameters
        ----------
        field: array.
            The field in the box to be gridded.

        average: bool, default True.
            Whether the field grids are averaged or summed into sky pixels.

        mask: bool, default True.
            If True, the sky map is then masked by the survey selection function.

        wproj: :class:`astropy.wcs.WCS` object, default None.
            **WCS only.** The WCS for the output sky map. Default uses ``self.wproj``.

        num_pix_x: int, default None.
            **WCS only.** Number of pixels along the first map axis. Default ``self.num_pix_x``.

        num_pix_y: int, default None.
            **WCS only.** Number of pixels along the second map axis. Default ``self.num_pix_y``.

        los_sel: array-like, default None.
            Optional selector of line-of-sight (last-axis) indices represented by ``field``.
            If None, ``field`` is assumed to contain all LOS slices with shape
            ``self.box_ndim``. If provided, ``field`` must have shape
            ``(self.box_ndim[0], self.box_ndim[1], len(los_sel))`` and the projected
            output still uses the full LOS map axis (``self.nu.size``), so multiple
            chunked calls can be merged by accumulating mass/count outputs.

        Returns
        -------
        map_bin: array.
            The output sky map (WCS ``(nx, ny, n_ch)`` or HEALPix ``(n_pix, n_ch)``).
        count_bin: array.
            Per-cell accumulation used for averaging (WCS) or voxel counts (HEALPix).

        """
        fmt = self.skymap.format
        if fmt == "wcs":
            return self._grid_field_to_sky_map_wcs(
                field,
                average=average,
                mask=mask,
                wproj=wproj,
                num_pix_x=num_pix_x,
                num_pix_y=num_pix_y,
                los_sel=los_sel,
            )
        if fmt == "healpix":
            return self._grid_field_to_sky_map_healpix(
                field,
                average=average,
                mask=mask,
                los_sel=los_sel,
            )

    def gen_random_poisson_galaxy(
        self, sel=None, num_g_rand=None, seed=None, dndz=None
    ):
        """
        Generate a random galaxy catalogue from the map cube following the Poisson distribution.
        The generation of the sample does not use the instance seed if not explicitly passed and will use a random one otherwise.
        If you want to generate multiple random catalogues, you need to set a different seed manually for each catalogue.

        Angular positions are drawn uniformly in solid angle within the
        selected footprint (WCS or HEALPix). RA and Dec are always taken
        from the same map pixel before intra-pixel sampling.

        Parameters
        ----------
        sel: array, default None
            Boolean (or truthy) sky-plane selection matching ``ra_map`` /
            ``dec_map``. If None, use ``self.W_HI[..., 0]`` (first frequency
            channel), which works for both WCS ``(nx, ny, nch)`` and HEALPix
            ``(n_pix, nch)`` maps.
        num_g_rand: int, default None
            The number of galaxies to generate. Default uses the number of galaxies stored in the data in `self.ra_gal`.
        seed: int, default None
            The seed for the random number generator.
        dndz: callable or array, default None
            Optional radial selection in the same per-volume convention as
            :attr:`~meer21cm.mock.MockSimulation.discrete_source_dndz`
            (shape only; overall count is ``num_g_rand``). If a callable, it
            is evaluated on candidate redshifts ``dndz(z)`` and used as
            comoving number-density weights (renormalised), i.e. sampling
            uses ``p(chi) ∝ dndz(z(chi)) * chi**2``. If an array, it must
            match the number of frequency channels and is interpolated in
            redshift. If None, use constant comoving number density
            (``p(chi) ∝ chi**2``), matching the mock default
            ``discrete_source_dndz=np.ones_like``.

        Returns
        -------
        ra_rand: np.ndarray.
            The ra of the random galaxies.
        dec_rand: np.ndarray.
            The dec of the random galaxies.
        freq_rand: np.ndarray.
            The ``frequency`` of the random galaxies. The redshift of the random galaxies can
            be calculated by ``meer21cm.util.redshift_to_freq(z_rand)``.
        """
        if sel is None:
            sel = self.W_HI[..., 0]
        sel = np.asarray(sel)
        if sel.shape != self.ra_map.shape:
            raise ValueError(
                f"sel shape {sel.shape} must match ra_map shape {self.ra_map.shape}"
            )
        if num_g_rand is None:
            num_g_rand = self.ra_gal.size
        num_g_rand = int(num_g_rand)
        if num_g_rand < 0:
            raise ValueError("num_g_rand must be non-negative")
        rng = np.random.default_rng(seed=seed)

        mask = np.asarray(sel, dtype=bool)
        if not np.any(mask):
            raise ValueError(
                "selection is empty; no pixels to draw random galaxies from"
            )
        idx_all = np.flatnonzero(mask.ravel())
        ra_flat = np.asarray(self.ra_map, dtype=float).ravel()
        dec_flat = np.asarray(self.dec_map, dtype=float).ravel()

        fmt = self.skymap.format
        if fmt == "healpix":
            # Equal-area pixels: uniform among selected, then uniform in pixel.
            pick = rng.choice(idx_all.size, size=num_g_rand, replace=True)
            ipix = np.asarray(self.pixel_id, dtype=np.int64)[idx_all[pick]]
            ra_rand, dec_rand = _random_radec_in_healpix_pixels(
                self.hp_nside, ipix, rng
            )
        else:
            # WCS: weight by pixel solid angle for equal Δα×Δδ cells, then sample
            # uniformly in solid angle inside each RA–Dec rectangle.
            half = float(self.pix_resol) / 2.0
            dec0 = dec_flat[idx_all]
            solid = np.abs(
                np.sin(np.radians(dec0 + half)) - np.sin(np.radians(dec0 - half))
            )
            solid = solid / solid.sum()
            pick = rng.choice(idx_all.size, size=num_g_rand, replace=True, p=solid)
            idx = idx_all[pick]
            ra_rand, dec_rand = _random_radec_in_radec_rectangles(
                ra_flat[idx], dec_flat[idx], half, half, rng
            )

        chi_min = (
            self.astropy_cosmo_fiducial.comoving_distance(self.z_ch.min())
            .to("Mpc")
            .value
        )
        chi_max = (
            self.astropy_cosmo_fiducial.comoving_distance(self.z_ch.max())
            .to("Mpc")
            .value
        )
        if dndz is None:
            # Constant comoving number density: p(χ) ∝ χ² ⇒ sample via χ³ CDF.
            u = rng.uniform(0.0, 1.0, size=num_g_rand)
            cov_dist_rand = np.cbrt(chi_min**3 + u * (chi_max**3 - chi_min**3))
            z_rand = self.z_as_func_of_comov_dist(cov_dist_rand)
        else:
            # Per-volume weights on an equal-χ grid: p(χ) ∝ dndz(z(χ)) χ².
            n_grid = max(512, int(num_g_rand))
            cov_grid = np.linspace(chi_min, chi_max, n_grid)
            z_grid = self.z_as_func_of_comov_dist(cov_grid)
            if callable(dndz):
                w = np.asarray(dndz(z_grid), dtype=float)
            else:
                dndz_arr = np.asarray(dndz, dtype=float)
                if dndz_arr.shape == self.z_ch.shape:
                    from scipy.interpolate import interp1d

                    w = interp1d(
                        self.z_ch,
                        dndz_arr,
                        kind="linear",
                        bounds_error=False,
                        fill_value=0.0,
                    )(z_grid)
                else:
                    raise ValueError(
                        "dndz array must match frequency-channel redshift shape"
                    )
            w = np.clip(w, 0.0, None) * cov_grid**2
            if not np.any(w > 0):
                raise ValueError("dndz weights are zero everywhere in the survey range")
            w = w / w.sum()
            cov_dist_rand = rng.choice(cov_grid, size=num_g_rand, replace=True, p=w)
            # small continuous jitter within bin
            dc = (cov_grid[1] - cov_grid[0]) if n_grid > 1 else 0.0
            cov_dist_rand = cov_dist_rand + rng.uniform(
                -0.5 * dc, 0.5 * dc, size=num_g_rand
            )
            cov_dist_rand = np.clip(cov_dist_rand, chi_min, chi_max)
            z_rand = self.z_as_func_of_comov_dist(cov_dist_rand)
        return ra_rand, dec_rand, redshift_to_freq(z_rand)

    def apply_taper_to_field(
        self,
        field,
        taper_func=None,
        axis=[
            2,
        ],
    ):
        """
        Apply a taper to the field, by multiplying the taper function to the
        corresponding weights of the field.

        Parameters
        ----------
        field: int.
            The index of the field to be tapered, either 1 or 2.
        taper_func: function, default None.
            The taper function. Default uses the stored ``self.taper_func``.
        axis: list, default [2,].
            The axis to apply the taper to. Default is the z-axis which is approximately the los.
        """
        if taper_func is None:
            taper_func = self.taper_func
        taper_i = [taper_func(self.box_ndim[i]) for i in range(3)]
        taper = 1
        for i in axis:
            slice_list_i = [None, None, None]
            slice_list_i[i] = slice(None, None, None)
            slice_list_i = tuple(slice_list_i)
            taper = taper * taper_i[i][slice_list_i]
        setattr(self, f"weights_{field}", getattr(self, f"weights_{field}") * taper)

    @property
    @tagging("cosmo_fiducial", "nu", "mock", "box")
    def box_voxel_redshift(self):
        """
        The redshift of each voxel in the rectangular box.
        """
        if self._box_voxel_redshift is None:
            return np.ones(self.box_ndim) * self.z
        return self._box_voxel_redshift
