import matplotlib.pyplot as plt
from meer21cm.grid import *
from meer21cm.grid import _random_radec_in_healpix_pixels
import pytest
from meer21cm import PowerSpectrum


def W_mas(dims, window="nnb", FullPk=False):
    """
    Hockney Eastwood mass assignment corrections

    taken from meerpower
    """
    if window == "nnb" or "ngp":
        p = 1
    if window == "cic":
        p = 2
    if window == "tsc":
        p = 3
    if window == "pcs":
        p = 4
    lx, ly, lz, nx, ny, nz = dims[:6]
    nyqx, nyqy, nyqz = nx * np.pi / lx, ny * np.pi / ly, nz * np.pi / lz
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=lx / nx)[:, np.newaxis, np.newaxis]
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=ly / ny)[np.newaxis, :, np.newaxis]
    if FullPk == False:
        kz = (
            2
            * np.pi
            * np.fft.fftfreq(nz, d=lz / nz)[: int(nz / 2) + 1][
                np.newaxis, np.newaxis, :
            ]
        )
    if FullPk == True:
        kz = 2 * np.pi * np.fft.fftfreq(nz, d=lz / nz)[np.newaxis, np.newaxis, :]
    qx, qy, qz = (
        (np.pi * kx) / (2 * nyqx),
        (np.pi * ky) / (2 * nyqy),
        (np.pi * kz) / (2 * nyqz),
    )
    wx = np.divide(np.sin(qx), qx, out=np.ones_like(qx), where=qx != 0.0)
    wy = np.divide(np.sin(qy), qy, out=np.ones_like(qy), where=qy != 0.0)
    wz = np.divide(np.sin(qz), qz, out=np.ones_like(qz), where=qz != 0.0)
    return (wx * wy * wz) ** (p)


@pytest.mark.parametrize("window", list(allowed_window_scheme))
def test_fourier_window_for_assignment(window):
    test_1 = fourier_window_for_assignment([10, 10, 10], window=window)
    test_2 = W_mas([1, 1, 1, 10, 10, 10], window=window, FullPk=False)
    assert np.allclose(test_1, test_2)


def test_find_rotation_matrix():
    vec = np.random.randn(3)
    vec /= np.sqrt(np.sum(vec**2))
    rot_mat = find_rotation_matrix(vec)
    assert np.allclose(rot_mat @ vec, [0, 0, 1])


def test_minimum_enclosing_box_of_lightcone():
    ra_rand = np.random.uniform(0, 359.9, 1)[0]
    dec_rand = np.random.uniform(-90, 89.9, 1)[0]
    vec_rand = hp.ang2vec(ra_rand, dec_rand, lonlat=True)
    rot_mat_rand = find_rotation_matrix(vec_rand)
    ang_div = 0.005 * np.pi / 180
    z_2 = np.cos(ang_div)
    vec_arr = np.zeros((4, 3))
    sign_i = [-1, 0, 1, 0]
    sign_j = [0, -1, 0, 1]
    for indx in range(4):
        vec_arr[indx] = np.linalg.inv(rot_mat_rand) @ np.array(
            [
                sign_i[indx] * np.sqrt(1 - z_2**2),
                sign_i[indx] * np.sqrt(1 - z_2**2),
                z_2,
            ]
        )
    ra_test, dec_test = hp.vec2ang(vec_arr, lonlat=True)
    freq_test = np.array((f_21 / (1 + 0.5001), f_21 / (1 + 0.5)))
    (
        xmin,
        ymin,
        zmin,
        xlen,
        ylen,
        zlen,
        rot_mat,
        pos_arr,
    ) = minimum_enclosing_box_of_lightcone(
        ra_test, dec_test, freq_test, return_coord=True
    )
    cov_dist = Planck18.comoving_distance(0.5).value
    cov_scale = (
        Planck18.comoving_distance(0.5001) - Planck18.comoving_distance(0.5)
    ).value
    assert np.abs((xlen - cov_dist * 0.01 * np.pi / 180) / xlen) < 1e-3
    assert np.abs((ylen - cov_dist * 0.01 * np.pi / 180) / ylen) < 1e-3
    assert np.abs((cov_scale - zlen) / cov_scale) < 1e-3
    test_vec = rot_mat @ np.array(
        [(xmin + xlen / 2), (ymin + ylen / 2), (zmin + zlen / 2)]
    )
    test_vec /= np.sqrt(np.sum(test_vec**2))
    ra_cen, dec_cen = hp.vec2ang(test_vec, lonlat=True)
    assert np.abs(ra_rand - ra_cen) < 0.05
    assert np.abs(dec_rand - dec_cen) < 0.05
    ra_test = np.ones((4, 2)) * ra_test[:, None]
    dec_test = np.ones((4, 2)) * dec_test[:, None]
    freq_test = np.ones((4, 2)) * freq_test[None, :]

    (
        xmin,
        ymin,
        zmin,
        xlen,
        ylen,
        zlen,
        rot_mat,
        pos_arr,
    ) = minimum_enclosing_box_of_lightcone(
        ra_test, dec_test, freq_test, tile=False, return_coord=True
    )
    assert np.abs((xlen - cov_dist * 0.01 * np.pi / 180) / xlen) < 1e-3
    assert np.abs((ylen - cov_dist * 0.01 * np.pi / 180) / ylen) < 1e-3
    assert np.abs((cov_scale - zlen) / cov_scale) < 1e-3
    test_vec = rot_mat @ np.array(
        [(xmin + xlen / 2), (ymin + ylen / 2), (zmin + zlen / 2)]
    )
    test_vec /= np.sqrt(np.sum(test_vec**2))
    ra_cen, dec_cen = hp.vec2ang(test_vec, lonlat=True)
    assert np.abs(ra_rand - ra_cen) < 0.05
    assert np.abs(dec_rand - dec_cen) < 0.05


def test_project_function():
    for scheme in allowed_window_scheme:
        s_arr = np.linspace(-1.5, 1.5, 601)
        weight_arr = project_function(s_arr, scheme)
        assert np.abs(np.trapezoid(weight_arr, x=s_arr) - 1) < 1e-2
        s_test = 0.25
        func_value = project_function(s_test, scheme)
        if scheme == "nnb":
            assert func_value == 1
        elif scheme == "cic":
            assert func_value == 0.75
        elif scheme == "tsc":
            assert func_value == (0.75 - s_test**2)
        elif scheme == "pcs":
            assert func_value == (4 - 6 * s_test**2 + 3 * s_test**3) / 6
        s_test = 0.75
        func_value = project_function(s_test, scheme)
        if scheme == "nnb":
            assert func_value == 0
        elif scheme == "cic":
            assert func_value == 0.25
        elif scheme == "tsc":
            assert func_value == 0.5 * (1.5 - s_test) ** 2
        elif scheme == "pcs":
            assert func_value == (4 - 6 * s_test**2 + 3 * s_test**3) / 6
        s_test = 1.5
        func_value = project_function(s_test, scheme)
        if scheme == "pcs":
            assert func_value == (2 - s_test) ** 3 / 6
    scheme = "test"
    with pytest.raises(ValueError):
        weight_arr = project_function(s_arr, scheme)


def test_particle_to_mesh_distance():
    box_len = np.array([20, 30, 40])
    box_ndim = np.array([2, 3, 4])
    xx, yy, zz = np.meshgrid(
        np.arange(box_ndim[0]),
        np.arange(box_ndim[1]),
        np.arange(box_ndim[2]),
        indexing="ij",
    )
    _xx = xx.ravel() * 10 + 5
    _yy = yy.ravel() * 10 + 5
    _zz = zz.ravel() * 10 + 5
    par_pos = np.zeros((np.prod(box_ndim), 3))
    par_pos[:, 0] = _xx
    par_pos[:, 1] = _yy
    par_pos[:, 2] = _zz
    dist, indx_grid = particle_to_mesh_distance(par_pos, box_len, box_ndim)
    assert np.allclose(dist, np.zeros_like(dist))
    assert np.allclose(indx_grid, [xx.ravel(), yy.ravel(), zz.ravel()])
    dist, indx_grid = particle_to_mesh_distance(
        np.array([[-10, -10, -10]]), box_len, box_ndim
    )
    assert np.allclose(dist.ravel(), [-1.5, -1.5, -1.5])
    assert np.allclose(np.array(indx_grid).ravel(), [0, 0, 0])
    dist, indx_grid = particle_to_mesh_distance(
        np.array([[25, 35, 45]]), box_len, box_ndim
    )
    assert np.allclose(dist.ravel(), [1, 1, 1])
    assert np.allclose(np.array(indx_grid).ravel(), [1, 2, 3])


@pytest.mark.parametrize("window", list(allowed_window_scheme))
def test_project_particle_to_regular_grid(window):
    # test par on grid centres
    box_len = np.array([20, 30, 40])
    box_ndim = np.array([2, 3, 4])
    xx, yy, zz = np.meshgrid(
        np.arange(box_ndim[0]),
        np.arange(box_ndim[1]),
        np.arange(box_ndim[2]),
        indexing="ij",
    )
    _xx = xx.ravel() * 10
    _yy = yy.ravel() * 10
    _zz = zz.ravel() * 10
    par_pos = np.zeros((np.prod(box_ndim), 3))
    par_pos[:, 0] = _xx
    par_pos[:, 1] = _yy
    par_pos[:, 2] = _zz
    par_pos += 5
    test_map, test_w, test_c = project_particle_to_regular_grid(
        par_pos,
        box_len,
        box_ndim,
        grid_scheme=window,
    )
    assert np.allclose(test_map, np.ones_like(test_map))
    test_map, test_w, test_c = project_particle_to_regular_grid(
        par_pos,
        box_len,
        box_ndim,
        grid_scheme=window,
        shift=10.0,
        compensate=True,
    )
    assert np.allclose(test_map, np.zeros_like(test_map))
    if window == "nnb":
        # shift 5, should still be the same
        test_map, test_w, test_c = project_particle_to_regular_grid(
            par_pos,
            box_len,
            box_ndim,
            grid_scheme=window,
            shift=-0.5,
        )
        assert np.allclose(test_c, np.ones_like(test_c))
        # slightly more, should be less
        test_map, test_w, test_c = project_particle_to_regular_grid(
            par_pos,
            box_len,
            box_ndim,
            grid_scheme=window,
            shift=-0.5001,
        )
        assert not np.allclose(test_c, np.ones_like(test_c))
    if window == "nnb" or window == "pcs":
        x = 0
    else:
        # test interpolation, put particles on the grid edges
        xx, yy, zz = np.meshgrid(
            np.arange(box_ndim[0] + 1),
            np.arange(box_ndim[1] + 1),
            np.arange(box_ndim[2] + 1),
            indexing="ij",
        )
        _xx = xx.ravel() * 10
        _yy = yy.ravel() * 10
        _zz = zz.ravel() * 10
        par_pos = np.zeros((np.prod(box_ndim + 1), 3))
        par_pos[:, 0] = _xx
        par_pos[:, 1] = _yy
        par_pos[:, 2] = _zz
        par_mass = (xx + yy + zz).ravel()
        pars_mass = xx + yy + zz
        pars_mass = (pars_mass[1:, 1:, 1:] + pars_mass[:-1, :-1, :-1]) / 2
        test_map, test_w, test_c = project_particle_to_regular_grid(
            par_pos, box_len, box_ndim, grid_scheme=window, particle_mass=par_mass
        )
        # output is the mean at between the grid edges
        assert np.allclose(test_map, pars_mass)
        # effective count is still 1
        assert np.allclose(test_w, np.ones_like(test_w))
        assert np.allclose(test_c, np.ones_like(test_c))


def test_rotation_matrix_to_radec0():
    ra = np.random.uniform(0, 360.0, 1)[0]
    dec = np.random.uniform(-90.0, 90.0, 1)[0]
    rot_mat = rotation_matrix_to_radec0(ra, dec)
    vec = hp.ang2vec(ra, dec, lonlat=True)
    assert np.allclose(rot_mat @ vec, [1, 0, 0])


def test_sky_partition_for_radecrange():
    ra_range = np.array([130, 250])
    dec_range = np.array([-5, 7.5])
    pix_for_i, rot_mat_for_patch_i = sky_partition_for_radecrange(ra_range, dec_range)
    map_out = np.zeros(hp.nside2npix(128))
    for i, pix_id in enumerate(pix_for_i):
        map_out[pix_id] += 1
    # partition does not overlap
    assert map_out.max() == 1
    i = np.random.randint(0, len(pix_for_i))
    pix_id = pix_for_i[i]
    rot_mat_i = np.linalg.inv(rot_mat_for_patch_i[i])
    nside = 128
    npix = hp.nside2npix(nside)
    ra_grid, dec_grid = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    selection_grid_0 = angle_in_range(
        ra_grid, ra_range[0], ra_range[1]
    ) * angle_in_range(dec_grid, dec_range[0], dec_range[1])
    ra_region_0 = ra_grid[selection_grid_0]
    dec_region_0 = dec_grid[selection_grid_0]
    vec_region_0 = hp.ang2vec(ra_region_0, dec_region_0, lonlat=True)
    vec_region_rot = np.dot(rot_mat_i, vec_region_0.T)
    pix_region_rot = hp.vec2pix(
        nside, vec_region_rot[0], vec_region_rot[1], vec_region_rot[2]
    )
    map_rot = np.zeros(hp.nside2npix(nside))
    np.add.at(map_rot, pix_region_rot, 1)
    assert np.isin(np.arange(npix)[map_rot > 0], pix_id).mean() > 0.999


@pytest.mark.parametrize("window", list(allowed_window_scheme)[:-1])
def test_shot_noise_correction_from_gridding(window):
    ps = PowerSpectrum()
    ps._box_len = np.array([1000, 1000, 1000])
    ps._box_ndim = np.array([20, 20, 20])
    ps.propagate_field_k_to_model()
    rand_pos = np.random.uniform(0, 1000, size=(10000, 3))
    gal_count, _, _ = project_particle_to_regular_grid(
        rand_pos,
        ps.box_len,
        ps.box_ndim,
        grid_scheme=window,
        average=False,
    )
    ps.field_1 = gal_count / gal_count.mean() - 1
    ps3d_rand = ps.auto_power_3d_1
    sn_rand = (
        shot_noise_correction_from_gridding(ps.box_ndim, window)
        * np.prod(ps.box_len)
        / rand_pos.shape[0]
    )
    ratio = ps3d_rand / sn_rand
    assert np.allclose(ratio.mean(), 1, atol=3.0e-1)


def test_random_radec_in_healpix_pixels_raise_error():
    with pytest.raises(ValueError, match="hp_nside must be a power of 2, got 1023"):
        _random_radec_in_healpix_pixels(1023, 0, 42)
