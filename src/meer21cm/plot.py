import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


def plot_pixels_along_los(
    map_in,
    map_has_sampling,
    zaxis=None,
    xlabel="",
    ylabel="",
    lw=0.01,
    title="",
    los_axis=-1,
):
    plt.figure()
    map_plot = map_in.copy()
    map_plot[map_has_sampling == 0] = np.nan
    if los_axis < 0:
        los_axis += 3
    axes = [0, 1, 2]
    axes.remove(los_axis)
    # make sure los is the last axis
    axes = axes + [
        los_axis,
    ]
    map_plot = np.transpose(map_plot, axes=axes)
    nz = map_plot.shape[-1]
    map_plot = map_plot.reshape((-1, nz))
    if zaxis is None:
        zaxis = np.arange(nz)
    for i in range(len(map_plot)):
        plt.plot(zaxis, map_plot[i], lw=lw, color="black")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)


def plot_map(
    map_plot,
    wproj,
    W=None,
    title="",
    have_cbar=True,
    cbar_label="",
    cbarshrink=1,
    ZeroCentre=False,
    vmin=None,
    vmax=None,
    cmap="magma",
    invert_x=True,
    dpi=100,
    cbar_aspect=25,
):
    """
    Stolen from meerpower
    """
    plt.figure(dpi=dpi)
    plt.subplot(projection=wproj)
    ax = plt.gca()
    lon = ax.coords[0]
    lat = ax.coords[1]
    lon.set_major_formatter("d")
    lon.set_ticks_position("b")
    lat.set_ticks_position("l")
    plt.grid(True, color="grey", ls="solid", lw=0.5)
    map_in = map_plot.copy()
    if len(np.shape(map_in)) == 3:
        map_in = np.mean(
            map_in, 2
        )  # Average along 3rd dimention (LoS) as default if 3D map given
        if W is not None:
            W = W.copy()
            W = np.mean(W, 2)
    if vmax is not None:
        map_in[map_in > vmax] = vmax
    if vmin is not None:
        map_in[map_in < vmin] = vmin
    if ZeroCentre == True:
        divnorm = colors.TwoSlopeNorm(
            vmin=np.min(map_in), vcenter=0, vmax=np.max(map_in)
        )
        cmap = copy.copy(matplotlib.cm.get_cmap("seismic"))
        cmap.set_bad(color="grey")
    else:
        divnorm = None
    if W is not None:
        map_in[W == 0] = np.nan
    plt.imshow(map_in.T, cmap=cmap, norm=divnorm)
    if vmax is not None or vmin is not None:
        plt.clim(vmin, vmax)
    if have_cbar:
        cbar = plt.colorbar(
            orientation="horizontal",
            shrink=cbarshrink,
            pad=0.2,
            aspect=cbar_aspect,
        )
        cbar.set_label(cbar_label)
    if invert_x:
        ax.invert_xaxis()
    plt.xlabel("R.A [deg]")
    plt.ylabel("Dec. [deg]")
    plt.title(title)


def plot_eigenspectrum(
    eigenval,
    eignumb=None,
    eignumb_cut=40,
    title="",
):
    plt.figure()
    if len(eigenval.shape) == 1:
        eigenval = np.array(
            [
                eigenval,
            ]
        )
    if eignumb is None:
        eignumb = np.arange(eigenval.shape[1])
    for i in range(len(eigenval)):
        plt.plot(eignumb[:eignumb_cut], eigenval[i][:eignumb_cut], "-o")
    plt.yscale("log")
    plt.xlabel("Eigennumber")
    plt.ylabel("Eigenvalue")
    plt.title(title)


def plot_projected_map(A, data, wproj, W=None):
    S_pca_full = np.nan_to_num(A[:, None, :] * A[None, :, :])
    for i in range(A.shape[1]):
        res_i = np.einsum("ij,abj->abi", S_pca_full[:, :, i], np.nan_to_num(data))
        plot_map(res_i, wproj, W=W)


def visualise_patch_split(mask_arr, wproj):
    """
    Visualise the patch split by plotting the mask array.
    """
    for nu_indx in range(mask_arr.shape[2]):
        mask_map = np.zeros_like(mask_arr[0, 0, 0, :, :, 0]) + np.nan
        for i in range(mask_arr.shape[0]):
            for j in range(mask_arr.shape[1]):
                sel = mask_arr[i, j, nu_indx].sum(-1) > 0
                mask_map[sel] = i + j * mask_arr.shape[0]
        plot_map(mask_map, wproj, have_cbar=False, title=f"frequency bin {nu_indx}")
