# Third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Local
from . import utils


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_error_map(errors, data_config, title=None, step_length=3):
    """
    Plot a heatmap of errors of different variables at different
    predictions horizons
    errors: (pred_steps, d_f)
    """
    errors_np = errors.T.cpu().numpy()  # (d_f, pred_steps)
    d_f, pred_steps = errors_np.shape

    # Normalize all errors to [0,1] for color map
    max_errors = errors_np.max(axis=1)  # d_f
    errors_norm = errors_np / np.expand_dims(max_errors, axis=1)

    fig, ax = plt.subplots(figsize=(15, 10))

    ax.imshow(
        errors_norm,
        cmap="OrRd",
        vmin=0,
        vmax=1.0,
        interpolation="none",
        aspect="auto",
        alpha=0.8,
    )

    # ax and labels
    for (j, i), error in np.ndenumerate(errors_np):
        # Numbers > 9999 will be too large to fit
        formatted_error = f"{error:.3f}" if error < 9999 else f"{error:.2E}"
        ax.text(i, j, formatted_error, ha="center", va="center", usetex=False)

    # Ticks and labels
    label_size = 15
    ax.set_xticks(np.arange(pred_steps))
    pred_hor_i = np.arange(pred_steps) + 1  # Prediction horiz. in index
    pred_hor_h = step_length * pred_hor_i  # Prediction horiz. in hours
    ax.set_xticklabels(pred_hor_h, size=label_size)
    ax.set_xlabel("Lead time (h)", size=label_size)

    ax.set_yticks(np.arange(d_f))
    y_ticklabels = [
        f"{name} ({unit})"
        for name, unit in zip(
            data_config.dataset.var_names, data_config.dataset.var_units
        )
    ]
    ax.set_yticklabels(y_ticklabels, rotation=30, size=label_size)

    if title:
        ax.set_title(title, size=15)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_prediction(
    pred, target, obs_mask, data_config, title=None, vrange=None
):
    """
    Plot example prediction and grond truth.
    Each has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = min(vals.min().cpu().item() for vals in (pred, target))
        vmax = max(vals.max().cpu().item() for vals in (pred, target))
    else:
        vmin, vmax = vrange

    # Set up masking of border region
    mask_reshaped = obs_mask.reshape(*data_config.grid_shape_state)
    pixel_alpha = (
        mask_reshaped.clamp(0.7, 1).cpu().numpy()
    )  # Faded border region

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 7),
        subplot_kw={"projection": data_config.coords_projection},
    )

    # Plot pred and target
    for ax, data in zip(axes, (target, pred)):
        ax.coastlines()  # Add coastline outlines
        data_grid = data.reshape(*data_config.grid_shape_state).cpu().numpy()
        im = ax.imshow(
            data_grid,
            origin="lower",
            alpha=pixel_alpha,
            vmin=vmin,
            vmax=vmax,
            cmap="plasma",
        )

    # Ticks and labels
    axes[0].set_title("Ground Truth", size=15)
    axes[1].set_title("Prediction", size=15)
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)

    if title:
        fig.suptitle(title, size=20)

    return fig


@matplotlib.rc_context(utils.fractional_plot_bundle(1))
def plot_spatial_error(error, obs_mask, data_config, title=None, vrange=None):
    """
    Plot errors over spatial map
    Error and obs_mask has shape (N_grid,)
    """
    # Get common scale for values
    if vrange is None:
        vmin = error.min().cpu().item()
        vmax = error.max().cpu().item()
    else:
        vmin, vmax = vrange

    # Set up masking of border region
    mask_reshaped = obs_mask.reshape(*data_config.grid_shape_state)
    pixel_alpha = (
        mask_reshaped.clamp(0.7, 1).cpu().numpy()
    )  # Faded border region

    fig, ax = plt.subplots(
        figsize=(5, 4.8),
        subplot_kw={"projection": data_config.coords_projection},
    )

    ax.coastlines()  # Add coastline outlines
    error_grid = error.reshape(*data_config.grid_shape_state).cpu().numpy()

    im = ax.imshow(
        error_grid,
        origin="lower",
        alpha=pixel_alpha,
        vmin=vmin,
        vmax=vmax,
        cmap="OrRd",
    )

    # Ticks and labels
    cbar = fig.colorbar(im, aspect=30)
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.yaxis.get_offset_text().set_fontsize(10)
    cbar.formatter.set_powerlimits((-3, 3))

    if title:
        fig.suptitle(title, size=10)

    return fig


def plot_hist(var_name, data):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data)
    mx = np.max(data)
    mn = np.min(data)
    
    # Make proper bin sizes using the equation max-min/sqrt(n). Then
    # extend the bin range to 4x the standard deviation
    binsize = 0.25
    print('binsize: ', binsize)
    bins = np.arange(mean-(4*std),mean+(4*std),binsize)

    # Now plot figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(data, bins=20)
    
    # Add labels
    plt.xlabel(var_name)
    plt.ylabel('Count')
    plt.title(f'{var_name} ratio from ', fontsize=14)
    text =f' total: {n}\n mean: {mean:.4f}\n std: {std:.4f}\n max: {mx:.4f}\n min: {mn:.4f}'
    ax.text(0.2, 0.7, text, transform=ax.transAxes, fontsize=12)
    # data = df[var_name]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.hist(data)
    # plt.title(var_name)
    dpi=150
    plt.tight_layout()
    pngfile = f'figures/error_ratio_inv_hist_{var_name}.png'
    fig.savefig(pngfile)


def plot_map(var_name, x, y, z, title='title'):
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    zmin = z.min()
    zmax = z.max()
    zstd = z.std()
    zavg = z.mean()
    # zcnt = z.size()
    # Set colorbar
    cmax =  zmax
    cmin =  zmin
    cmap=plt.get_cmap('jet')
    
    # Set plot variable unit
    units = 'W m-2 sr-1 m'
    units = 'K'

    fig = plt.figure(figsize=(12,8))
    
    # Initialize the plot pointing to the project
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # Get scatter data
    sc = ax.scatter(x, y,
                    c=z, s=1.5, marker="o", linewidth=6, alpha=1.0, vmin=cmin, vmax=cmax,
                    transform=ccrs.PlateCarree(), cmap=cmap, norm=None, edgecolor='none', antialiased=True)
    
    # Plot colorbar
    cbar = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.1, fraction=0.15, aspect=40, extend='both')
    cbar.ax.set_xlabel(units, fontsize=10, loc='right')
    
    # Plot globally
    # ax.set_global()

    # Add land and ocean
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    
    # Add gridlines
    gline=ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, color='lightgray', alpha=0.5, linewidth=1.0, linestyle='--')
    gline.top_labels=False
    gline.right_labels=False
    
    # Get title and png file from the input filename
    
    title = f'{title}'
    # Add figure labels
    ax.set_title(title, pad=15, fontsize=20)
    #   text = f"Total Count: {zcnt:0.0f}     Max: {zmax:0.3f}     Min: {zmin:0.3f}     Mean: {zavg:0.3f}     Std: {zstd:0.3f} {units}"
    text = f"     Max: {zmax:0.3f}     Min: {zmin:0.3f}     Mean: {zavg:0.3f}     Std: {zstd:0.3f}"
    ax.text(0.2, -0.1, text, transform=ax.transAxes, va='bottom', fontsize=12)
    dpi=150
    plt.tight_layout()
    pngfile = f'figures/map_{var_name}.png'
    fig.savefig(pngfile)

