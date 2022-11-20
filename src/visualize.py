import numpy as np
from matplotlib import pyplot as plt

Z_IDX = 0
C0 = 299792458
FC = 77e9
LAMBDA = C0 / FC

def display(
    img,
    color_map=plt.get_cmap("viridis"),
    img_title=None,
    cmap_label=None,
    alphadata=None,
    xvec=None,
    yvec=None,
    dynamic_range=None,
    clim=None,
    xlabel=None,
    ylabel=None,
    ax = None
):
    """Helper function to display data as an image, i.e., on a 2D regular raster.

    Args:
        img: 2D array.
        color_map: The Colormap instance or registered colormap name to map scalar data to colors.
        img_title: Text to use for the title.
        cmap_label: Set the label for the x-axis.
        alphadata: The alpha blending value, between 0 (transparent) and 1 (opaque).
        xvec: coordinate vectors in x.
        yvec: coordinate vectors in y.
        dynamic_range: The dynamic range that the colormap will cover.
        clim: Set the color limits of the current image.
    """

    max_image = np.max(img)
    if dynamic_range is None:
        imshow_args = {}
    else:
        imshow_args = {"vmin": max_image - dynamic_range, "vmax": max_image}

    if xvec is None or yvec is None:
        ax.imshow(img, cmap=color_map, alpha=alphadata, origin="lower", **imshow_args)
    else:
        im = ax.imshow(
            img,
            cmap=color_map,
            alpha=alphadata,
            extent=[xvec[0], xvec[-1], yvec[0], yvec[-1]],
            origin="lower",
            **imshow_args,
        )

    if clim is not None:
        cbar = plt.colorbar(im)
        cbar.mappable.set_clim(clim)

    if img_title is not None:
        ax.set_title(img_title)

    if xlabel is not None:
        ax.set_xlabel(xlabel)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    # cbar = ax.colorbar()
    # cbar.ax.set_ylabel(cmap_label)
  


def visualize_features(features, vis_dict, filename=None):
    fig, axs = plt.subplots(2, 3, figsize=(17, 10))
    
    x_vec = vis_dict["x_vec"]
    y_vec = vis_dict["y_vec"]
    z_vec = vis_dict["z_vec"]
    alpha_data = vis_dict["alpha_data"]
    Nx = x_vec.size
    Ny = y_vec.size

    kx = (np.arange(-Nx / 2, Nx / 2 - 1)) / ((Nx - 1) * np.diff(x_vec[:2]))
    ky = (np.arange(-Ny / 2, Ny / 2 - 1)) / ((Ny - 1) * np.diff(y_vec[:2]))

    Nx = x_vec.size
    Ny = y_vec.size

    kx_n = kx * LAMBDA
    ky_n = ky * LAMBDA
    max_z = vis_dict['max_z']

    display(
        features[0],
        img_title="Maximum intensity projection (MIP)",
        cmap_label="Normalized magnitude in dB",
        xvec=x_vec,
        yvec=y_vec,
        dynamic_range=30,
        xlabel="$x$ in m",
        ylabel="$y$ in m",
        ax = axs[0,0]
    )
    
    display(
        features[1],
        color_map=plt.get_cmap("twilight"),
        img_title="MIP Phase",
        cmap_label="Phase in degree",
        alphadata=alpha_data,
        xvec=x_vec,
        yvec=y_vec,
        xlabel="$x$ in m",
        ylabel="$y$ in m",
        ax = axs[0,1]
    )
    
    display(
        features[2],
        color_map=plt.get_cmap("twilight"),
        img_title=f"Single slice phase (z = {z_vec[max_z - 1]:.4f} m)",
        cmap_label="Phase in degree",
        alphadata=alpha_data,
        xvec=x_vec,
        yvec=y_vec,
        xlabel="$x$ in m",
        ylabel="$y$ in m",
        ax = axs[0,2]
    )

    display(
        features[3],
        img_title="MIP Distance",
        cmap_label="Distance in m",
        alphadata=alpha_data,
        xvec=x_vec,
        yvec=y_vec,
        clim=(0.15, 0.28),
        xlabel="$x$ in m",
        ylabel="$y$ in m",
        ax = axs[1,0]
    )

    display(
        features[4],
        img_title="MIP 2D FFT",
        dynamic_range=35,
        xvec=kx_n,
        yvec=ky_n,
        xlabel="$k_x$ in $2\\pi \\,/\\, \\lambda$",
        ylabel="$k_y$ in $2\\pi \\,/\\, \\lambda$",
        ax = axs[1,1]
    )

    display(
        features[5],
        img_title=f"Single slice 2D FFT (z = {z_vec[max_z - 1]:.4f} m)",
        dynamic_range=35,
        xvec=kx_n,
        yvec=ky_n,
        xlabel="$k_x$ in $2\\pi \\,/\\, \\lambda$",
        ylabel="$k_y$ in $2\\pi \\,/\\, \\lambda$",
        ax = axs[1,2]
    )

    plt.savefig('../out/' + filename + '.png')

    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data
       



