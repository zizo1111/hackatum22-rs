import numpy as np


def compute_slice(_volume, _z_idx=0):
    """Compute a single slice of the given 3D volumetric data in z-direction.

    Args:
        _volume: 3D volumetric data.
        _z_idx: Index on z-direction.

    Returns:
        A single slice (img) of the given 3D volumetric data in z-direction.
    """

    img = _volume[:, :, _z_idx]
    return img


def compute_mip(_volume):
    """Compute the maximum intensity projection (MIP) of a 3D volume.

    Args:
        _volume (ndarray): 3D volumetric data.

    Returns:
        The MIP of the 3D volume (v_max) and the estimated maximum indices in z-direction (k_max).
    """

    k_max = np.argmax(np.abs(_volume), axis=2)
    i, j = np.indices(k_max.shape)
    v_max = _volume[i, j, k_max]

    return v_max, k_max


def complex2magphase(data):
    """Computes the magnitude and the phase of each data.

    Args:
         data: Any complex data (a + bi).

    Returns:
        Computes the magnitude (mag) and the phase (phi) for each element of the complex data.
    """

    mag = np.abs(data)
    phi = np.angle(data)

    return mag, phi


def compute_fft(img):
    """Compute the 2D FFT of an image.

    Args:
        img (ndarray): 2D array.

    Returns:
        2D FFT of the image in x and y-direction.
    """

    fft_img = np.fft.fftshift(np.fft.fft2(img), axes=(0, 1))
    return fft_img