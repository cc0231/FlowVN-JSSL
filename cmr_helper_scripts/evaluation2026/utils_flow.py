import numpy as np
import scipy

def k2i_numpy(x, ax=None):
    """
    Convert k-space into image with ifft.

    Args:
        x (np.ndarray):
            K-space array (typically complex-valued).
        ax (list[int] | tuple[int] | None):
            Axes along which to apply the inverse FFT. If None, defaults to [-2, -1].

    Returns:
        np.ndarray:
            Image-domain array with the same shape as x.

    """
    if ax is None:
        ax = [-2, -1]
    return scipy.fft.fftshift(
        scipy.fft.ifftn(
            scipy.fft.ifftshift(x, axes=ax),
            axes=ax,
            norm="ortho",
            workers=-1,
        ),
        axes=ax,
    )

def i2k_numpy(x, ax=None):
    if ax is None:
        ax = [-2, -1]
    return scipy.fft.fftshift(scipy.fft.fftn(scipy.fft.ifftshift(x, axes=ax), axes=ax, norm="ortho", workers=-1),axes=ax)
                            
def complex2magflow(x, venc=None):
    """
    Convert complex 4D Flow images into magnitude and velocity.

    Args:
        x (np.ndarray):
            Complex image array with velocity-encoding dimension first: x[v, ...].
              - v=0: reference / flow-compensated encoding
              - v=1..Nv-1: flow-encoded directions (typically 3 directions)
        venc (np.ndarray | None):
            Per-direction VENC values corresponding to x[1:] (typically shape (3,)).
            If provided, phase (radians) is converted to velocity units via:
                vel = phase / pi * VENC

    Returns:
        mag (np.ndarray):
            Image Magnitude.
        flow (np.ndarray):
            If venc is None:
                Wrapped phase difference in radians:
                    angle(x[1:] * conj(x[0:1]))
            If venc is provided:
                Velocity (same physical units as venc) computed as:
                    angle(x[1:] * conj(x[0:1])) / pi * VENC
    """
    mag = np.abs(x)
    flow = np.angle(x[1:] * np.conj(x[0:1]))
    if venc is not None:
        flow = flow / np.pi * venc[:, None, None, None, None]
    return mag, flow