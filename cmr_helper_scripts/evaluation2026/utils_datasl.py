import os
import h5py
import numpy as np
import pandas as pd
import hdf5storage

def read_params_csv(filepath):
    """
    Read acquisition/reconstruction parameters from a CSV file.

    The CSV is expected to have at least one row; only the first row is used.
    Values are parsed into Python/numpy types according to field name:

    - Array-like fields are split by ';' and stripped:
        * resolution, FOV, VENC -> list[float]
        * matrix_size           -> np.ndarray[int]
        * spatial_order         -> list[str]
        * venc_order            -> list[str]
    - Scalar float fields:
        * RR, FA, TE, TR, field_strength -> float
    - Missing (NaN) values are converted to None.
    - All other fields are returned as-is.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.

    Returns
    -------
    dict | None
        Parsed parameters dictionary, or None if the CSV is empty.

    Raises
    ------
    RuntimeError
        If the file cannot be read or parsing fails.
    """
    try:
        df = pd.read_csv(filepath, encoding="utf-8")
        if df.empty:
            return None

        row = df.iloc[0]

        array_fields = {"resolution", "FOV", "VENC", "matrix_size", "spatial_order", "venc_order"}
        float_list_fields = {"resolution", "FOV", "VENC"}
        int_array_fields = {"matrix_size"}
        float_scalar_fields = {"RR", "FA", "TE", "TR", "field_strength"}

        params = {}

        for key, value in row.items():
            if pd.isna(value):
                params[key] = None
                continue

            if key in array_fields:
                items = [s.strip() for s in str(value).split(";") if s.strip() != ""]
                if key in int_array_fields:
                    params[key] = np.array([int(float(v)) for v in items], dtype=int)
                elif key in float_list_fields:
                    params[key] = [float(v) for v in items]
                else:
                    params[key] = items
                continue

            if key in float_scalar_fields:
                params[key] = float(value)
                continue

            params[key] = value

        return params

    except Exception as exc:
        raise RuntimeError(f"Failed to read params CSV: {filepath}") from exc
    
def save_mat(out_path: str, key: str, arr: np.ndarray, overwrite: bool = True):
    """
    Save a NumPy array to a MATLAB .mat file (via hdf5storage, typically v7.3/HDF5).

    Args:
        out_path: Output .mat file path.
        key: Variable name to use inside the .mat file (the name you get in MATLAB).
        arr: NumPy array (or array-like) to save.
        overwrite: If True and the file exists, delete it first.
    """
    if overwrite and os.path.exists(out_path):
        os.remove(out_path)
    arr = np.asarray(arr)
    # Reverse the axis order, e.g. (H, W, C) -> (C, W, H).
    # This is sometimes done to match MATLAB/Fortran-style conventions or
    # to keep consistency with a particular downstream loader.
    arr = np.transpose(arr, axes=tuple(range(arr.ndim - 1, -1, -1)))
    arr = np.ascontiguousarray(arr)
    options = hdf5storage.Options(
        matlab_compatible=True,      
        store_python_metadata=False,  
        compress=True,               
    )
    hdf5storage.savemat(out_path, {key: arr}, options=options)

def load_mat(path, key, complex_mode="auto"):
    """
    Lazily read a dataset from a MAT v7.3-style (HDF5) file.

    This function returns a lightweight object that keeps the HDF5 file open and
    supports lazy slicing. Actual IO happens only when indexing (obj[idx]).

    Supported interface of the returned object:
    - obj.shape
    - obj.dtype
    - obj[idx]         (lazy slice/read)
    - obj.close()
    - context manager: with load_mat(...) as obj: ...

    Complex handling depends on complex_mode and on how the dataset is stored:
    - If the stored dtype is a compound dtype with fields "real" and "imag",
      it can be converted to a numpy complex array on read.

    Parameters
    ----------
    path : str
        Path to the HDF5/MAT v7.3 file.
    key : str
        Dataset key inside the file.
    complex_mode : {"auto", "struct", "complex"}, default "auto"
        - "auto": convert to complex only if dtype is compound ("real","imag"),
                  otherwise return raw data.
        - "struct": always return raw stored data (no conversion).
        - "complex": force conversion; raises TypeError if dtype is not compound
                     ("real","imag").

    Returns
    -------
    object
        Lazy dataset reader object.

    Raises
    ------
    KeyError
        If the key does not exist in the HDF5 file.
    ValueError
        If complex_mode is not one of the supported values.
    TypeError
        If complex_mode="complex" but the stored dtype is not compatible.
    """

    class _Lazy:
        def __init__(self, path, key):
            self.path = path
            self.key = key
            self._f = h5py.File(path, "r")
            self.dset = self._f[key]

        @property
        def shape(self):
            return self.dset.shape

        @property
        def dtype(self):
            return self.dset.dtype

        def __getitem__(self, idx):
            x = self.dset[idx]

            if complex_mode == "struct":
                return x

            dt = getattr(x, "dtype", None)
            is_compound_ri = (
                dt is not None
                and dt.fields is not None
                and ("real" in dt.fields)
                and ("imag" in dt.fields)
            )

            if complex_mode == "auto":
                if is_compound_ri:
                    return x["real"] + 1j * x["imag"]
                return x

            if complex_mode == "complex":
                if not is_compound_ri:
                    raise TypeError(f"{key} is not stored as compound (real/imag); dtype={dt}")
                return x["real"] + 1j * x["imag"]

            raise ValueError("complex_mode must be one of: 'auto', 'struct', 'complex'")

        def close(self):
            if getattr(self, "_f", None) is not None:
                self._f.close()
                self._f = None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

    return _Lazy(path, key)


def save_coo_npz(path, arr):
    """
    Save COO to a compressed .npz with fields: coords, data, shape.

    Parameters
    ----------
    path : str
        File path to save.
    arr : np.ndarray
        Dense array (complex64).
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    arr = np.asarray(arr).astype('complex64')
    coords = np.argwhere(arr != 0).astype(np.int32)
    data = arr[tuple(coords.T)] if coords.size else arr.reshape(-1)[:0]

    np.savez_compressed(
        path,
        coords=coords,
        data=data,
        shape=np.array(arr.shape, dtype=np.int64),
    )


def load_coo_npz(path, as_dense=True):
    """
    Load COO from .npz.

    Parameters
    ----------
    path : str
        File path to load.
    as_dense : bool
        If True, returns a reconstructed dense ndarray.

    Returns
    -------
    (coords, data, shape) by default.
    If as_dense=True, returns a reconstructed dense ndarray.
    """
    z = np.load(path)
    coords = z["coords"]
    data = z["data"]
    shape = tuple(z["shape"])

    if not as_dense:
        return coords, data, shape

    out = np.zeros(shape, dtype=data.dtype)
    if coords.size:
        out[tuple(coords.T)] = data
    return out