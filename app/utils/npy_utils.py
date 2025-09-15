import numpy as np


def save_npy(data: np.ndarray, file_path: str):
    np.save(file_path, data)
