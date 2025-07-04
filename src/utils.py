import os
import lmdb
import pickle
from PIL import Image

def load_lmdb(path: str, map_size: int = int(1e9)) -> lmdb.Environment:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return lmdb.open(path, map_size=map_size)

def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

def show_image(path: str) -> Image.Image:
    return Image.open(path)