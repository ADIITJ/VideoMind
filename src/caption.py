import pickle
from PIL import Image
import torch
from .utils import load_lmdb
from models.blip import BLIPDecoder

# Initialize BLIP captioning
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
captioner = BLIPDecoder(device=device)

# LMDB cache for captions
env = load_lmdb('cache/captions.lmdb', map_size=int(1e8))
transform = None  # BLIPProcessor handles resizing internally

def caption_frame(path: str) -> str:
    key = path.encode()
    with env.begin() as txn:
        if v := txn.get(key):
            return pickle.loads(v)
    # load image and caption
    image = Image.open(path).convert('RGB')
    text = captioner.generate_caption(image)
    with env.begin(write=True) as txn:
        txn.put(key, pickle.dumps(text))
    return text