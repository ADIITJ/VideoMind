import pickle
from PIL import Image
import torch
from .utils import load_lmdb

# Initialize BLIP captioning

# LMDB cache for captions
print("Loading LMDB for captions...")
env = load_lmdb('cache/captions.lmdb', map_size=int(1e8))
transform = None  # BLIPProcessor handles resizing internally

def caption_frame(path: str) -> str:
    from models.blip import BLIPDecoder
    device = 'cpu'
    print(f"Creating BLIPDecoder for frame: {path}")
    captioner = BLIPDecoder(device=device)
    key = path.encode()
    with env.begin() as txn:
        if v := txn.get(key):
            print(f"Cache hit for {path}")
            return pickle.loads(v)
    print(f"Cache miss for {path}, generating caption...")
    # load image and caption
    image = Image.open(path).convert('RGB')
    text = captioner.generate_caption(image)
    with env.begin(write=True) as txn:
        txn.put(key, pickle.dumps(text))
    print(f"Caption cached for {path}")
    return text

def refine_caption_with_llm(caption: str) -> str:
    from src.llm_reasoning import refine_caption_llm
    return refine_caption_llm(caption)