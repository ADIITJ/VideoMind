from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class BLIPDecoder:
    def __init__(self, device='cpu'):
        print("Initializing BLIPDecoder...")
        self.device = torch.device(device)
        print("Loading BLIPProcessor...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
        print("Loading BLIPForConditionalGeneration...")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        print("BLIPDecoder initialized.")

    def generate_caption(self, image: Image.Image) -> str:
        print("Generating caption for image...")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        print(f"Caption generated: {caption}")
        return caption