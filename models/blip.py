from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

class BLIPDecoder:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)

    def generate_caption(self, image: Image.Image) -> str:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption