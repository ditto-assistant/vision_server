from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import torch

class DittoImageCaption:
    def __init__(self):
        self.load_model()
        self.generated_caption = "Loading..."

    def load_model(self):
        try:
            checkpoint = "microsoft/git-base"
            self.processor = AutoProcessor.from_pretrained(checkpoint)
            self.model = AutoModelForCausalLM.from_pretrained(checkpoint)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            print("Error loading image caption model...")
            self.model  = []

    def get_caption(self, image_path=None, image: Image = None):
        if image_path:
            image = Image.open(image_path).convert("RGB")
        elif image:
            image = image
        else:
            print("No image provided...")
            return None
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        pixel_values = inputs.pixel_values
        generated_ids = self.model.generate(pixel_values=pixel_values, max_length=50)
        self.generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return self.generated_caption

if __name__ == '__main__':

    test_image = 'river-sunset.png'
    ditto_image_caption = DittoImageCaption()

    generated_caption = ditto_image_caption.get_caption(image_path=test_image)
    print(generated_caption)