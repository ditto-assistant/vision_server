from transformers import ViltProcessor, ViltForQuestionAnswering
import torch
from PIL import Image

class DittoQAImage:
    def __init__(self):
        self.load_model()
        self.response = "Loading..."

    def load_model(self):
        try:
            self.processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            self.model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")            
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            print("Error loading image caption model...")
            self.model  = []

    def query(self, query, image_path=None, image: Image = None):
        if image_path:
            image = Image.open(image_path).convert("RGB")
        elif image:
            image = image
        else:
            print("No image provided...")
            return None

        # prepare inputs
        encoding = self.processor(image, query, return_tensors="pt")

        # forward pass
        outputs = self.model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        
        res = self.model.config.id2label[idx]
        confidence = torch.softmax(logits, dim=1)[0, idx].item()
        
        # print()
        # print("Predicted answer:", res)
        # print()

        return res, confidence

if __name__ == '__main__':

    test_image = 'river-sunset.png'

    ditto_qa_image = DittoQAImage()

    res = ditto_qa_image.query(
        'Is there a person in this image?',
        image_path=test_image
    )
