import platform
from flask import Flask
from flask import request
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import logging
import json

# import image caption model
from modules.caption_image import DittoImageCaption

# import image Q/A model
from modules.qa_image import DittoQAImage

# set up logging for server
log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO)

# set Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# set OS variable
OS = "Windows"
if platform.system() == "Linux":
    OS = "Linux"
elif platform.system() == "Darwin":
    OS = "Darwin"

# initialize image caption model
caption_model = DittoImageCaption()

# initialize image Q/A model
qa_model = DittoQAImage()

@app.route("/caption", methods=["POST"])
def caption_handler():
    # get image from request
    if "image" not in request.files:
        return ErrMissingFile("image")
    image = Image.open(BytesIO(base64.b64decode(request.files['image'].read())))
    try:
        log.info(f"sending image to caption model")
        caption = caption_model.get_caption(image=image)
        print()
        print(f'Generated caption: {caption}')
        print()
        return json.dumps({"response": caption})
    except BaseException as e:
        log.error(e)
        return ErrException(e)


@app.route("/qa", methods=["POST"])
def qa_handler():
    requests = request.args

    if "prompt" not in requests:
        return ErrMissingArg("prompt")
    if "image" not in request.files:
        return ErrMissingFile("image")
    
    prompt = requests["prompt"]
    image = image = Image.open(BytesIO(base64.b64decode(request.files['image'].read())))

    try:
        log.info(f"sending prompt and image to Q/A model: {prompt}")
        qa_response = qa_model.query(query=prompt, image=image)
        return json.dumps({"response": qa_response})
    
    except BaseException as e:
        log.error(e)
        return ErrException(e)


@app.route("/status", methods=["GET"])
def status_handler():
    return '{"status": "on"}'


class Server:
    def __init__(self):
        self.app = app


if __name__ == "__main__":
    server = Server()


def ErrMissingFile(file: str):
    return '{"error": "missing file %s"}' % file

def ErrMissingArg(arg: str):
    return '{"error": "missing argument %s"}' % arg

def ErrException(e: BaseException):
    return '{"error": "%s"}' % e
