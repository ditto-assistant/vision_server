import platform
from flask import Flask
from flask import request
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import logging
import os
import json

# import image caption model
from modules.caption_image import DittoImageCaption

# import image Q/A model
from modules.qa_image import DittoQAImage

# import FaceNet module
from modules.facenet import DittoFacenet

# set up logging for server
log = logging.getLogger("server")
logging.basicConfig(level=logging.INFO)

# supress tf messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

# initialize FaceNet model
facenet_model = DittoFacenet()

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
        qa_response, confidence = qa_model.query(query=prompt, image=image)
        print(f'Generated response: {qa_response} ({int(confidence * 100)}%)')
        if 'yes' in qa_response and confidence < 0.8:
            qa_response = 'no'
        return json.dumps({"response": qa_response})
    
    except BaseException as e:
        log.error(e)
        return ErrException(e)

@app.route("/scan_face", methods=["POST"])
def scan_face():

    user_id = 'admin'

    if "image" not in request.files:
        return ErrMissingFile("image")
    
    try:
    
        image = image = Image.open(BytesIO(base64.b64decode(request.files['image'].read())))
        prompt = 'Is there a person facing the camera?'

    
        # log.info(f"sending prompt and image to Q/A model: {prompt}")
        qa_response = qa_model.query(query=prompt, image=image)[0]
        if 'yes' in qa_response.lower():
            face_embeddings = facenet_model.get_embeddings(single_image=image)
            res = facenet_model.get_name_from_face_vector(
                face_vector=face_embeddings, 
                user_id=user_id
            )
            face_name = res['face_name']
            confidence = res['confidence']
            if face_name == 'unknown':
                return json.dumps({"face_name": face_name, "person_detected": 'yes'})
            else:
                log.info(f"Facial Recognition Model: {face_name} ({int(confidence * 100)}%)")
                return json.dumps({"face_name": face_name, "person_detected": 'yes'})

        else: # no
            return json.dumps({"face_name": "unknown", "person_detected": 'no'})
    
    except BaseException as e:
        log.error(e)
        return ErrException(e)

@app.route("/save_face", methods=["POST"])
def save_face():
    user_id = 'admin'
    requests = request.args
    if "image" not in request.files:
        return ErrMissingFile("image")
    if "face_name" not in requests:
        return ErrMissingArg("face_name")
    image = image = Image.open(BytesIO(base64.b64decode(request.files['image'].read())))
    face_name = requests["face_name"]
    try:
        face_embeddings = facenet_model.get_embeddings(image=image)
        facenet_model.add_face_vector(face_vector=face_embeddings, face_name=face_name, user_id=user_id)
        log.info(f"added {face_name} to face store")
        return json.dumps({f"response": "added {face_name} to face store"})
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
