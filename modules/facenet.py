import numpy as np
from mtcnn import MTCNN
from PIL import Image
from keras_facenet import FaceNet

class Facenet:

    def __init__(self):
        self.load_models()
        self.create_load_face_vector_store()
    
    def load_models(self):
        self.facenet_model = FaceNet()
        self.detector_model = MTCNN()
    
    def create_load_face_vector_store(self):
        self.face_vector_store = {}
        # TODO: create a face vector store that can be loaded and saved
        # Tell the LLm agent that any peson you are talking to with a 
        # face vector similarity score of 0.65 or higher is the same person
        # and should be treated as such, i.e. when the user tells the agent
        # "my name is John" and then later says "Who am I" the agent should
        # know that the user is referring to the same person from the previous
        # conversation and visual memory.


    def extract_face(self, image: Image):
        # convert to RGB, if needed
        image = image.convert('RGB')
        # resize to 160x160
        image = image.resize((160, 160))
        # convert to array
        pixels = np.asarray(image)
        # detect faces in the image
        results = self.detector_model.detect_faces(pixels)
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize((160, 160))
        face_array = np.asarray(image)
        return face_array
        
    def get_embeddings(self, image: Image):
        # convert to RGB, if needed
        image = image.convert('RGB')
        # get face
        face_array = self.extract_face(image)
        # get embedding
        embeddings = self.facenet_model.embeddings([face_array])
        return embeddings[0]  
    
    def get_similarity(self, embedding1, embedding2):
        # calculate distance between embeddings
        distance = np.linalg.norm(embedding1 - embedding2)
        # calculate similarity
        similarity = 1.0 / (1.0 + distance)
        return similarity

if __name__ == '__main__':
    img1 = 'sample_images/face1.jpg' # same person
    img2 = 'sample_images/face2.jpg' # same person
    img3 = 'sample_images/face3.jpg' # different person

    facenet = Facenet()
    image1 = Image.open(img1)
    image2 = Image.open(img2)
    image3 = Image.open(img3)

    embedding1 = facenet.get_embeddings(image1)
    embedding2 = facenet.get_embeddings(image2)
    embedding3 = facenet.get_embeddings(image3)

    print('same person (1 vs 2):')
    print(facenet.get_similarity(embedding1, embedding2))

    print('different person (1 vs 3):')
    print(facenet.get_similarity(embedding1, embedding3))

    print('different person (2 vs 3):')
    print(facenet.get_similarity(embedding2, embedding3))