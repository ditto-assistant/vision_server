import numpy as np
from mtcnn import MTCNN
from PIL import Image
from keras_facenet import FaceNet
from modules.facial_recognition import FacialRecognitionModel
import os
import shutil
import pickle
import logging
import time
import cv2

# impoer K and clear session 
import keras.backend as K

log = logging.getLogger("facenet")
logging.basicConfig(level=logging.INFO)

import tensorflow as tf
# turn off all tensorflow messages
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class DittoFacenet:

    def __init__(self, train_face_recognition_model=False):
        self.load_models(train_face_recognition_model)
        self.create_load_face_vector_store()
    
    def load_models(self, train_face_recognition_model):
        self.facenet_model = FaceNet()
        self.detector_model = MTCNN()
        # check if training data exists for admin
        if not os.path.exists('face_store/admin/training_data.pkl'):
            log.info('No training data found for face recognition model')
        else:
            if train_face_recognition_model:
                pass
            else:
                self.facial_recognition_model = FacialRecognitionModel(
                    data_file='face_store/admin/training_data.pkl'
                )
                self.facial_recognition_model.load_weights()

    def train_facial_recognition_model(self, user_id="admin"):
        self.facial_recognition_model = FacialRecognitionModel(
                data_file='face_store/admin/training_data.pkl'
            )
        self.facial_recognition_model.train(
            max_epochs=100,
            batch_size=32
        )
    
    def create_load_face_vector_store(self, user_id="admin"):
        self.face_vector_store = {}
        if not os.path.exists(f'face_store/{user_id}'):
            os.makedirs(f'face_store/{user_id}')
        if not os.path.exists(f'face_store/{user_id}/face_vector_store.pkl'):
            log.info(f'Creating face vector store for user: {user_id}')
            self.face_vector_store[user_id] = {
                'face_vectors': [],
                'face_names': [],
                'image_paths': [],
            }
            self.save_face_vector_store(user_id=user_id)
        else:
            log.info(f'Loading face vector store for user: {user_id}')
            self.face_vector_store[user_id] = pickle.load(open(f'face_store/{user_id}/face_vector_store.pkl', 'rb'))
    
    def save_face_vector_store(self, user_id="admin"):
        log.info(f'Saving face vector store for user: {user_id}')
        pickle.dump(self.face_vector_store[user_id], open(f'face_store/{user_id}/face_vector_store.pkl', 'wb'))
        

    def add_face_vector(self, face_vector, face_name, user_id="admin"):
        self.face_vector_store[user_id]['face_vectors'].append(face_vector)
        self.face_vector_store[user_id]['face_names'].append(face_name)
        self.save_face_vector_store(user_id=user_id)

    def get_face_vectors_from_name(self, face_name, k=3, user_id="admin"):
        face_vectors = []
        for i, name in enumerate(self.face_vector_store[user_id]['face_names']):
            if i == k-1:
                break
            if name == face_name:
                #  -i to get the most recent face vectors
                face_vectors.append(self.face_vector_store[user_id]['face_vectors'][-i])
        return face_vectors

    def get_name_from_face_vector(self, face_vector, user_id="admin"):
        prediction = self.facial_recognition_model.predict(face_vector)
        K.clear_session()
        return {
            "face_name": prediction[0],
            "confidence" : prediction[1][0][np.argmax(prediction[1])]
        }

    def extract_face(self, image: Image):
        try:
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
        except Exception as e:
            log.error(e)
            return None
        
    def get_embeddings(self, single_image: Image = None, images: list = []):
        try:
            if images == []:
                images = [single_image]
            face_arrays = []
            for image in images:
                # convert to RGB, if needed
                image = image.convert('RGB')
                # get face
                face_array = self.extract_face(image)
                if face_array is None:
                    continue
                face_arrays.append(face_array)
            # get embeddings
            embeddings = self.facenet_model.embeddings(face_arrays)
            return embeddings
        except Exception as e:
            log.error(e)
            return None
    
    def get_similarity(self, embedding1, embedding2):
        # calculate distance between embeddings
        distance = np.linalg.norm(embedding1 - embedding2)
        # calculate similarity
        similarity = 1.0 / (1.0 + distance)
        return similarity
    
    def webcam_scan_save_to_face_store(self, face_name='Human', user_id="admin", first_time=True):
        log.info(f'Initializing webcam for face scanning for: {face_name}')
        cap = cv2.VideoCapture(0)
        images = [] 
        max_images = 50
        log.info(f'Scanning face and saving to face store for user: {user_id}')
        for i in range(max_images):
            log.info(f'Capturing image {i+1} of {max_images}')
            ret, frame = cap.read()
            if ret:
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                images.append(image)
                time.sleep(0.2)
            else:
                break
        cap.release()
        embeddings = self.get_embeddings(images=images)
        for face_vector in embeddings:
            self.add_face_vector(face_vector=face_vector, face_name=face_name)
        log.info(f'Finished scanning face exporting face store as training data for user: {user_id}')
        self.export_face_store_as_training_data(user_id=user_id, first_time=first_time)
    
    def export_face_store_as_training_data(self, user_id="admin", first_time=True):
        log.info(f'Exporting face store as training data for user: {user_id}')
        if first_time:
            # load unknown faces from data/ which contains 100 images of unknown faces
            unknown_faces = []
            for filename in os.listdir('data/'):
                image = Image.open(f'data/{filename}')
                unknown_faces.append(image)
            # get embeddings for unknown faces and save to face store with face_name='unknown'
            embeddings = self.get_embeddings(images=unknown_faces)
            for face_vector in embeddings:
                self.add_face_vector(face_vector=face_vector, face_name='unknown', user_id=user_id)
        # get all unique face names
        face_names = list(set(self.face_vector_store[user_id]['face_names']))
        # initialize training data object
        data = {
            'embeddings': [],
            'labels': []
        }
        # add face vectors and labels to training data object
        for i, face_vector in enumerate(self.face_vector_store[user_id]['face_vectors']):
            data['embeddings'].append(face_vector)
            data['labels'].append(face_names.index(self.face_vector_store[user_id]['face_names'][i]))
        # print class names
        data['class_names'] = face_names.copy()
        # save training data object
        pickle.dump(data, open(f'face_store/{user_id}/training_data.pkl', 'wb'))
        log.info(f'face_names: {face_names}')
        
    def test_facial_recognition_model(self):
        '''Open-cv loop showing face_name and confidence'''
        cap = cv2.VideoCapture(0)
        while True:
            try:
                ret, frame = cap.read()
                if ret:
                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    face_vector = self.get_embeddings(single_image=image)
                    face_name = self.facial_recognition_model.predict(face_vector)
                    cv2.putText(frame, f'{face_name}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow('frame', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
                time.sleep(1)
            except Exception as e:
                log.error(e)
                

if __name__ == '__main__':
    
    TRAIN_FACE_RECOGNITION_MODEL = False

    facenet = DittoFacenet(
        train_face_recognition_model=TRAIN_FACE_RECOGNITION_MODEL
    )
    # facenet.webcam_scan_save_to_face_store(face_name='Omar', first_time=True)
    # facenet.train_facial_recognition_model(user_id='admin')

    # facenet.webcam_scan_save_to_face_store(face_name='Justin', first_time=False)
    # facenet.train_facial_recognition_model(user_id='admin')

    facenet.test_facial_recognition_model()