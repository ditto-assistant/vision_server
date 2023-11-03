import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import pickle 

import logging

log = logging.getLogger("facenet")
logging.basicConfig(level=logging.INFO)

class DataLoader:
    def __init__(self, data_file):
        self.data_file = data_file

    def load_data(self):
        # Load data from file and preprocess
        # pickle
        self.data = pickle.load(open(self.data_file, 'rb'))
        x = np.array(self.data['embeddings'])
        y_raw = self.data['labels']
        # Convert labels to one-hot encoding
        blank = np.zeros(len(self.data['class_names']))
        log.info(f'Class names: {self.data["class_names"]}')
        y = []
        for i in range(len(y_raw)):
            y.append(blank.copy())
            y[i][y_raw[i]] = 1
        self.x = x
        self.y = np.array(y)
        

class FacialRecognitionModel(DataLoader):
    def __init__(self, data_file):
        super().__init__(data_file)
        self.load_data()
        x_dim = self.x.shape[1]
        y_dim = self.y.shape[1]
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=x_dim, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(y_dim, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, max_epochs=100, batch_size=32):
        # train model
        es = EarlyStopping(monitor='loss', mode='auto', patience=5)
        self.model.fit(
            self.x, self.y, epochs=max_epochs, batch_size=batch_size, callbacks=[es])
        # save weights
        self.save_weights()

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy

    def predict(self, X):
        ypred = self.model.predict(X, verbose=0)
        face_name_ndx = np.argmax(ypred)
        str_labels = self.data['class_names']
        face_name = str_labels[face_name_ndx]
        return face_name, ypred
    
    def save_weights(self, user_id="admin"):
        log.info(f'Saving facial recognition model weights for user: {user_id}')
        path = f'face_store/{user_id}/facial_recognition_model_weights.h5'
        self.model.save_weights(path)

    def load_weights(self, user_id="admin"):
        log.info(f'Loading facial recognition model weights for user: {user_id}')
        path = f'face_store/{user_id}/facial_recognition_model_weights.h5'
        self.model.load_weights(path)