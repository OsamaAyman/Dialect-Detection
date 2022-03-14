import os
from typing import Dict, List, Union
import joblib
import numpy as np
import tensorflow as tf



class Model():
    def __init__(self) -> None:

        self.__models_path=os.path.join( os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))), 'models')

        if not os.path.exists(self.__models_path):
            os.mkdir(self.__models_path)

        self.__ml_model_path = os.path.join(
            self.__models_path, 'tfidf-SVM_model.joblib')

        if os.path.exists(self.__ml_model_path):
            self.ml_model = joblib.load(self.__ml_model_path)

        else:
            self.ml_model = None


        self.__dl_model_path = os.path.join(
            self.__models_path, 'lstm_bi')

        self.__dl_tokenizer_path = os.path.join(
            self.__models_path, 'tokenizer.joblib')

        self.__dl_encoder_path = os.path.join(
            self.__models_path, 'LabelEncoder.joblib')


        if os.path.exists(self.__dl_model_path) and os.path.exists(self.__dl_tokenizer_path) and\
                os.path.exists(self.__dl_encoder_path):
            self.dl_model = tf.keras.models.load_model('dialect_api/models/lstm_bi')
            self.tokin = joblib.load('dialect_api/models/tokenizer.joblib')
            self.labelencoder = joblib.load('dialect_api/models/LabelEncoder.joblib')

        else:
            self.dl_model = None



    def predict_ml(self,texts: List[str])->List[Dict]:
        response = []
        model=self.ml_model

        if model:
            predictions = model.predict(texts)
            for i, pred in enumerate(predictions):
                row_pred = {}
                row_pred['text'] = texts[i]
                row_pred['prediction'] = pred
                response.append(row_pred)
        else:
            raise Exception("No Trained model was found.")
        return response

    def predict_dl(self, texts: List[str]) -> List[Dict]:
        response = []
        model = self.dl_model

        if model:
            sequences = self.tokin.texts_to_sequences(texts)
            seq = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=50)
            pred = np.argsort(self.dl_model .predict(seq))
            for i, text in enumerate(texts):
                class_num = pred[i][-1]
                row_pred = {}
                row_pred['text'] = text
                row_pred['prediction'] = self.labelencoder.inverse_transform([class_num])[0]
                response.append(row_pred)

        else:
            raise Exception("No Trained model was found.")
        return response


