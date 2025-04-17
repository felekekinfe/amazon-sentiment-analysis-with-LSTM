
"""
LSTM model for Amazon Review Polarity Dataset with logistic regression baseline.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SentimentModel:
    def __init__(delf,max_words=10000,max_len=200,embedding_dim=100):
        """
        Initialize model.
        
        Args:
            max_words (int): Vocabulary size.
            max_len (int): Sequence length.
            embedding_dim (int): Embedding dimension.
        """
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.lstm_model = None
        self.lr_model = None
        self.tokenizer = None
    
    def buils_lstm(self):
        """
        Build LSTM model.
        
        Returns:
            Sequential: Compiled model.
            
        """

        model=Sequential([
            Embedding(self.max_words,self.embedding_dim,input_length=self.max_len),
            SpatialDropout1D(0.2),
            LSTM(100,dropout=0.2,recurrent_dropout=0.2),
            Dense(1,activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, X_train, y_train, X_val, y_val, X_train_tfidf, X_val_tfidf, epochs=5, batch_size=64):
        """
        Train models.
        
        Args:
            X_train, X_val: Sequences.
            y_train, y_val: Labels (0 = negative, 1 = positive).
            X_train_tfidf, X_val_tfidf: TF-IDF vectors.
            epochs (int): Epochs.
            batch_size (int): Batch size.
        """

        
        
        
        logging.info("Training LSTM model...")
        self.lstm_model=self.build_lstm()
        self.lstm_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1

        )

        logging.info("Training logistic regression model...")
        self.lr_model=LogisticRegression(max_iter=1000)
        self.lr_model.fit(
            X_train_tfidf, y_train

        )

    def predict(self,X):

        """
        Predict with LSTM.
        
        Args:
            X: Sequence.
        
        Returns:
            np.array: Probabilities.
        """

        return self.lstm_model.predict(X,verbose=0).flatten()

    def save_models(self,lstm_path,lr_path):
        """
        Save models.
        
        Args:
            lstm_path (str): LSTM path.
            lr_path (str): Logistic regression path.
        """

        self.lstm_model.save(lstm._path)
        with open(lr_path, 'wb') as f:
            pickle.dump(self.lr_model,f)
        logging.info(f"Models saved to {lstm_path} and {lr_path}")

    def load_models(self,lstm_path,lr_path):
        """
        Load models.
        
        Args:
            lstm_path (str): LSTM path.
            lr_path (str): Logistic regression path.
        """

        self.lstm_model=tf.keras.models.load_model(lstm_path)
        with open(lr_path,'rb') as f:
            self.lr_model=pickle.load(f)
        logging.info(f"Models loaded from {lstm_path} and {lr_path}")

    def analyze_errors(self, X_test, y_test, reviews):
        """
        Analyze errors.
        
        Args:
            X_test: Sequences.
            y_test: Labels.
            reviews: Texts.
        
        Returns:
            list: Misclassified reviews.
        
        """

        pred_probs=self.predict(X_test)
        pred_labels=(pred_probs>0.5).astype(int)
        misclassified = []
        
        for review_text,true_label,pred_label,confidence in zip(reviews, y_test, pred_labels, pred_probs):
            if true_label != pred_label:
                misclassified.append({
                    'review': review_text,
                    'true_label': true_label,
                    'predicted_label': pred_label,
                    'confidence': float(confidence)
                })
        return misclassified
