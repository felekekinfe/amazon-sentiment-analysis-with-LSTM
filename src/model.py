
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