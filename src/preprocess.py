import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_text(text):
    """
    Clean text by unescaping quotes, newlines, and standardizing.
    
    Args:
        text (str): Raw review (title + body).
    
    Returns:
        str: Cleaned text.
    """

    if not isinstance(text,str):
        return 'text should be string instance'

    text=text.replace('""','"').replace('\\n',' ')
    text=text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens=word_tokenize(text)
    stop_words=set(stopwords.words('english'))
    tokens=[token for token in tokens if token not in stop_words]
    lemmatizer=WordNetLemmatizr()
    tokens=[lemmatizer.lemmatize(token) for token in tokens]
    return  ' '.join(tokens)


