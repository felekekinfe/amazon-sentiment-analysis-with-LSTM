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

# nltk.download('punkt', quiet=True)
# nltk.download('stopwords', quiet=True)
# nltk.download('wordnet', quiet=True)
# nltk.download('omw-1.4', quiet=True)
# nltk.download('punkt_tab')


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
    lemmatizer=WordNetLemmatizer()
    tokens=[lemmatizer.lemmatize(token) for token in tokens]
    return  ' '.join(tokens)


print(clean_text('The runners were running quickly!! \\n"It\'s amazing," she said.'))

def vectorize_text(texts,max_words=10000,max_len=200,use_tfidf=False):
    """
    Convert text to sequences or TF-IDF vectors.
    
    Args:
        texts (list): Cleaned texts.
        max_words (int): Vocabulary size.
        max_len (int): Sequence length.
        use_tfidf (bool): If True, use TF-IDF.
    
    Returns:
        tuple: (tokenizer/vectorizer, train_vectors, test_vectors)
        
   """
    if use_tfidf:
        vectorizer=TfidfVectorizer(max_features=max_words)
        vectors=vectorizer.fit_transform(texts)
        mid=len(texts)//2
        return vectorizer,vectors[:mid],vectors[mid:]
    else:
        tokenizer = Tokenizer(num_words=max_words)
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences, maxlen=max_len)
        mid = len(texts) // 2
        return tokenizer, padded[:mid], padded[mid:]