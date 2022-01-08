import string
import re
from pyvi import ViTokenizer
from nltk.stem.porter import PorterStemmer

###################################
# LOAD STOPWORDS
with open('data/vietnamese-stopwords.txt', encoding='utf-8') as f:
    lines = f.read()
    
# Danh sách stopword
stopwords = lines.split('\n')
# insert underscore between words
stopwords = [re.sub(r'\s+', '_', word) for word in stopwords]
stopwords[:10]


###################################
def text_preprocessor(X):
    ps = PorterStemmer()
    preprocessed_text = []
    for news in X:
        # remove special characters
        news = [word for word in news if word not in string.punctuation]
        news = ''.join(news)
        
        # lower case
        news = news.lower()
        
        # tokenizing
        news = ViTokenizer.tokenize(news)
        
        # remove stopwords and stemming
        temp = news.split()
        news = [ps.stem(word) for word in temp if word not in stopwords]
        news = ' '.join(news)
        
        preprocessed_text.append(news)
        
    return preprocessed_text











