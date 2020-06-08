from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pandas as pd
import string
import re

def ProcessText(text):
    '''
    Takes in a line of text, removes punctutations and
    html tags(data cleaning) and returns a list of words
    '''
    if type(text) == list:
        text = ' '.join(text)
    nopunc = [char for char in text.split() if char not in string.punctuation]
    nopunc = ' '.join(nopunc)
    cleaned_word = " ".join([word for word in nopunc.split() if word.lower() not in stopwords.words('english') ])
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '',cleaned_word)
    return cleantext.split()

movie_train = pd.read_csv('Train.csv')
movie_test = pd.read_csv('Test.csv')

movie_train_clean = movie_train['text'].apply(ProcessText)
bow_transformer = CountVectorizer(analyzer=ProcessText).fit(movie_train_clean)
message_bow = bow_transformer.transform(movie_train_clean)
tfidf_transformer = TfidfTransformer().fit(message_bow)
message_tfidf = tfidf_transformer.transform(message_bow)

def DoGridSearchCV():
    """
    Performs grid search and returns the optimal parameters for SVC model.
    
    """
    param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
    grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
    grid.fit(message_tfidf, movie_train['label'])
    return grid.best_params_['C'], grid.best_params_['gamma'], grid.best_params_['kernel']


best_C, best_gamma, best_kernel = DoGridSearchCV()     
pipe = Pipeline([
    ('bow', CountVectorizer(analyzer=ProcessText)),
    ('tfidf', TfidfTransformer()),
    ('classifier',SVC(C=best_C, gamma=best_gamma, kernel=best_kernel))
])

pipe.fit(movie_train['text'],movie_train['label'])
pred_svm = pipe.predict(movie_test['text'])
print(classification_report(movie_test['label'], pred_svm))
