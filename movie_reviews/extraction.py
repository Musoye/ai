from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(strip_accents=None, lowercase=False, preprocessor=None)
small_param_grid = [
 {
  'vect__ngram_range': [(1, 1)],
  'vect__stop_words': [None],
  'vect__tokenizer': [tokenizer, tokenizer_porter],
  'clf__penalty': ['l2'],
  'clf__C': [1.0, 10.0]
  },
  {
   'vect__ngram_range': [(1, 1)],
   'vect__stop_words': [stop, None],
   'vect__tokenizer': [tokenizer],
   'vect__use_idf':[False],
   'vect__norm':[None],
   'clf__penalty': ['l2'],
   'clf__C': [1.0, 10.0]
   },
]
lr_tfidf = Pipeline([('vect', tfidf),('clf', LogisticRegression(solver='liblinear'))])
gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid, scoring='accuracy', cv=5, verbose=2, n_jobs=1)
gs_lr_tfidf.fit(X_train, y_train)


import re
def preprocessor(text):
    text = re.sub('<[^]*', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = (re.sub('[\W]+', ' ', text.lower()) +' '.join(emoticons).replace('-', ''))
    return text