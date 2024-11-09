#!/usr/bin/env python3
import pandas as pd
df = pd.read_csv('movie_data.csv', encoding='utf-8')
df.rename(columns={'0': 'review', '1': 'sentiment'}, inplace=True)
df.head(3)
df.shape
from sklearn.feature_extraction.text import CountVectorizer
count = CountVectorizer(stop_words='english', max_df=.1, max_features=5000)
X = count.fit_transform(df['review'].values)
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')
X_topics = lda.fit_transform(X)
print(lda.components_.shape)
n_top_words = 5
feature_names = count.get_feature_names_out()
for topic_idx, topic in enumerate(lda.components_):
    print(f'Topic {(topic_idx + 1)}:')
    print(' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))