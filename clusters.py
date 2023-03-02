from flask import Flask
from flask import request
from flask import app, render_template
import requests
import pandas as pd
import pickle
from nltk.corpus import stopwords
import nltk
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings

#
# import data
data = pd.read_csv("test_csv_feedback.csv")
data['URL'] = data['URL'].str.replace('/content/canadasite', 'www.canada.ca')
data['URL'] = data['URL'].str.replace('www.canada.ca', 'https://www.canada.ca')
data['URL'] = data['URL'].str.replace('https://https://', 'https://')

data = data[["Problem Details", 'Language', 'Date Entered', 'URL']]

data_en = data.loc[data['Language'] == 'en']
data_en = data_en.reset_index(drop=True)

data_fr = data.loc[data['Language'] == 'fr']
data_fr = data_fr.reset_index(drop=True)


if data_en.empty == False:
    def cleanPunc(sentence):
        cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
        cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\n", " ")
        return cleaned

    # function to convert to lowercase
    def keepAlpha(sentence):
        alpha_sent = ""
        for word in sentence.split():
            alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
            alpha_sent += alpha_word
            alpha_sent += " "
        alpha_sent = alpha_sent.strip()
        return alpha_sent

    # function to stem feedbck (English)
    stemmer_en = SnowballStemmer("english")

    def stemming_en(sentence):
        stemSentence = ""
        for word in sentence.split():
            stem = stemmer_en.stem(word)
            stemSentence += stem
            stemSentence += " "
        stemSentence = stemSentence.strip()
        return stemSentence

    clean_columns = ['Comment']
    clean_en = pd.DataFrame(columns=clean_columns)
    clean_en['Comment'] = data_en['Problem Details'].str.lower()
    clean_en['Comment'] = clean_en['Comment'].apply(cleanPunc)
    clean_en['Comment'] = clean_en['Comment'].apply(keepAlpha)
    clean_en['Comment'] = clean_en['Comment'].apply(stemming_en)

    from sklearn.feature_extraction.text import TfidfVectorizer

    all_text_en = []
    all_text_en = clean_en['Comment'].values.astype('U')

    vects_en = []
    all_x_en = []
    vectorizer_en = TfidfVectorizer(
        strip_accents='unicode', analyzer='word', ngram_range=(1, 3), norm='l2')
    vects_en = vectorizer_en.fit(all_text_en)
    all_x_en = vects_en.transform(all_text_en)

    from sklearn.cluster import AffinityPropagation
    from sklearn import metrics
    from sklearn.datasets import make_blobs

    X = all_x_en

    afprop = AffinityPropagation(max_iter=300, damping=0.7)
    afprop.fit(X)
    cluster_centers_indices = afprop.cluster_centers_indices_
    X = X.todense()
    P = afprop.predict(X)

    import collections

    occurrences = collections.Counter(P)

    cluster_columns = ['Number of feedback', 'Representative comment']
    clusters = pd.DataFrame(columns=cluster_columns)

    cluster_rep = list(cluster_centers_indices)
    rep_comment = []
    for indice in cluster_rep:
        rep_comment.append(data_en['Problem Details'][indice])

    clusters['Representative comment'] = rep_comment

    cluster_couple = sorted(occurrences.items())
    cluster_count = []

    cluster_count = [x[1] for x in cluster_couple]
    clusters['Number of feedback'] = cluster_count
    clusters = clusters.sort_values(by='Number of feedback', ascending=False)
    clusters = clusters.reset_index(drop=True)

    data_en['group'] = P
    cluster_group = data_en[['Problem Details', 'Date Entered', 'group']]

    rep_dict = {i: rep_comment[i] for i in range(0, len(rep_comment))}

    cluster_group['group'] = cluster_group['group'].map(rep_dict)

    clusters.to_csv('clusters_en.csv', index=False)
    cluster_group.to_csv('comments_by_cluster_en.csv', index=False)


if data_fr.empty == False:
    def cleanPunc(sentence):
        cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
        cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
        cleaned = cleaned.strip()
        cleaned = cleaned.replace("\n", " ")
        return cleaned

    # function to convert to lowercase
    def keepAlpha(sentence):
        alpha_sent = ""
        for word in sentence.split():
            alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
            alpha_sent += alpha_word
            alpha_sent += " "
        alpha_sent = alpha_sent.strip()
        return alpha_sent

    # function to stem feedbck (English)
    stemmer_fr = SnowballStemmer("french")

    def stemming_fr(sentence):
        stemSentence = ""
        for word in sentence.split():
            stem = stemmer_fr.stem(word)
            stemSentence += stem
            stemSentence += " "
        stemSentence = stemSentence.strip()
        return stemSentence

    clean_columns = ['Comment']
    clean_fr = pd.DataFrame(columns=clean_columns)
    clean_fr['Comment'] = data_fr['Problem Details'].str.lower()
    clean_fr['Comment'] = clean_fr['Comment'].apply(cleanPunc)
    clean_fr['Comment'] = clean_fr['Comment'].apply(keepAlpha)
    clean_fr['Comment'] = clean_fr['Comment'].apply(stemming_fr)

    from sklearn.feature_extraction.text import TfidfVectorizer

    all_text_fr = []
    all_text_fr = clean_fr['Comment'].values.astype('U')

    vects_fr = []
    all_x_fr = []
    vectorizer_fr = TfidfVectorizer(
        strip_accents='unicode', analyzer='word', ngram_range=(1, 3), norm='l2')
    vects_fr = vectorizer_fr.fit(all_text_en)
    all_x_fr = vects_en.transform(all_text_fr)

    from sklearn.cluster import AffinityPropagation
    from sklearn import metrics
    from sklearn.datasets import make_blobs

    X = all_x_fr

    afprop = AffinityPropagation(max_iter=300, damping=0.7)
    afprop.fit(X)
    cluster_centers_indices = afprop.cluster_centers_indices_
    X = X.todense()
    P = afprop.predict(X)

    import collections

    occurrences = collections.Counter(P)

    cluster_columns = ['Number of feedback', 'Representative comment']
    clusters = pd.DataFrame(columns=cluster_columns)

    cluster_rep = list(cluster_centers_indices)
    rep_comment = []
    for indice in cluster_rep:
        rep_comment.append(data_fr['Problem Details'][indice])

    clusters['Representative comment'] = rep_comment

    cluster_couple = sorted(occurrences.items())
    cluster_count = []

    cluster_count = [x[1] for x in cluster_couple]
    clusters['Number of feedback'] = cluster_count
    clusters = clusters.sort_values(by='Number of feedback', ascending=False)
    clusters = clusters.reset_index(drop=True)

    data_fr['group'] = P
    cluster_group = data_fr[['Problem Details', 'Date Entered', 'group']]

    rep_dict = {i: rep_comment[i] for i in range(0, len(rep_comment))}

    cluster_group['group'] = cluster_group['group'].map(rep_dict)

    clusters.to_csv('clusters_fr.csv', index=False)
    cluster_group.to_csv('comments_by_cluster_fr.csv', index=False)
