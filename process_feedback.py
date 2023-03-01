#import libraries
import requests
from airtable import Airtable
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings
import pickle
import gzip, pickletools
from configparser import ConfigParser
#import the OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
#import Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

#import models to train algorithm
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier

#function to clean the word of any punctuation or special characters
def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

#function to convert to lowercase
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

def stemming_en(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer_en.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

def stemming_fr(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer_fr.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)

#save to pickle files
def serialize(obj, file, protocol=-1):
    with gzip.open(file, "wb") as f:
        pickled = pickle.dumps(obj)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)


# use creds to create a client to interact with the Google Drive API
scope = [
'https://www.googleapis.com/auth/spreadsheets',
'https://www.googleapis.com/auth/drive'
]
creds = ServiceAccountCredentials.from_json_keyfile_name('client_secret.json', scope)
client = gspread.authorize(creds)

# Find a spreadsheet by name and open the first sheet
sheet = client.open("Tier 1 - CronJob Models by URL ").sheet1

# Add URLs to dictionary as key, with model as value
list_of_entries = sheet.get_all_records()
url_model_dict = {row['URL']: row['MODEL'] for row in list_of_entries}

#get api key from config file and get data from Airtable - base and api key are in a hidden config folder
config = ConfigParser()
config.read('config/config.ini')
key = config.get('default', 'api_key')
bases = [
    config.get('default', 'base'),
    config.get('default', 'base_health'),
    config.get('default', 'base_cra'),
    config.get('default', 'base_travel'),
    config.get('default', 'base_ircc')
]

airtables = [Airtable(base, 'Page feedback', api_key=key) for base in bases]

record_lists = [airtable.get_all() for airtable in airtables]
print('Fetched the data - PF')

#convert data to Pandas dataframe

#If you want to experiment with this script without setting up an AirTable, you can do so by loading the tagged_feedback.csv file from the repo and convert it to a Pandas dataframe, with this line of code: "data = pd.read_csv('tagged_feedback.csv')".

data = pd.concat([pd.DataFrame([record['fields'] for record in record_list]) for record_list in record_lists], ignore_index=True, sort=True)

# Remove unneccessary columns
data = data[['Comment', 'Lookup_tags', 'URL', 'Model function', 'Tags confirmed', 'Lang']]

# Add model column


# Add appropriate models to model column by retreiving URL in dict
data['model'] = data['URL'].map(url_model_dict)

#split dataframe for English comments
data_en = data[data['Lang'].str.contains("EN", na=False)]


#keep only relevant columns from the dataframe

#remove all rows thave have null content - this, in effect, removes comments for which the tags haven't been confirmed by a human
data_en_topic = data_en.dropna()
data_en_topic = data_en_topic.drop_duplicates(subset ="Comment")
data_en_topic = data_en_topic.reset_index(drop=True)

#converts the tags to a string (instead of a list) - needed for further processing - and puts it in a new column

data_en_topic['topics'] = data_en_topic['Lookup_tags'].apply(lambda x: ','.join(map(str, x)))
data_en_topic['Model function'] = data_en_topic['Model function'].apply(lambda x: ','.join(map(str, x)))
# Check if models are the same
for i in range(len(data_en_topic)):
    if data_en_topic['Model function'][i] != data_en_topic['model'][i]:
        print('unmatching models: ' + data_en_topic['Model function'][i] + " , " + data_en_topic['model'][i] + ", URL: " + data_en_topic['URL'][i])


# Remove unneccessary columns
data_en_topic = data_en_topic.drop(columns=['Lookup_tags'])
data_en_topic = data_en_topic.drop(columns=['Model function'])
data_en_topic = data_en_topic.drop(columns=['Tags confirmed'])

#get the different possible models
topics_en = list(data_en_topic['model'].unique())

#creates a dictionary (key = model, value = tagged feedback for that model)
sections_en = {topic : data_en_topic[data_en_topic['model'].str.contains(topic, na=False)] for topic in topics_en}

#reset index for each model
for cat in sections_en:
    sections_en[cat] = sections_en[cat].reset_index(drop=True)


#convert English feedback to sparse matrix
cats_en = {}
for section in sections_en:
    mlb = MultiLabelBinarizer()
    # Define a function to split topics and convert to set
    def split_topics(x):
        return set(x.split(','))
    # Apply the split_topics function to the topics column and transform using the multi-label binarizer
    mhv = mlb.fit_transform([split_topics(x) for x in sections_en[section]['topics']])
    # Create a pandas DataFrame from the transformed data
    cats_en[section] = pd.DataFrame(mhv, columns=mlb.classes_)
    # Add the 'Feedback' column to the DataFrame
    cats_en[section].insert(0, 'Feedback', sections_en[section]['Comment'])



stemmer_en = SnowballStemmer("english")
#apply pre-process functions to English
for cat in cats_en:
    cats_en[cat]['Feedback'] = (cats_en[cat]['Feedback'].str.lower().apply(cleanPunc).apply(keepAlpha).apply(stemming_en).apply(remove_stopwords))

# Get all English text to build vectorizer as a dictionary (one key per model)
all_text_en = {}
for cat in cats_en:
    all_text_en[cat] = cats_en[cat]['Feedback'].values.astype('U')

# Peform vectorization for each English model (using dictionaries)
vectorizer_en = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
vects_en = {cat: vectorizer_en.fit(all_text_en[cat]) for cat in all_text_en}
all_x_en = {cat: vects_en[cat].transform(all_text_en[cat]) for cat in all_text_en}

# Split the English labels from the value - get all possible tags for each model
all_y_en = {}
categories_en = {}
for cat in all_x_en:
    all_y_en[cat] = cats_en[cat].drop(labels=['Feedback'], axis=1)
    categories_en[cat] = list(all_y_en[cat].columns.values)

# Create English model
model_en = {cat: {category: Pipeline([('clf', OneVsRestClassifier(MultinomialNB(alpha=0.3, fit_prior=True, class_prior=None)))])
                  .fit(all_x_en[cat], cats_en[cat][category])
                  for category in categories_en[cat]}
            for cat in categories_en}


#split dataframe for French comments - same comments as above for each line
data_fr = data[data['Lang'].str.contains("FR", na=False)]
data_fr_topic = data_fr.dropna()
data_fr_topic = data_fr_topic.drop_duplicates(subset ="Comment")
data_fr_topic = data_fr_topic.reset_index(drop=True)
data_fr_topic['topics'] = [','.join(map(str, l)) for l in data_fr_topic['Lookup_tags']]
data_fr_topic['Model function'] = [','.join(map(str, l)) for l in data_fr_topic['Model function']]

# Check if models are the same
for i in range(len(data_fr_topic)):
    if data_fr_topic['Model function'][i] != data_fr_topic['model'][i]:
        print("Found unmatching models - FR")


data_fr_topic = data_fr_topic.drop(columns=['Lookup_tags'])
data_fr_topic = data_fr_topic.drop(columns=['Model function'])
data_fr_topic = data_fr_topic.drop(columns=['Tags confirmed'])


topics_fr = list(data_fr_topic['model'].unique())


sections_fr = {topic : data_fr_topic[data_fr_topic['model'].str.contains(topic, na=False)] for topic in topics_fr}

#reset index for each model
for cat in sections_fr:
    sections_fr[cat] = sections_fr[cat].reset_index(drop=True)


#convert French feedback to sparse matrix
cats_fr = {}
for section in sections_fr:
    mlb = MultiLabelBinarizer()
    mhv= mlb.fit_transform(sections_fr[section]['topics'].apply(lambda x: set(x.split(','))))
    cats_fr[section] = pd.DataFrame(mhv,columns=mlb.classes_)
    cats_fr[section].insert(0, 'Feedback', sections_fr[section]['Comment'])



#pre-process feedback for NLP

if not sys.warnoptions:
    warnings.simplefilter("ignore")

#function to stem French feedback
stemmer_fr = SnowballStemmer("french")
#apply processing function to French feedback
for cat in cats_fr:
    cats_fr[cat]['Feedback'] = cats_fr[cat]['Feedback'].str.lower()
    cats_fr[cat]['Feedback'] = cats_fr[cat]['Feedback'].apply(cleanPunc)
    cats_fr[cat]['Feedback'] = cats_fr[cat]['Feedback'].apply(keepAlpha)
    cats_fr[cat]['Feedback'] = cats_fr[cat]['Feedback'].apply(stemming_fr)







#get all French text to build vectorizer as a dictionary (one key per model)
all_text_fr = {}
for cat in cats_fr:
    all_text_fr[cat] = cats_fr[cat]['Feedback'].values.astype('U')

#peform vectoriztion for each French model (using dictionaries)
all_x_fr = {}
vects_fr = {}
for cat in all_text_fr:
    vectorizer_fr = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
    vects_fr[cat] = vectorizer_fr.fit(all_text_fr[cat])
    all_x_fr[cat] = vects_fr[cat].transform(all_text_fr[cat])

#split the French labels from the value - get all possible tags for each model
all_y_fr = {}
categories_fr = {}
for cat in all_x_fr:
    all_y_fr[cat] = cats_fr[cat].drop(labels = ['Feedback'], axis=1)
    categories_fr[cat] = list(all_y_fr[cat].columns.values)







#create French model
model_fr = {}
for cat in categories_fr:
    model_fr[cat] = {}
    for category in categories_fr[cat]:
        NB_pipeline = Pipeline([
            ('clf', OneVsRestClassifier(MultinomialNB(alpha=0.3, fit_prior=True, class_prior=None))),
            ])
        NB_pipeline.fit(all_x_fr[cat], cats_fr[cat][category])
        model_fr[cat][category] = NB_pipeline




serialize(categories_en, 'data/categories_en.pickle')
serialize(categories_fr, 'data/categories_fr.pickle')
serialize(vects_en, 'data/vectorizer_en.pickle')
serialize(vects_fr, 'data/vectorizer_fr.pickle')
serialize(model_en, 'data/model_en.pickle')
serialize(model_fr, 'data/model_fr.pickle')

print('Processing feedback process complete - PF')
