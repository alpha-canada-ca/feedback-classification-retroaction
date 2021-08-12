#import libraries
import requests
from airtable import Airtable
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import sys
import warnings
import pickle
import gzip, pickletools
from configparser import ConfigParser

#get api key from config file and get data from Airtable - base and api key are in a hidden config folder
config = ConfigParser()
config.read('config/config.ini')
key = config.get('default', 'api_key')
base = config.get('default', 'base')
base_health = config.get('default', 'base_health')
base_cra= config.get('default', 'base_cra')
base_travel= config.get('default', 'base_travel')
print('Accessed the keys')

airtable_main = Airtable(base, 'Page feedback', api_key=key)
airtable_health = Airtable(base_health, 'Page feedback', api_key=key)
airtable_cra = Airtable(base_cra, 'Page feedback', api_key=key)
airtable_travel= Airtable(base_travel, 'Page feedback', api_key=key)

record_list_main = airtable_main.get_all()
record_list_health = airtable_health.get_all()
record_list_cra = airtable_cra.get_all()
record_list_travel = airtable_travel.get_all()
print('Fetched the data')

#convert data to Pandas dataframe
data_main = pd.DataFrame([record['fields'] for record in record_list_main])
data_health = pd.DataFrame([record['fields'] for record in record_list_health])
data_cra = pd.DataFrame([record['fields'] for record in record_list_cra])
data_travel = pd.DataFrame([record['fields'] for record in record_list_travel])



#If you want to experiment with this script without setting up an AirTable, you can do so by loading the tagged_feedback.csv file from the repo and convert it to a Pandas dataframe, with this line of code: "data = pd.read_csv('tagged_feedback.csv')".


data_2 = data_main.append(data_health, ignore_index=True, sort=True)

data_1 = data_2.append(data_cra, ignore_index=True, sort=True)

data = data_1.append(data_travel, ignore_index=True, sort=True)


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
url_model_dict = {}
for i in range(len(list_of_entries)):
    if list_of_entries[i]['URL'] not in url_model_dict.keys():
        url_model_dict[list_of_entries[i]['URL']] = list_of_entries[i]['MODEL']


# Remove unneccessary columns
data = data[['Comment', 'Lookup_tags', 'URL', 'Model function', 'Tags confirmed', 'Lang']]

# Add model column
data['model'] = ""

# Add appropriate models to model column by retreiving URL in dict
for i in range(len(data)):
    data['model'][i] = url_model_dict.get(data['URL'][i])

#split dataframe for English comments
data_en = data[data['Lang'].str.contains("EN", na=False)]

#keep only relevant columns from the dataframe

#remove all rows thave have null content - this, in effect, removes comments for which the tags haven't been confirmed by a human
data_en_topic = data_en.dropna()
data_en_topic = data_en_topic.drop_duplicates(subset ="Comment")
data_en_topic = data_en_topic.reset_index(drop=True)

#converts the tags to a string (instead of a list) - needed for further processing - and puts it in a new column
data_en_topic['topics'] = [','.join(map(str, l)) for l in data_en_topic['Lookup_tags']]
data_en_topic['Model function']  = [','.join(map(str, l)) for l in data_en_topic['Model function']] ####### might not be needed

# Check if models are the same
for i in range(len(data_en_topic)):
    if data_en_topic['Model function'][i] != data_en_topic['model'][i]:
        print("Found unmatching models - EN")
        # print(data_en_topic['Model function'][i]) 
        # print(data_en_topic['model'][i])
        # print(i)
        # print()

# Remove unneccessary columns
data_en_topic = data_en_topic.drop(columns=['Lookup_tags'])
data_en_topic = data_en_topic.drop(columns=['Model function'])
data_en_topic = data_en_topic.drop(columns=['Tags confirmed'])

for i in range(len(data_en_topic)):
    data_en_topic['model'][i] = data_en_topic['model'][i].lower()

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
        # print(data_en_topic['Model function'][i]) 
        # print(data_en_topic['model'][i])
        # print(i)
        # print()

data_fr_topic = data_fr_topic.drop(columns=['Lookup_tags'])
data_fr_topic = data_fr_topic.drop(columns=['Model function'])
data_fr_topic = data_fr_topic.drop(columns=['Tags confirmed'])

for i in range(len(data_fr_topic)):
    data_fr_topic['model'][i] = data_fr_topic['model'][i].lower()

#get the different possible models

topics_en = list(data_en_topic['model'].unique())

topics_fr = list(data_fr_topic['model'].unique())

#creates a dictionary (key = model, value = tagged feedback for that model)
sections_en = {topic : data_en_topic[data_en_topic['model'].str.contains(topic, na=False)] for topic in topics_en}
sections_fr = {topic : data_fr_topic[data_fr_topic['model'].str.contains(topic, na=False)] for topic in topics_fr}

#reset index for each model
for cat in sections_en:
    sections_en[cat] = sections_en[cat].reset_index(drop=True)

for cat in sections_fr:
    sections_fr[cat] = sections_fr[cat].reset_index(drop=True)

#import the OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

#convert English feedback to sparse matrix
cats_en = {}
for section in sections_en:
    mlb = MultiLabelBinarizer()
    mhv= mlb.fit_transform(sections_en[section]['topics'].apply(lambda x: set(x.split(','))))
    cats_en[section] = pd.DataFrame(mhv,columns=mlb.classes_)
    cats_en[section].insert(0, 'Feedback', sections_en[section]['Comment'])

#convert French feedback to sparse matrix
cats_fr = {}
for section in sections_en:
    mlb = MultiLabelBinarizer()
    mhv= mlb.fit_transform(sections_fr[section]['topics'].apply(lambda x: set(x.split(','))))
    cats_fr[section] = pd.DataFrame(mhv,columns=mlb.classes_)
    cats_fr[section].insert(0, 'Feedback', sections_fr[section]['Comment'])



#pre-process feedback for NLP

if not sys.warnoptions:
    warnings.simplefilter("ignore")

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


#function to stem feedbck (English)
stemmer_en = SnowballStemmer("english")
def stemming_en(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer_en.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

#apply pre-process functions to English
for cat in cats_en:
    cats_en[cat]['Feedback'] = cats_en[cat]['Feedback'].str.lower()
    cats_en[cat]['Feedback'] = cats_en[cat]['Feedback'].apply(cleanPunc)
    cats_en[cat]['Feedback'] = cats_en[cat]['Feedback'].apply(keepAlpha)
    cats_en[cat]['Feedback'] = cats_en[cat]['Feedback'].apply(stemming_en)


#function to stem French feedback
stemmer_fr = SnowballStemmer("french")
def stemming_fr(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer_fr.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

#apply processing function to French feedback
for cat in cats_fr:
    cats_fr[cat]['Feedback'] = cats_fr[cat]['Feedback'].str.lower()
    cats_fr[cat]['Feedback'] = cats_fr[cat]['Feedback'].apply(cleanPunc)
    cats_fr[cat]['Feedback'] = cats_fr[cat]['Feedback'].apply(keepAlpha)
    cats_fr[cat]['Feedback'] = cats_fr[cat]['Feedback'].apply(stemming_fr)



#import Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


#get all English text to build vectorizer as a dictionary (one key per model)
all_text_en = {}
for cat in cats_en:
    all_text_en[cat] = cats_en[cat]['Feedback'].values.astype('U')

#peform vectoriztion for each English model (using dictionaries)
vects_en = {}
all_x_en = {}
for cat in all_text_en:
    vectorizer_en = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
    vects_en[cat] = vectorizer_en.fit(all_text_en[cat])
    all_x_en[cat] = vects_en[cat].transform(all_text_en[cat])

#split the English labels from the value - get all possible tags for each model
all_y_en = {}
categories_en = {}
for cat in all_x_en:
    all_y_en[cat] = cats_en[cat].drop(labels = ['Feedback'], axis=1)
    categories_en[cat] = list(all_y_en[cat].columns.values)


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




#import models to train algorithm
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier


#create English model
model_en = {}
for cat in categories_en:
    model_en[cat] = {}
    for category in categories_en[cat]:
        NB_pipeline = Pipeline([
            ('clf', OneVsRestClassifier(MultinomialNB(alpha=0.3, fit_prior=True, class_prior=None))),
            ])
        NB_pipeline.fit(all_x_en[cat], cats_en[cat][category])
        model_en[cat][category] = NB_pipeline

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


#save to pickle files
def serialize(obj, file, protocol=-1):
    with gzip.open(file, "wb") as f:
        pickled = pickle.dumps(obj)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)


serialize(categories_en, 'data/categories_en.pickle')
serialize(categories_fr, 'data/categories_fr.pickle')
serialize(vects_en, 'data/vectorizer_en.pickle')
serialize(vects_fr, 'data/vectorizer_fr.pickle')
serialize(model_en, 'data/model_en.pickle')
serialize(model_fr, 'data/model_fr.pickle')

print('Processing feedback process complete')
