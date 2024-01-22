#!/usr/bin/env python3
# import libraries
import os
from pyairtable import Api
import pickletools
import gzip
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from configparser import ConfigParser
import pickle
import warnings
import sys
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import requests
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import nltk

# import the OneHotEncoder

# import Vectorizer

# import models to train algorithm

# function to clean the word of any punctuation or special characters


def cleanPunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r"", sentence)
    cleaned = re.sub(r"[.|,|)|(|\|/]", r" ", cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n", " ")
    return cleaned


# function to remove non alpha chars
def remove_non_alpha_chars(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub("[^a-z A-Z]+", " ", word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent


def get_stemmed_text_en(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer_en.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


def get_stemmed_text_fr(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer_fr.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


def remove_stopwords(sentence):
    stop_words = set(stopwords.words("english"))
    words = sentence.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return " ".join(filtered_words)


# save to pickle files
def serialize(obj, file, protocol=-1):
    with gzip.open(file, "wb") as f:
        pickled = pickle.dumps(obj)
        optimized_pickle = pickletools.optimize(pickled)
        f.write(optimized_pickle)


# set the scope for the Google Sheets API
scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Use the credentials from client_secret.json in the config directory to authorize Google Sheets API access
# Use the credentials from client_secret.json located in the config directory at the root
creds = ServiceAccountCredentials.from_json_keyfile_name(
    "/config/client_secret.json", scope
)
client = gspread.authorize(creds)

# open the specific Google Sheet
sheet = client.open("Tier 1 - CronJob Models by URL ").sheet1

list_of_entries = sheet.get_all_records()

# create a dictionary mapping URLs to model names from the records
url_model_dict = {row["URL"]: row["MODEL"] for row in list_of_entries}

# read configuration settings from config.ini
config = ConfigParser()
config.read("config/config.ini")

# Get the API key from config.ini
api_key = config.get("default", "api_key")

# Initialize the API with your Airtable API key
api = Api(api_key)

base_ids_and_tables = [
    (config.get("default", "base"), config.get("default", "base_table_name")),
    (config.get("default", "base_health"), config.get("default", "base_table_name")),
    (config.get("default", "base_cra"), config.get("default", "base_table_name")),
    (config.get("default", "base_travel"), config.get("default", "base_table_name")),
    (config.get("default", "base_ircc"), config.get("default", "base_table_name")),
    # ... and so on for other bases
]

# Initialize list to store records
record_lists = []

# Fetch records for each base and table
for base_id, table_name in base_ids_and_tables:
    table = api.table(base_id, table_name)
    records = table.all()  # Fetch all records
    record_lists.append(records)

# concatenate all the feedback records into one DataFrame
data = pd.concat(
    [
        pd.DataFrame([record["fields"] for record in record_list])
        for record_list in record_lists
    ],
    ignore_index=True,
    sort=True,
)

# select only the relevant columns from the DataFrame
data = data[
    ["Comment", "Lookup_tags", "URL", "Model function", "Tags confirmed", "Lang"]
]

# map URLs to model names using the url_model_dict dictionary
data["model"] = data["URL"].map(url_model_dict)

# select only the English feedback
data_en = data[data["Lang"].str.contains("EN", na=False)]

data_en_topic = data_en.dropna()

data_en_topic = data_en_topic.drop_duplicates(subset="Comment")

data_en_topic = data_en_topic.reset_index(drop=True)

# create a new column "topics" that concatenates all tags associated with each feedback
data_en_topic["topics"] = data_en_topic["Lookup_tags"].apply(
    lambda x: ",".join(map(str, x))
)

# create a new column "Model function" that concatenates all model names associated with each feedback
data_en_topic["Model function"] = data_en_topic["Model function"].apply(
    lambda x: ",".join(map(str, x))
)

for i in range(len(data_en_topic)):
    # Check if Model function and model are not same
    if data_en_topic["Model function"][i] != data_en_topic["model"][i]:
        # Print mismatched models with URL
        print(
            "unmatching models: "
            + data_en_topic["Model function"][i]
            + " , "
            + data_en_topic["model"][i]
            + ", URL: "
            + data_en_topic["URL"][i]
        )

data_en_topic = data_en_topic.drop(columns=["Lookup_tags"])
data_en_topic = data_en_topic.drop(columns=["Model function"])
data_en_topic = data_en_topic.drop(columns=["Tags confirmed"])

# Get the different possible models
topics_en = list(data_en_topic["model"].unique())

sections_en = {
    topic: data_en_topic[data_en_topic["model"].str.contains(topic, na=False)]
    for topic in topics_en
}

# Reset index for each model
for cat in sections_en:
    sections_en[cat] = sections_en[cat].reset_index(drop=True)

# Convert English feedback to sparse matrix
cats_en = {}

for section in sections_en:
    # Instantiate a multi-label binarizer
    mlb = MultiLabelBinarizer()

    # Define a function to split topics and convert to set
    def split_topics(x):
        return set(x.split(","))

    # Apply the split_topics function to the topics column and transform using the multi-label binarizer
    mhv = mlb.fit_transform([split_topics(x) for x in sections_en[section]["topics"]])

    # Create a pandas DataFrame from the transformed data
    cats_en[section] = pd.DataFrame(mhv, columns=mlb.classes_)

    # Add the 'Feedback' column to the DataFrame
    cats_en[section].insert(0, "Feedback", sections_en[section]["Comment"])

# Instantiate an English stemmer
stemmer_en = SnowballStemmer("english")


# Apply pre-process functions to English feedback
for cat in cats_en:
    cats_en[cat]["Feedback"] = (
        cats_en[cat]["Feedback"]
        .str.lower()
        .apply(cleanPunc)
        .apply(remove_non_alpha_chars)
        .apply(get_stemmed_text_en)
        .apply(remove_stopwords)
    )

# Get all English text to build vectorizer as a dictionary (one key per model)
all_text_en = {}
for cat in cats_en:
    all_text_en[cat] = cats_en[cat]["Feedback"].values.astype("U")

# Peform vectorization for each English model (using dictionaries)
vectorizer_en = TfidfVectorizer(
    strip_accents="unicode", analyzer="word", ngram_range=(1, 3), norm="l2"
)
vects_en = {cat: vectorizer_en.fit(all_text_en[cat]) for cat in all_text_en}
all_x_en = {cat: vects_en[cat].transform(all_text_en[cat]) for cat in all_text_en}

# Split the English labels from the value - get all possible tags for each model
all_y_en = {}
categories_en = {}
for cat in all_x_en:
    all_y_en[cat] = cats_en[cat].drop(labels=["Feedback"], axis=1)
    categories_en[cat] = list(all_y_en[cat].columns.values)

# Create English model
model_en = {
    cat: {
        category: Pipeline(
            [
                (
                    "clf",
                    OneVsRestClassifier(
                        MultinomialNB(alpha=0.3, fit_prior=True, class_prior=None)
                    ),
                )
            ]
        ).fit(all_x_en[cat], cats_en[cat][category])
        for category in categories_en[cat]
    }
    for cat in categories_en
}


# split dataframe for French comments - same comments as above for each line
data_fr = data[data["Lang"].str.contains("FR", na=False)]
data_fr_topic = data_fr.dropna()
data_fr_topic = data_fr_topic.drop_duplicates(subset="Comment")
data_fr_topic = data_fr_topic.reset_index(drop=True)
data_fr_topic["topics"] = [",".join(map(str, l)) for l in data_fr_topic["Lookup_tags"]]
data_fr_topic["Model function"] = [
    ",".join(map(str, l)) for l in data_fr_topic["Model function"]
]

# Check if models are the same
for i in range(len(data_fr_topic)):
    if data_fr_topic["Model function"][i] != data_fr_topic["model"][i]:
        print("Found unmatching models - FR")


data_fr_topic = data_fr_topic.drop(columns=["Lookup_tags"])
data_fr_topic = data_fr_topic.drop(columns=["Model function"])
data_fr_topic = data_fr_topic.drop(columns=["Tags confirmed"])


topics_fr = list(data_fr_topic["model"].unique())


sections_fr = {
    topic: data_fr_topic[data_fr_topic["model"].str.contains(topic, na=False)]
    for topic in topics_fr
}

# reset index for each model
for cat in sections_fr:
    sections_fr[cat] = sections_fr[cat].reset_index(drop=True)


# convert French feedback to sparse matrix
cats_fr = {}
for section in sections_fr:
    mlb = MultiLabelBinarizer()
    mhv = mlb.fit_transform(
        sections_fr[section]["topics"].apply(lambda x: set(x.split(",")))
    )
    cats_fr[section] = pd.DataFrame(mhv, columns=mlb.classes_)
    cats_fr[section].insert(0, "Feedback", sections_fr[section]["Comment"])


# pre-process feedback for NLP

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# function to stem French feedback
stemmer_fr = SnowballStemmer("french")
# apply processing function to French feedback
for cat in cats_fr:
    cats_fr[cat]["Feedback"] = cats_fr[cat]["Feedback"].str.lower()
    cats_fr[cat]["Feedback"] = cats_fr[cat]["Feedback"].apply(cleanPunc)
    cats_fr[cat]["Feedback"] = cats_fr[cat]["Feedback"].apply(remove_non_alpha_chars)
    cats_fr[cat]["Feedback"] = cats_fr[cat]["Feedback"].apply(get_stemmed_text_fr)


# get all French text to build vectorizer as a dictionary (one key per model)
all_text_fr = {}
for cat in cats_fr:
    all_text_fr[cat] = cats_fr[cat]["Feedback"].values.astype("U")

# peform vectoriztion for each French model (using dictionaries)
all_x_fr = {}
vects_fr = {}
for cat in all_text_fr:
    vectorizer_fr = TfidfVectorizer(
        strip_accents="unicode", analyzer="word", ngram_range=(1, 3), norm="l2"
    )
    vects_fr[cat] = vectorizer_fr.fit(all_text_fr[cat])
    all_x_fr[cat] = vects_fr[cat].transform(all_text_fr[cat])

# split the French labels from the value - get all possible tags for each model
all_y_fr = {}
categories_fr = {}
for cat in all_x_fr:
    all_y_fr[cat] = cats_fr[cat].drop(labels=["Feedback"], axis=1)
    categories_fr[cat] = list(all_y_fr[cat].columns.values)


# create French model
model_fr = {}
for cat in categories_fr:
    model_fr[cat] = {}
    for category in categories_fr[cat]:
        NB_pipeline = Pipeline(
            [
                (
                    "clf",
                    OneVsRestClassifier(
                        MultinomialNB(alpha=0.3, fit_prior=True, class_prior=None)
                    ),
                ),
            ]
        )
        NB_pipeline.fit(all_x_fr[cat], cats_fr[cat][category])
        model_fr[cat][category] = NB_pipeline


serialize(categories_en, "data/categories_en.pickle")
serialize(categories_fr, "data/categories_fr.pickle")
serialize(vects_en, "data/vectorizer_en.pickle")
serialize(vects_fr, "data/vectorizer_fr.pickle")
serialize(model_en, "data/model_en.pickle")
serialize(model_fr, "data/model_fr.pickle")

print("Processing feedback process complete - PF")
