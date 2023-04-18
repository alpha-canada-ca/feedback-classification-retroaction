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
import time
import re
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import requests
from airtable import Airtable
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
import numpy as np
import nltk
from urllib.parse import urlparse, urlunparse

scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# use the credentials from client_secret.json to authorize Google Sheets API access
creds = ServiceAccountCredentials.from_json_keyfile_name(
    "client_secret.json", scope)
client = gspread.authorize(creds)

# open the specific Google Sheet
sheet = client.open("Tier 2 - Urls").sheet1

list_of_entries = sheet.get_all_records()


def remove_query_params(url):
    parsed = urlparse(url)
    new_url = urlunparse(
        (parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
    return new_url


counter = 2
# Iterate over each entry in list_of_entries
for entry in list_of_entries:
    # check if the URL starts with "www." and append "https" to it
    url = entry['URL']

    parsed_url = urlparse(url)
    if parsed_url.query:
        url = remove_query_params(url)
        sheet.update_cell(counter, 1, url)  # update cell at current counter
        print("REMOVED QUERY PARAMS.")

    # Send an HTTP request to the modified URL
    response = None
    try:
        response = requests.get(url)
    except:
        print("exception")

    if response is not None and 'adobedtm' in response.text:
        sheet.update_cell(counter, 2, "AEM")
    else:
        sheet.update_cell(counter, 2, "non-AEM")

    # Check if the page HTML contains "gc-pg-hlpfl" in any of the IDs
    if response is not None and "gc-pg-hlpfl" in response.text:
        print(f"{counter}, GOOD: {url} contains tool gc-pg-hlpfl in the HTML IDs.")

    # If the response code is 404, delete the current entry from the spreadsheet
    elif response is not None and response.status_code == 404:
        print(f"{counter}, NO RESPONSE: Deleting entry with URL {url}")
        sheet.delete_rows(counter)  # delete current row at counter
        counter -= 1  # decrement counter since row has been deleted
        time.sleep(1)

    elif response is not None and "gc-pg-hlpfl" not in response.text:
        print(f"{counter}, BAD: no tool, deleting entry with URL because it does not contain pg-hlpfl: {url}")
        sheet.delete_rows(counter)  # delete current row at counter
        counter -= 1  # decrement counter since row has been deleted
        print(response.text)
        time.sleep(1)

    # Print the response status code and URL
    elif response is not None:
        print(f"{counter}, Response: {response}, URL: {url}")

    else:
        print(f"{counter}, No response: {url}")

    # print(response.text)
    print("--------------")

    counter += 1
