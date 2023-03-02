from airtable import Airtable
import pandas as pd
import pickle
import gzip
import pickletools
from configparser import ConfigParser
from pymongo import MongoClient
from bson import ObjectId


# get api key from config file and get data from AirTabe
config = ConfigParser()
config.read('config/config.ini')
key = config.get('default', 'api_key')
dbConnectionString = config.get('default', 'mongo_db')
base = config.get('default', 'base')


client = MongoClient(dbConnectionString)
print("Connected to DB.")
problem = client.pagesuccess.problem

print('Fetched the problem collection.')

print('Fetched Airtable Tags')

# define deserialize


def deserialize(file):
    with gzip.open(file, 'rb') as f:
        p = pickle.Unpickler(f)
        return p.load()


# import data as pickle
data = deserialize('data/all_data.pickle')
data = data[["Lookup_tags", "Unique ID"]]
data = data.dropna()
data = data.reset_index(drop=True)
data['Lookup_tags'] = [', '.join(map(str, l)) for l in data['Lookup_tags']]
problemCount = problem.count_documents({})
print(problemCount)

counter = 0
for index in range(len(data)):
    if "-" not in data['Unique ID'][index] and len(data['Unique ID'][index]) == 24:
        if (problem.find_one({"_id": ObjectId(data['Unique ID'][index])})):
            mutualEntry = problem.find_one(
                {"_id": ObjectId(data['Unique ID'][index])})
            mutualEntry = ', '.join(mutualEntry['tags'])
            if mutualEntry != (data['Lookup_tags'][index]):
                print("updated index: " + str(index))
                counter += 1
                problem.find_one_and_update({"_id": ObjectId((data['Unique ID'][index]))}, {
                                            "$set": {"tags": [(data['Lookup_tags'][index])]}})

print("Entries updated: " + str(counter))
