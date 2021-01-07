from airtable import Airtable
import pandas as pd
import pickle
from configparser import ConfigParser
from pymongo import MongoClient
from bson import ObjectId

print("test test test test test test")
#get api key from config file and get data from AirTabe
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

def deserialize(file):
    with open(file, 'rb') as f:
        return pickle.load(f)

#import data as pickle
data = deserialize('data/all_data.pickle')
data = data[["Lookup_tags", "Unique ID"]]
data = data.dropna()
data = data.reset_index(drop=True)
data['Lookup_tags'] = [', '.join(map(str, l)) for l in data['Lookup_tags']]
problemCount = problem.count_documents({})
print(problemCount)

counter = 0
for index in range(len(data)):
    if "-" not in data['Unique ID'][index]:
        if(problem.find_one({"_id" : ObjectId(data['Unique ID'][index])})):
            x = problem.find_one({"_id" : ObjectId(data['Unique ID'][index])})
            x = ', '.join(x['tags'])
            if x != (data['Lookup_tags'][index]):
                print("updated index: " + str(index))
                problem.find_one_and_update({"_id" : ObjectId((data['Unique ID'][index]))},{"$set":{"tags": [(data['Lookup_tags'][index])]}})
                print()
            else:
                print("already updated. " + str(index))
        else:
            print("didnt find")
    else:
        print("contains '-'. " + str(index))
