from airtable import Airtable
import pandas as pd
import pickle
from configparser import ConfigParser
from pymongo import MongoClient

#get api key from config file and get data from AirTabe
config = ConfigParser()
config.read('config/config.ini')
key = config.get('default', 'api_key')
base = config.get('default', 'base')
airtable = Airtable(base, 'Page feedback', api_key=key)
mango_db = config.get('default', 'mango_db')
dbase = config.get('default', 'dbase')
print('Accessed the keys')
record_list = airtable.get_all()
client = MongoClient(mango_db)
db = client[dbase]
problem = db.problem
print('Fetched the data')
#convert data to Pandas dataframe
data = pd.DataFrame([record['fields'] for record in record_list])
yes_no_db = pd.DataFrame(list(problem.find()))



print('Created the dataframes')
#define serialize function
def serialize(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)

#save data as a pickle
serialize(data, 'data/all_data.pickle')
serialize(yes_no_db, 'data/yes_no_db.pickle')

print('Saved dataframes as pickle files')
print('Process complete')
