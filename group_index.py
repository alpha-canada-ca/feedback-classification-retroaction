def groupindex():

    #import libraries
    from flask import Flask
    from flask import request
    from flask import app, render_template
    import requests
    import pandas as pd
    import pickle

    #define deserialize
    def deserialize(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    #import data as pickle
    data = deserialize('data/all_data.pickle')
    lang = request.args.get('lang', 'en')

    data = data[["Main section", 'URL', 'Status']]
    data = data[data.Status != 'Spam']
    data = data[data.Status != 'Ignore']
    data = data[data.Status != 'Duplicate']
    data = data.drop(columns=['Status'])
    data = data.dropna()
    data = data.reset_index(drop=True)
    data = data.drop_duplicates()
    data = data.reset_index(drop=True)
    data['URL'] = data['URL'].str.replace('/content/canadasite', 'www.canada.ca')
    data['URL'] = data['URL'].str.replace('www.canada.ca', 'https://www.canada.ca')
    data['URL'] = data['URL'].str.replace('https://https://', 'https://')
    sections = list(data['Main section'].unique())

    group_dict = {}

    group_dict = {k: g["URL"].tolist() for k,g in data.groupby("Main section")}

    groups = list(group_dict.keys())
    pages_list = list(group_dict.values())

    group_list = ["bygroup?group=" + group for group in groups]
    group_list_fr = ["bygroup?group=" + group + "&lang=fr" for group in groups]

    by_group_dict = {'Group':groups,'Pages':pages_list}

    by_group_table = pd.DataFrame(by_group_dict)

    by_group_table['group_list'] = group_list

    by_group_table_fr = by_group_table.copy()
    by_group_table_fr['group_list'] = group_list_fr


    if lang == 'en':
        return render_template("group_index_en.html", by_group_table = by_group_table, zip=zip)

    if lang == 'fr':
        return render_template("group_index_fr.html", by_group_table_fr = by_group_table_fr, zip=zip)



    #split feedback by What's wrongfully

    #get most meaningful word by what's wrong
