def index():

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

    data['URL'] = data['URL'].str.replace('/content/canadasite', 'www.canada.ca')
    data['URL'] = data['URL'].str.replace('www.canada.ca', 'https://www.canada.ca')

    data = data[["URL", 'Lookup_page_title', 'Status']]
    data = data[data.Status != 'Spam']
    data = data[data.Status != 'Ignore']
    data = data[data.Status != 'Duplicate']
    data = data.drop(columns=['Status'])
    data = data.dropna()
    data = data.reset_index(drop=True)
    data['Lookup_page_title'] = [','.join(map(str, l)) for l in data['Lookup_page_title']]
    data = data.drop_duplicates()
    data = data.drop(data[data.URL == 'TEST.html'].index)
    data = data.reset_index(drop=True)
    data = data[['Lookup_page_title', 'URL']]
    titles = list(data['Lookup_page_title'])
    urls = list(data['URL'])

    url_list = ["bypage?page=" + url for url in urls]
    url_list_fr = ["bypage?page=" + url + "&lang=fr" for url in urls]
    title_list = [title + "</a>" for title in titles]
    title_chart = [a + b for a, b in zip(url_list, title_list)]
    by_page_dict = {'Page':titles,'URL':urls}
    by_page_table = pd.DataFrame(by_page_dict)
    by_page_table['url_list'] = url_list
    by_page_table_fr = by_page_table.copy()
    by_page_table_fr['url_list'] = url_list_fr

    if lang == 'en':
        return render_template("index_en.html", by_page_table = by_page_table, zip=zip)

    if lang == 'fr':
        return render_template("index_fr.html", by_page_table_fr = by_page_table_fr, zip=zip)



    #split feedback by What's wrongfully

    #get most meaningful word by what's wrong
