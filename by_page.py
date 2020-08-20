
from flask import Flask
from flask import request
from flask import app, render_template


app = Flask(__name__)

@app.route('/bypage')

def bypage():

    #import libraries
    import requests
    import pandas as pd
    from nltk.corpus import stopwords
    import nltk
    from nltk.stem.snowball import SnowballStemmer
    import re
    import sys
    import warnings
    import pickle

    #define deserialize
    def deserialize(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    #import data as pickle
    data = deserialize('data/all_data.pickle')

    #get variables
    lang = request.args.get('lang')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    page = request.args.get('page')

    data = data[data['Date'] <= end_date]
    data = data[data['Date'] >= start_date]

    if lang == 'en':
        #split dataframe for English comments
        data_en = data[data['Lang'].str.contains("EN", na=False)]

        #keep only relevant columns from the dataframe
        data_en_pages = data_en[["Comment", "Lookup_page_title", "What's wrong", "Lookup_tags", 'Tags confirmed']]
        data_en_pages["What's wrong"].fillna(False, inplace=True)

        #remove all rows thave have null content
        data_en_pages = data_en_pages.dropna()

        #convert the page title to a string
        data_en_pages['Page_title'] = [','.join(map(str, l)) for l in data_en_pages['Lookup_page_title']]

        #converts the tags to a string (instead of a list) - needed for further processing - and puts it in a new column
        data_en_pages['tags'] = [','.join(map(str, l)) for l in data_en_pages['Lookup_tags']]

        #remove the Lookup_tags column (it's not needed anymore)
        data_en_pages = data_en_pages.drop(columns=['Lookup_tags'])

        #remove the Lookup_page_title column (it's not needed anymore)
        data_en_pages = data_en_pages.drop(columns=['Lookup_page_title'])

        #remove the Lookup_page_title column (it's not needed anymore)
        data_en_pages = data_en_pages.drop(columns=['Tags confirmed'])

        #resets the index for each row - needed for further processing
        data_en_pages = data_en_pages.reset_index(drop=True)

        #split dataframe for French comments - same comments as above for each line

        #get data for specific page
        page_data_en = data_en_pages.loc[data_en_pages['Page_title'] == page]

        page_data_en = page_data_en.reset_index(drop=True)

        #split tags and expand
        tags_en = page_data_en["tags"].str.split(",", n = 3, expand = True)
        page_data_en = page_data_en.join(tags_en)
        page_data_en = page_data_en.drop(columns=['tags'])

        #get most frequent words for all of page
        #get all words in a list
        word_list_en = page_data_en["Comment"].tolist()
        word_list_en = [str(i) for i in word_list_en]
        all_words_en = ' '.join([str(elem) for elem in word_list_en])

        #tokenize words
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        tokens_en = tokenizer.tokenize(all_words_en)
        words_en = []
        for word in tokens_en:
                words_en.append(word.lower())

        #remove English stop words to get most frequent words
        nltk.download('stopwords')
        sw = nltk.corpus.stopwords.words('english')
        sw.append('covid')
        sw.append('19')
        words_ns_en = []
        for word in words_en:
                if word not in sw:
                    words_ns_en.append(word)
        #get most common words
        from nltk import FreqDist
        fdist1 = FreqDist(words_ns_en)
        most_common = fdist1.most_common(15)


        #count the number for each tag
        tag_count = tags_en.apply(pd.Series.value_counts)
        tag_count = tag_count.fillna(0)
        tag_count = tag_count.astype(int)
        tag_count = tag_count[0] + tag_count[1]
        tag_count = tag_count.sort_values(ascending = False)
        by_tag = tag_count.to_frame()
        by_tag = by_tag.sort_index(axis=0, level=None, ascending=True)
        by_tag.columns = ['Feedback count']


        #split feedback by tag

        tag_dict = {}

        if 2 in page_data_en.columns:
            for tag, topic_df_en in page_data_en.groupby(2):
                tag_dict[tag] = ' '.join(topic_df_en['Comment'].tolist())

        if 1 in page_data_en.columns:
            for tag, topic_df_en in page_data_en.groupby(1):
                tag_dict[tag] = ' '.join(topic_df_en['Comment'].tolist())

        for tag, topic_df_en in page_data_en.groupby(0):
            tag_dict[tag] = ' '.join(topic_df_en['Comment'].tolist())


        #tokenize

        tokenizer = nltk.RegexpTokenizer(r"\w+")

        for value in tag_dict:
            tag_dict[value] = tokenizer.tokenize(tag_dict[value])

        #get most meaningful words by tags
        topic_list_en = []
        for keys in tag_dict.keys():
            topic_list_en.append(keys)

        topic_words_en = []
        for values in tag_dict.values():
            topic_words_en.append(values)

        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        from nltk.corpus import stopwords

        topic_words_en = [[word.lower() for word in value] for value in topic_words_en]
        topic_words_en = [[lemmatizer.lemmatize(word) for word in value] for value in topic_words_en]
        topic_words_en = [[word for word in value if word not in sw] for value in topic_words_en]

        from gensim.corpora.dictionary import Dictionary

        dictionary_en = Dictionary(topic_words_en)

        corpus_en = [dictionary_en.doc2bow(tag) for tag in topic_words_en]

        from gensim.models.tfidfmodel import TfidfModel

        tfidf_en = TfidfModel(corpus_en)

        tfidf_weights_en = [sorted(tfidf_en[doc], key=lambda w: w[1], reverse=True) for doc in corpus_en]

        weighted_words_en = [[(dictionary_en.get(id), weight) for id, weight in ar] for ar in tfidf_weights_en]

        imp_words_en = pd.DataFrame({'Tag': topic_list_en, 'EN_words':  weighted_words_en})

        imp_words_en = imp_words_en.sort_values(by = 'Tag')

        imp_words_en = imp_words_en.reset_index(drop=True)

        imp_words_en['EN_words'] = imp_words_en ['EN_words'].apply(lambda x: list(x))

        imp_words_en['EN_words'] = imp_words_en['EN_words'].apply(lambda x: x[:9])

        imp_words_en['EN_words'] = imp_words_en['EN_words'].apply(lambda x: [y[0] for y in x])

        imp_words_en = imp_words_en.set_index('Tag', drop=True, append=False)

        by_tag['Significant words'] = imp_words_en['EN_words']


        by_tag = by_tag.sort_values(by = 'Feedback count', ascending=False)

        by_tag = by_tag.reset_index()

        column_names = ['Tag count', 'Tag', 'Significant words']

        by_tag['Significant words'] = by_tag['Significant words'].apply(lambda x: ', '.join(x))

        by_tag = by_tag[['Feedback count', 'index', 'Significant words']]

        return render_template("info_by_page_en.html", most_common = most_common, column_names = column_names, row_data = list(by_tag.values.tolist()), zip = zip, page = page)



    #process to follow if English
    else:

        data_fr = data[data['Lang'].str.contains("FR", na=False)]
        data_fr_pages= data_fr[["Comment", "Lookup_page_title", "What's wrong", "Lookup_tags", 'Tags confirmed']]
        data_fr_pages["What's wrong"].fillna(False, inplace=True)
        data_fr_pages = data_fr_pages.dropna()
        data_fr_pages['Page_title'] = [','.join(map(str, l)) for l in data_fr_pages['Lookup_page_title']]
        data_fr_pages['tags'] = [','.join(map(str, l)) for l in data_fr_pages['Lookup_tags']]
        data_fr_pages = data_fr_pages.drop(columns=['Lookup_tags'])
        data_fr_pages = data_fr_pages.drop(columns=['Lookup_page_title'])
        data_fr_pages = data_fr_pages.drop(columns=['Tags confirmed'])

        page_data_fr = data_fr_pages.loc[data_fr_pages['Page_title'] == page]

        page_data_fr = page_data_fr.reset_index(drop=True)

        #split tags and expand
        tags_fr = page_data_fr["tags"].str.split(",", n = 3, expand = True)
        page_data_fr = page_data_fr.join(tags_fr)
        page_data_fr = page_data_fr.drop(columns=['tags'])

        #get most frequent words for all of page
        #get all words in a list
        word_list_fr = page_data_fr["Comment"].tolist()
        word_list_fr = [str(i) for i in word_list_fr]
        all_words_fr = ' '.join([str(elem) for elem in word_list_fr])

        #tokenize words
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        tokens_fr = tokenizer.tokenize(all_words_fr)
        words_fr = []
        for word in tokens_fr:
                words_fr.append(word.lower())

        #remove English stop words to get most frequent words
        nltk.download('stopwords')
        sw = nltk.corpus.stopwords.words('french')
        sw.append('covid')
        sw.append('19')
        words_ns_fr = []
        for word in words_fr:
                if word not in sw:
                    words_ns_fr.append(word)
        #get most common words
        from nltk import FreqDist
        fdist1 = FreqDist(words_ns_fr)
        most_common_fr = fdist1.most_common(15)

        return render_template("info_by_page_fr.html", most_common = most_common)

if __name__ == '__main__':
    app.run()



    #split feedback by What's wrongfully

    #get most meaningful word by what's wrong
