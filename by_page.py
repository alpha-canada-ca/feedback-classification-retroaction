
from flask import Flask
from flask import request
from flask import app, render_template


app = Flask(__name__)

@app.route('/bypage', methods=['GET', 'POST'])

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

    lang = request.args.get('lang')
    page = request.args.get('page')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    #get variables

    if lang == 'en':

        data = data[data['Date'] <= end_date]
        data = data[data['Date'] >= start_date]

        #split dataframe for English comments
        data_en = data[data['Lang'].str.contains("EN", na=False)]

        #keep only relevant columns from the dataframe
        data_en_pages = data_en[["Comment", "Combined EN/FR field", "What's wrong", "Lookup_tags", 'Tags confirmed']]
        data_en_pages["What's wrong"].fillna(False, inplace=True)

        #remove all rows thave have null content
        data_en_pages = data_en_pages.dropna()

        #converts the tags to a string (instead of a list) - needed for further processing - and puts it in a new column
        data_en_pages['tags'] = [','.join(map(str, l)) for l in data_en_pages['Lookup_tags']]

        #remove the Lookup_tags column (it's not needed anymore)
        data_en_pages = data_en_pages.drop(columns=['Lookup_tags'])


        #remove the Lookup_page_title column (it's not needed anymore)
        data_en_pages = data_en_pages.drop(columns=['Tags confirmed'])

        #resets the index for each row - needed for further processing
        data_en_pages = data_en_pages.reset_index(drop=True)

        #split dataframe for French comments - same comments as above for each line

        #get data for specific page
        page_data_en = data_en_pages.loc[data_en_pages['Combined EN/FR field'] == page]

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


        #by what's wrong reason
        page_data_en[["What's wrong"]] = page_data_en[["What's wrong"]].replace([False], ['None'])
        page_data_en[["What's wrong"]] = page_data_en[["What's wrong"]].replace(["The information isn't clear"], ["The information isn’t clear"])
        page_data_en[["What's wrong"]] = page_data_en[["What's wrong"]].replace(["I'm not in the right place"], ["I’m not in the right place"])
        reasons = page_data_en["What's wrong"].value_counts()
        by_reason= reasons.to_frame()
        by_reason.columns = ['Feedback count']

        reason_dict = {}

        for reason, topic_df_en in page_data_en.groupby("What's wrong"):
            reason_dict[reason] = ' '.join(topic_df_en['Comment'].tolist())


        tokenizer = nltk.RegexpTokenizer(r"\w+")

        for value in reason_dict:
            reason_dict[value] = tokenizer.tokenize(reason_dict[value])


        reason_list_en= []
        for keys in reason_dict.keys():
            reason_list_en.append(keys)


        reason_words_en = []
        for values in reason_dict.values():
            reason_words_en.append(values)


        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        from nltk.corpus import stopwords

        reason_words_en = [[word.lower() for word in value] for value in reason_words_en]
        reason_words_en = [[lemmatizer.lemmatize(word) for word in value] for value in reason_words_en]
        reason_words_en = [[word for word in value if word not in sw] for value in reason_words_en]
        reason_words_en = [[word for word in value if word.isalpha()] for value in reason_words_en]

        from gensim.corpora.dictionary import Dictionary

        reason_dictionary_en = Dictionary(reason_words_en)

        reason_corpus_en = [reason_dictionary_en.doc2bow(reason) for reason in reason_words_en]

        from gensim.models.tfidfmodel import TfidfModel

        reason_tfidf_en = TfidfModel(reason_corpus_en)

        reason_tfidf_weights_en = [sorted(reason_tfidf_en[doc], key=lambda w: w[1], reverse=True) for doc in reason_corpus_en]

        reason_weighted_words_en = [[(reason_dictionary_en.get(id), weight) for id, weight in ar] for ar in reason_tfidf_weights_en]

        reason_imp_words_en = pd.DataFrame({'Reason': reason_list_en, 'EN_words':  reason_weighted_words_en})

        reason_imp_words_en = reason_imp_words_en.sort_values(by = 'Reason')

        reason_imp_words_en = reason_imp_words_en.reset_index(drop=True)

        reason_imp_words_en['EN_words'] = reason_imp_words_en['EN_words'].apply(lambda x: list(x))

        reason_imp_words_en['EN_words'] = reason_imp_words_en['EN_words'].apply(lambda x: x[:15])

        reason_imp_words_en['EN_words'] = reason_imp_words_en['EN_words'].apply(lambda x: [y[0] for y in x])

        by_reason = by_reason.reset_index()

        by_reason['Significant words'] = reason_imp_words_en['EN_words']

        by_reason= by_reason.sort_values(by = 'Feedback count', ascending=False)


        reason_column_names = ['Feedback count', 'Reason', 'Significant words']

        by_reason['Significant words'] = by_reason['Significant words'].apply(lambda x: ', '.join(x))

        by_reason = by_reason[['Feedback count', 'index', 'Significant words']]



        #count the number for each tag
        tag_count = tags_en.apply(pd.Series.value_counts)
        tag_count = tag_count.fillna(0)
        tag_count = tag_count.astype(int)
        if 2 in tag_count.columns:
            tag_count = tag_count[0] + tag_count[1] + tag_count[2]
        elif 1 in tag_count.columns:
            tag_count = tag_count[0] + tag_count[1]
        else:
            tag_count = tag_count[0]
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



        return render_template("info_by_page_en.html", most_common = most_common, column_names = column_names, row_data = list(by_tag.values.tolist()), zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()))



    #process to follow if English
    else:

        data_fr = data[data['Lang'].str.contains("FR", na=False)]

        #keep only relevant columns from the dataframe
        data_fr_pages = data_fr[["Comment", "Combined EN/FR field", "What's wrong", "Lookup_tags", 'Tags confirmed']]
        data_fr_pages["What's wrong"].fillna(False, inplace=True)

        #remove all rows thave have null content
        data_fr_pages = data_fr_pages.dropna()


        #converts the tags to a string (instead of a list) - needed for further processing - and puts it in a new column
        data_fr_pages['tags'] = [','.join(map(str, l)) for l in data_fr_pages['Lookup_tags']]

        #remove the Lookup_tags column (it's not needed anymore)
        data_fr_pages = data_fr_pages.drop(columns=['Lookup_tags'])


        #remove the Lookup_page_title column (it's not needed anymore)
        data_fr_pages = data_fr_pages.drop(columns=['Tags confirmed'])

        #resets the index for each row - needed for further processing
        data_fr_pages = data_fr_pages.reset_index(drop=True)

        #split dataframe for French comments - same comments as above for each line

        #get data for specific page
        page_data_fr = data_fr_pages.loc[data_fr_pages['Combined EN/FR field'] == page]

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
        sw.append('a')
        sw.append('si')
        sw.append('avoir')
        sw.append('savoir')
        sw.append('combien')
        sw.append('être')
        sw.append('où')
        sw.append('comment')
        sw.append('puis')
        sw.append('peuvent')
        sw.append('fait')
        sw.append('aucun')
        sw.append('bonjour')
        sw.append('depuis')
        sw.append('chez')
        sw.append('faire')
        sw.append('peut')
        sw.append('plus')
        sw.append('veux')
        sw.append('dois')
        sw.append('doit')
        sw.append('dit')
        sw.append('merci')
        sw.append('cela')
        sw.append('pouvons')
        sw.append('pouvaient')
        sw.append('vers')

        words_ns_fr = []
        for word in words_fr:
                if word not in sw:
                    words_ns_fr.append(word)



        #get most common words
        from nltk import FreqDist
        fdist1 = FreqDist(words_ns_fr)
        most_common = fdist1.most_common(15)


        #by what's wrong reason
        page_data_fr[["What's wrong"]] = page_data_fr[["What's wrong"]].replace([False], ['None'])
        page_data_fr[["What's wrong"]] = page_data_fr[["What's wrong"]].replace(["The information isn't clear"], ["The information isn’t clear"])
        page_data_fr[["What's wrong"]] = page_data_fr[["What's wrong"]].replace(["I'm not in the right place"], ["I’m not in the right place"])
        reasons = page_data_fr["What's wrong"].value_counts()
        by_reason= reasons.to_frame()
        by_reason.columns = ['Feedback count']

        reason_dict = {}

        for reason, topic_df_fr in page_data_fr.groupby("What's wrong"):
            reason_dict[reason] = ' '.join(topic_df_fr['Comment'].tolist())


        tokenizer = nltk.RegexpTokenizer(r"\w+")

        for value in reason_dict:
            reason_dict[value] = tokenizer.tokenize(reason_dict[value])


        reason_list_fr = []
        for keys in reason_dict.keys():
            reason_list_fr.append(keys)


        reason_words_fr = []
        for values in reason_dict.values():
            reason_words_fr.append(values)


        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        from nltk.corpus import stopwords

        reason_words_fr = [[word.lower() for word in value] for value in reason_words_fr]
        reason_words_fr = [[lemmatizer.lemmatize(word) for word in value] for value in reason_words_fr]
        reason_words_fr = [[word for word in value if word not in sw] for value in reason_words_fr]
        reason_words_fr = [[word for word in value if word.isalpha()] for value in reason_words_fr]

        from gensim.corpora.dictionary import Dictionary

        reason_dictionary_fr = Dictionary(reason_words_fr)

        reason_corpus_fr = [reason_dictionary_fr.doc2bow(reason) for reason in reason_words_fr]

        from gensim.models.tfidfmodel import TfidfModel

        reason_tfidf_fr = TfidfModel(reason_corpus_fr)

        reason_tfidf_weights_fr = [sorted(reason_tfidf_fr[doc], key=lambda w: w[1], reverse=True) for doc in reason_corpus_fr]

        reason_weighted_words_fr = [[(reason_dictionary_fr.get(id), weight) for id, weight in ar] for ar in reason_tfidf_weights_fr]

        reason_imp_words_fr = pd.DataFrame({'Reason': reason_list_fr, 'FR_words':  reason_weighted_words_fr})

        reason_imp_words_fr = reason_imp_words_fr.sort_values(by = 'Reason')

        reason_imp_words_fr = reason_imp_words_fr.reset_index(drop=True)

        reason_imp_words_fr['FR_words'] = reason_imp_words_fr['FR_words'].apply(lambda x: list(x))

        reason_imp_words_fr['FR_words'] = reason_imp_words_fr['FR_words'].apply(lambda x: x[:15])

        reason_imp_words_fr['FR_words'] = reason_imp_words_fr['FR_words'].apply(lambda x: [y[0] for y in x])

        by_reason = by_reason.reset_index()

        by_reason['Significant words'] = reason_imp_words_fr['FR_words']

        by_reason= by_reason.sort_values(by = 'Feedback count', ascending=False)


        reason_column_names = ['Nombre', 'Raison', 'Mots sinificatifs']

        by_reason['Significant words'] = by_reason['Significant words'].apply(lambda x: ', '.join(x))

        by_reason = by_reason[['Feedback count', 'index', 'Significant words']]



        #count the number for each tag
        tag_count = tags_fr.apply(pd.Series.value_counts)
        tag_count = tag_count.fillna(0)
        tag_count = tag_count.astype(int)
        if 2 in tag_count.columns:
          tag_count = tag_count[0] + tag_count[1] + + tag_count[2]
        elif 1 in tag_count.columns:
          tag_count = tag_count[0] + tag_count[1]
        else:
          tag_count = tag_count[0]


        tag_count = tag_count.sort_values(ascending = False)
        by_tag = tag_count.to_frame()
        by_tag = by_tag.sort_index(axis=0, level=None, ascending=True)
        by_tag.columns = ['Feedback count']


        #split feedback by tag

        tag_dict = {}

        if 2 in page_data_fr.columns:
            for tag, topic_df_fr in page_data_fr.groupby(2):
                tag_dict[tag] = ' '.join(topic_df_fr['Comment'].tolist())


        if 1 in page_data_fr.columns:
            for tag, topic_df_fr in page_data_fr.groupby(1):
                tag_dict[tag] = ' '.join(topic_df_fr['Comment'].tolist())


        for tag, topic_df_fr in page_data_fr.groupby(0):
            tag_dict[tag] = ' '.join(topic_df_fr['Comment'].tolist())



        #tokenize

        tokenizer = nltk.RegexpTokenizer(r"\w+")

        for value in tag_dict:
            tag_dict[value] = tokenizer.tokenize(tag_dict[value])


        #get most meaningful words by tags
        topic_list_fr = []
        for keys in tag_dict.keys():
            topic_list_fr.append(keys)


        topic_words_fr = []
        for values in tag_dict.values():
            topic_words_fr.append(values)


        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        from nltk.corpus import stopwords

        topic_words_fr = [[word.lower() for word in value] for value in topic_words_fr]
        topic_words_fr = [[lemmatizer.lemmatize(word) for word in value] for value in topic_words_fr]
        topic_words_fr = [[word for word in value if word not in sw] for value in topic_words_fr]
        topic_words_fr = [[word for word in value if word.isalpha()] for value in topic_words_fr]

        from gensim.corpora.dictionary import Dictionary

        dictionary_fr = Dictionary(topic_words_fr)

        corpus_fr = [dictionary_fr.doc2bow(tag) for tag in topic_words_fr]

        from gensim.models.tfidfmodel import TfidfModel

        tfidf_fr = TfidfModel(corpus_fr)

        tfidf_weights_fr = [sorted(tfidf_fr[doc], key=lambda w: w[1], reverse=True) for doc in corpus_fr]

        weighted_words_fr = [[(dictionary_fr.get(id), weight) for id, weight in ar] for ar in tfidf_weights_fr]

        imp_words_fr = pd.DataFrame({'Tag': topic_list_fr, 'FR_words':  weighted_words_fr})

        imp_words_fr = imp_words_fr.sort_values(by = 'Tag')

        imp_words_fr = imp_words_fr.reset_index(drop=True)

        imp_words_fr['FR_words'] = imp_words_fr['FR_words'].apply(lambda x: list(x))

        imp_words_fr['FR_words'] = imp_words_fr['FR_words'].apply(lambda x: x[:9])

        imp_words_fr['FR_words'] = imp_words_fr['FR_words'].apply(lambda x: [y[0] for y in x])

        imp_words_fr = imp_words_fr.set_index('Tag', drop=True, append=False)

        by_tag['Significant words'] = imp_words_fr['FR_words']

        by_tag = by_tag.sort_values(by = 'Feedback count', ascending=False)

        by_tag = by_tag.reset_index()

        column_names = ['Nombre', 'Étiquette', 'Mots significatifs']

        by_tag['Significant words'] = by_tag['Significant words'].apply(lambda x: ', '.join(x))

        by_tag = by_tag[['Feedback count', 'index', 'Significant words']]

        return render_template("info_by_page_fr.html", most_common = most_common, column_names = column_names, row_data = list(by_tag.values.tolist()), zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()))

if __name__ == '__main__':
    app.run()



    #split feedback by What's wrongfully

    #get most meaningful word by what's wrong
