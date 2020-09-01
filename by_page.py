
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
    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    import io
    import base64

    #define deserialize
    def deserialize(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    #import data as pickle
    data = deserialize('data/all_data.pickle')

    page = request.args.get('page')
    start_date = request.args.get('start_date', '2020-01-01')
    end_date = request.args.get('end_date', '2020-12-31')

    #get variables

    data = data[data['Date'] <= end_date]
    data = data[data['Date'] >= start_date]

    data['URL'] = data['URL'].str.replace('/content/canadasite', 'www.canada.ca')
    data['URL'] = data['URL'].str.replace('www.canada.ca', 'https://www.canada.ca')
    page_data = data.loc[data['URL'] == page]
    page_data = page_data.reset_index(drop=True)

    url = page_data['URL'][0]

    if page_data['Lang'][0] == 'EN':

        page_data_en = page_data[["Comment", "Date", "Status",  "What's wrong", "Lookup_tags", 'Tags confirmed', 'Yes/No', 'Lookup_page_title']]
        page_data_en = page_data_en[page_data_en.Status != 'Spam']
        page_data_en = page_data_en[page_data_en.Status != 'Ignore']
        page_data_en = page_data_en[page_data_en.Status != 'Duplicate']

        page_data_en['Lookup_page_title'] = [','.join(map(str, l)) for l in page_data_en['Lookup_page_title']]

        title = page_data_en['Lookup_page_title'][0]

        yes_no = page_data_en[["Date", 'Yes/No']]
        yes_no = yes_no.dropna()
        yes_no = yes_no.sort_values(by = 'Date', ascending=False)
        yes_no = yes_no.reset_index(drop=True)

        by_date = {}
        for date in yes_no['Date']:
          by_date[date] = yes_no.loc[yes_no['Date'] == date]

        for date in by_date:
          by_date[date] =  by_date[date]['Yes/No'].value_counts()

        for date in by_date:
          if 'No' not in by_date[date]:
            by_date[date]['No'] = 0

        for date in by_date:
          if 'Yes' not in by_date[date]:
            by_date[date]['Yes'] = 0

        for date in by_date:
          by_date[date] = (by_date[date]['Yes']/(by_date[date]['Yes'] + by_date[date]['No'])) * 100


        dates = list(by_date.keys())
        values = list(by_date.values())
        dates.reverse()
        values.reverse()
        img = io.BytesIO()
        x = dates
        y = values
        plt.xticks(rotation=90)
        plt.ylim(0, 100)
        plt.plot(x, y, linewidth=2.0)
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)

        plot_url = base64.b64encode(img.getvalue()).decode()

        if yes_no.empty:
            score = 'unavailable'
            yes = 'unavailable'
            no = 'unavailable'
        else:
            total = yes_no['Yes/No'].value_counts()
            yes = total['Yes']
            no = total['No']
            score = (total['Yes'] / ( total['Yes'] +  total['No'])) * 100
            score = format(score, '.2f')

        page_data_en = page_data_en.drop(columns=['Status'])
        page_data_en = page_data_en.drop(columns=['Yes/No'])
        page_data_en["What's wrong"].fillna(False, inplace=True)
        page_data_en = page_data_en.dropna()

        #converts the tags to a string (instead of a list) - needed for further processing - and puts it in a new column
        page_data_en['tags'] = [','.join(map(str, l)) for l in page_data_en['Lookup_tags']]

        #remove the Lookup_tags column (it's not needed anymore)
        page_data_en = page_data_en.drop(columns=['Lookup_tags'])


        #remove the Lookup_page_title column (it's not needed anymore)
        page_data_en = page_data_en.drop(columns=['Tags confirmed'])

        #resets the index for each row - needed for further processing
        page_data_en = page_data_en.reset_index(drop=True)

        #split dataframe for French comments - same comments as above for each line

        #get data for specific page

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
                if word not in sw and word.isalpha():
                    words_ns_en.append(word)
        #get most common words
        from nltk import FreqDist
        fdist1 = FreqDist(words_ns_en)
        most_common = fdist1.most_common(15)
        mc = pd.DataFrame(most_common, columns =['Word', 'Count'])
        mc = mc[['Count', 'Word']]
        word_column_names = ['Count', 'Word']


        page_data_en = page_data_en.reset_index(drop=True)

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

        clean_columns = ['Comment']
        clean_en = pd.DataFrame(columns = clean_columns)
        clean_en['Comment'] = page_data_en['Comment'].str.lower()
        clean_en['Comment'] = clean_en['Comment'].apply(cleanPunc)
        clean_en['Comment'] = clean_en['Comment'].apply(keepAlpha)
        clean_en['Comment'] = clean_en['Comment'].apply(stemming_en)


        from sklearn.feature_extraction.text import TfidfVectorizer


        all_text_en = []
        all_text_en = clean_en['Comment'].values.astype('U')


        vects_en = []
        all_x_en = []
        vectorizer_en = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
        vects_en = vectorizer_en.fit(all_text_en)
        all_x_en = vects_en.transform(all_text_en)

        from sklearn.cluster import AffinityPropagation
        from sklearn import metrics
        from sklearn.datasets import make_blobs

        X = all_x_en


        from sklearn.cluster import AffinityPropagation
        from sklearn import metrics
        from sklearn.datasets import make_blobs

        afprop = AffinityPropagation(max_iter=300, damping=0.6)
        afprop.fit(X)
        cluster_centers_indices = afprop.cluster_centers_indices_
        X.toarray()
        P = afprop.predict(X)

        import collections

        occurrences = collections.Counter(P)



        cluster_columns = ['Number of feedback', 'Representative comment']
        clusters = pd.DataFrame(columns = cluster_columns)

        cluster_rep = list(cluster_centers_indices)
        rep_comment = []
        for indice in cluster_rep:
          rep_comment.append(page_data_en['Comment'][indice])


        clusters['Representative comment'] = rep_comment

        cluster_couple = sorted(occurrences.items())
        cluster_count = []

        cluster_count = [x[1] for x in cluster_couple]
        clusters['Number of feedback'] = cluster_count
        clusters = clusters.sort_values(by = 'Number of feedback', ascending=False)
        clusters = clusters.reset_index(drop=True)

        page_data_en['group'] = P
        cluster_group = page_data_en[['Comment', 'Date', 'group']]

        for group in cluster_group['group']:
          cluster_group[group] = rep_comment[group]


        group_dict = {}

        for group, group_df_en in page_data_en.groupby("group"):
              group_dict[group] = group_df_en[['Date', 'Comment']]


        for group in group_dict:
          group_dict[group] = group_dict[group].sort_values(by = 'Date', ascending=False)


        group_columns = ['Date', 'Comment']

        unique_groups = list(page_data_en['group'].unique())

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

        by_tag = by_tag.sort_values(by = 'Feedback count', ascending=False)
        unique_tags = list(by_tag.index)

        #split feedback by tag

        tag_dico = {}

        tag_dico_columns = ['Date', 'Comment']

        for tag in unique_tags:
          tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

        for tag, topic_df_en in page_data_en.groupby(0):
          tag_dico[tag] = topic_df_en[['Date', 'Comment']]


        if 1 in page_data_en.columns:
            for tag, topic_df_en in page_data_en.groupby(1):
                if tag_dico[tag].empty:
                    tag_dico[tag] = topic_df_en[['Date', 'Comment']]
                else:
                    tag_dico[tag] = tag_dico[tag].append(topic_df_en[['Date', 'Comment']])

        if 2 in page_data_en.columns:
            for tag, topic_df_en in page_data_en.groupby(2):
                if tag_dico[tag].empty:
                    tag_dico[tag] = topic_df_en[['Date', 'Comment']]
                else:
                    tag_dico[tag] = tag_dico[tag].append(topic_df_en[['Date', 'Comment']])


        for tag in tag_dico:
            tag_dico[tag] = tag_dico[tag].sort_values(by = 'Date', ascending=False)


        tag_columns = ['Date', 'Comment']


        column_names = ['Tag count', 'Tag', 'Significant words']



        return render_template("info_by_page_en.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, most_common = most_common, column_names = column_names, row_data = list(by_tag.values.tolist()), zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), word_column_names = word_column_names, row_data_word = list(mc.values.tolist()), cluster_columns = cluster_columns, groups = zip(list(clusters['Representative comment'].values.tolist()), list(clusters['Number of feedback'].values.tolist()), unique_groups), group_dict = group_dict, group_columns = group_columns, unique_groups = unique_groups, list = list, tag_columns = tag_columns, tags = zip(unique_tags, list(by_tag['Feedback count'].values.tolist()), unique_tags), tag_dico = tag_dico)



    #process to follow if English
    else:

        #keep only relevant columns from the dataframe
        page_data_fr = page_data[["Comment", "Date", "Status",  "What's wrong", "Lookup_tags", 'Tags confirmed', 'Yes/No', 'Lookup_page_title']]
        page_data_fr = page_data_fr[page_data_fr.Status != 'Spam']
        page_data_fr = page_data_fr[page_data_fr.Status != 'Ignore']
        page_data_fr = page_data_fr[page_data_fr.Status != 'Duplicate']

        page_data_fr['Lookup_page_title'] = [','.join(map(str, l)) for l in page_data_fr['Lookup_page_title']]

        title = page_data_fr['Lookup_page_title'][0]

        yes_no = page_data_fr[["Date", 'Yes/No']]
        yes_no = yes_no.dropna()
        yes_no = yes_no.sort_values(by = 'Date', ascending=False)
        yes_no = yes_no.reset_index(drop=True)

        by_date = {}
        for date in yes_no['Date']:
          by_date[date] = yes_no.loc[yes_no['Date'] == date]

        for date in by_date:
          by_date[date] =  by_date[date]['Yes/No'].value_counts()

        for date in by_date:
          if 'No' not in by_date[date]:
            by_date[date]['No'] = 0

        for date in by_date:
          if 'Yes' not in by_date[date]:
            by_date[date]['Yes'] = 0

        for date in by_date:
          by_date[date] = (by_date[date]['Yes']/(by_date[date]['Yes'] + by_date[date]['No'])) * 100


        dates = list(by_date.keys())
        values = list(by_date.values())
        dates.reverse()
        values.reverse()
        img = io.BytesIO()
        x = dates
        y = values
        plt.xticks(rotation=90)
        plt.ylim(0, 100)
        plt.plot(x, y, linewidth=2.0)
        plt.savefig(img, format='png')
        plt.close()
        img.seek(0)

        plot_url = base64.b64encode(img.getvalue()).decode()

        if yes_no.empty:
            score = 'unavailable'
        else:
            total = yes_no['Yes/No'].value_counts()
            yes = total['Yes']
            no = total['No']
            score = (total['Yes'] / ( total['Yes'] +  total['No'])) * 100
            score = format(score, '.2f')

        page_data_fr = page_data_fr.drop(columns=['Status'])
        page_data_fr = page_data_fr.drop(columns=['Yes/No'])
        page_data_fr["What's wrong"].fillna(False, inplace=True)
        page_data_fr = page_data_fr.dropna()

        #converts the tags to a string (instead of a list) - needed for further processing - and puts it in a new column
        page_data_fr['tags'] = [','.join(map(str, l)) for l in page_data_fr['Lookup_tags']]

        #remove the Lookup_tags column (it's not needed anymore)
        page_data_fr = page_data_fr.drop(columns=['Lookup_tags'])


        #remove the Lookup_page_title column (it's not needed anymore)
        page_data_fr = page_data_fr.drop(columns=['Tags confirmed'])

        #resets the index for each row - needed for further processing
        page_data_fr = page_data_fr.reset_index(drop=True)

        #split dataframe for French comments - same comments as above for each line

        #get data for specific page

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
                if word not in sw and word.isalpha():
                    words_ns_fr.append(word)


        #get most common words
        from nltk import FreqDist
        fdist1 = FreqDist(words_ns_fr)
        most_common = fdist1.most_common(15)
        mc = pd.DataFrame(most_common, columns =['Mots', 'Nombre'])
        mc = mc[['Nombre', 'Mots']]
        word_column_names = ['Nombre', 'Mots']


        page_data_fr = page_data_fr.reset_index(drop=True)

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
        stemmer_fr = SnowballStemmer("french")
        def stemming_fr(sentence):
            stemSentence = ""
            for word in sentence.split():
                stem = stemmer_fr.stem(word)
                stemSentence += stem
                stemSentence += " "
            stemSentence = stemSentence.strip()
            return stemSentence

        clean_columns = ['Comment']
        clean_fr = pd.DataFrame(columns = clean_columns)
        clean_fr['Comment'] = page_data_fr['Comment'].str.lower()
        clean_fr['Comment'] = clean_fr['Comment'].apply(cleanPunc)
        clean_fr['Comment'] = clean_fr['Comment'].apply(keepAlpha)
        clean_fr['Comment'] = clean_fr['Comment'].apply(stemming_fr)


        from sklearn.feature_extraction.text import TfidfVectorizer


        all_text_fr = []
        all_text_fr = clean_fr['Comment'].values.astype('U')


        vects_fr = []
        all_x_fr = []
        vectorizer_fr = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
        vects_fr = vectorizer_fr.fit(all_text_fr)
        all_x_fr = vects_fr.transform(all_text_fr)

        from sklearn.cluster import AffinityPropagation
        from sklearn import metrics
        from sklearn.datasets import make_blobs

        X = all_x_fr


        from sklearn.cluster import AffinityPropagation
        from sklearn import metrics
        from sklearn.datasets import make_blobs

        afprop = AffinityPropagation(max_iter=300, damping=0.6)
        afprop.fit(X)
        cluster_centers_indices = afprop.cluster_centers_indices_
        X.toarray()
        P = afprop.predict(X)

        import collections

        occurrences = collections.Counter(P)



        cluster_columns = ['Nombre de rétroactions', 'Commentaire représentatif']
        clusters = pd.DataFrame(columns = cluster_columns)

        cluster_rep = list(cluster_centers_indices)
        rep_comment = []
        for indice in cluster_rep:
          rep_comment.append(page_data_fr['Comment'][indice])


        clusters['Commentaire représentatif'] = rep_comment

        cluster_couple = sorted(occurrences.items())
        cluster_count = []

        cluster_count = [x[1] for x in cluster_couple]
        clusters['Nombre de rétroactions'] = cluster_count
        clusters = clusters.sort_values(by = 'Nombre de rétroactions', ascending=False)
        clusters = clusters.reset_index(drop=True)

        page_data_fr['group'] = P
        cluster_group = page_data_fr[['Comment', 'Date', 'group']]

        for group in cluster_group['group']:
          cluster_group[group] = rep_comment[group]


        group_dict = {}

        for group, group_df_fr in page_data_fr.groupby("group"):
              group_dict[group] = group_df_fr[['Date', 'Comment']]


        for group in group_dict:
          group_dict[group] = group_dict[group].sort_values(by = 'Date', ascending=False)


        group_columns = ['Date', 'Commentaire']

        unique_groups = list(page_data_fr['group'].unique())

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


        reason_list_fr= []
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


        reason_column_names = ['Nombre de rétroactions', 'Raison', 'Mots significatifs']

        by_reason['Significant words'] = by_reason['Significant words'].apply(lambda x: ', '.join(x))

        by_reason = by_reason[['Feedback count', 'index', 'Significant words']]



        #count the number for each tag
        tag_count = tags_fr.apply(pd.Series.value_counts)
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

        by_tag = by_tag.sort_values(by = 'Feedback count', ascending=False)
        unique_tags = list(by_tag.index)

        #split feedback by tag

        tag_dico = {}

        tag_dico_columns = ['Date', 'Commentaire']

        for tag in unique_tags:
          tag_dico[tag] = pd.DataFrame(columns = tag_dico_columns)

        for tag, topic_df_fr in page_data_fr.groupby(0):
          tag_dico[tag] = topic_df_fr[['Date', 'Comment']]


        if 1 in page_data_fr.columns:
            for tag, topic_df_fr in page_data_fr.groupby(1):
                if tag_dico[tag].empty:
                    tag_dico[tag] = topic_df_fr[['Date', 'Comment']]
                else:
                    tag_dico[tag].append(topic_df_fr[['Date', 'Comment']])

        if 2 in page_data_fr.columns:
            for tag, topic_df_fr in page_data_fr.groupby(2):
                if tag_dico[tag].empty:
                    tag_dico[tag] = topic_df_fr[['Date', 'Comment']]
                else:
                    tag_dico[tag].append(topic_df_fr[['Date', 'Comment']])


        for tag in tag_dico:
            tag_dico[tag] = tag_dico[tag].sort_values(by = 'Date', ascending=False)


        tag_columns = ['Date', 'Commentaire']


        column_names = ['Nombre de rétroactions', 'Étiquette', 'Mots significatifs']



        return render_template("info_by_page_fr.html", title = title, url = url, start_date = start_date, end_date = end_date, yes = yes, no = no, plot_url = plot_url, score = score, most_common = most_common, column_names = column_names, row_data = list(by_tag.values.tolist()), zip = zip, page = page, reason_column_names = reason_column_names, row_data_reason = list(by_reason.values.tolist()), word_column_names = word_column_names, row_data_word = list(mc.values.tolist()), cluster_columns = cluster_columns, groups = zip(list(clusters['Commentaire représentatif'].values.tolist()), list(clusters['Nombre de rétroactions'].values.tolist()), unique_groups), group_dict = group_dict, group_columns = group_columns, unique_groups = unique_groups, list = list, tag_columns = tag_columns, tags = zip(unique_tags, list(by_tag['Feedback count'].values.tolist()), unique_tags), tag_dico = tag_dico)

if __name__ == '__main__':
    app.run()



    #split feedback by What's wrongfully

    #get most meaningful word by what's wrong
