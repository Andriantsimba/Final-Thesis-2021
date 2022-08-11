import imp
from multiprocessing.sharedctypes import Array
from os import O_RDONLY
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score
from asyncio.windows_events import NULL
import math
from .models import Keyword, News, newsScoring as Scoring
import string
from django.shortcuts import render, redirect
from django.contrib import messages
import requests
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import date
from collections.abc import Iterable
from bs4 import BeautifulSoup
from mechanize import Browser
import re
import pandas as pd
import numpy as np
import json
from newspaper import Article
from googlesearch import search
from prettytable import PrettyTable
import articleDateExtractor
import tldextract
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('wordnet')
nltk.download('omw-1.4')
# Create your views here.


def index(request):
    if request.method == 'POST':
        query = request.POST['logasp']
        if query != '':
            url = []
            # url_down = []
            checkurl = ['facebook', 'twitter',
                        'youtube', 'dictionary', 'wikipedia', 'babla', 'tokopedia', 'shopee', 'amazon', 'photo', 'pdf', 'cnic', '-systems', 'edu']
            # add data to keyword Database
            k = request.POST['logasp']
            k = text_lowercase(k)
            k = remove_punctuation(k)
            k = remove_stopwords(k)
            k = stem_sentences(k)
            k = lem_sentences(k)
            key = Keyword(
                logasp=k
            )
            # key.save()
            request.session['query'] = query
            for j in search(query, num_results=20, lang='en'):
                print(j, '\n')
                url.append(j)
            print(url)
            unique_url = get_unique_urls(url)

            for i in unique_url:
                urlcheck = any(u in i for u in checkurl)
                if urlcheck is False:
                    try:
                        # target url
                        url = i

                        # Creating browser instance
                        br = Browser()
                        br.open(url)

                        print("~"*30, 'news ', "~"*30)
                        # getting the title
                        title = br.title()
                        print('title of the article:', title)
                    # preprocessing of the title
                        lower = text_lowercase(title)
                        rem_ponc = remove_punctuation(lower)
                        rem_stop = remove_stopwords(rem_ponc)
                        stem_sent = stem_sentences(rem_stop)
                        lem_sent = lem_sentences(stem_sent)
                        print('preprocessed title: ', lem_sent)

                        date_publish = getdate(i)
                        print('date of publish is: ', date_publish)
                        print(type(date_publish))

                        if title != '' and date_publish is not None:
                            news = News(
                                news_title=title,
                                pp_title=lem_sent,
                                news_link=i,
                                news_dop=date_publish
                            )
                            # news.save()
                        else:
                            pass

                        # else:
                            # url_down.append(i)
                            # pass
                    except:
                        pass
            return redirect('newsScoring')

        else:
            messages.info(request, ' Keywords field is Empty')
            redirect('/')
    # print('list urldown ', '\n', url_down)
    return render(request, 'index.html')
# check the uniqness of the url


def get_unique_urls(url):
    unique = []

    for u in url:
        if u in unique:
            continue
        else:
            unique.append(u)
    print(len(unique))
    print(unique)
    return unique


# get the dates
def getdate(url):

    dateurl = None
    try:
        d = articleDateExtractor.extractArticlePublishedDate(url)
        dateurl = d
        if dateurl is None:
            article = Article(
                url)
            article.download()
            article.parse()
            dateurl = article.publish_date
            print('the code passed here')
            print(dateurl)

    except:
        dateurl = None
    # if the date is still None after the two library usage
    if dateurl is None:
        dateurl = "1945-06-25"
    print('final date is: ', dateurl)
    return dateurl

# ~~~~~~~~~~~~~~~~~~~~~~~~~ text preprocessing ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def text_lowercase(text):
    return text.lower()


# poctuation removal
def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


# Stop_word remove
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    # filtered_text = [word for word in word_tokens if word not in stop_words]
    filtered_text = []
    for word in word_tokens:
        if word not in stop_words:
            filtered_text.append(word)
            filtered_text.append(" ")
            final_stop = "".join(filtered_text)
    return final_stop


# sentence stemming
def stem_sentences(text):
    stemmer = PorterStemmer()
    words = word_tokenize(text)
    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))
        stemmed_words.append(" ")
        final_stem = "".join(stemmed_words)
    return final_stem


# lemmentizing
def lem_sentences(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = []
    for word in words:
        lemmatized_words.append(lemmatizer.lemmatize(word))
        lemmatized_words.append(" ")
        final_lem = "".join(lemmatized_words)
    return final_lem

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~end of text preprocessing ~~~~~~~~~~~~~~~~~~~~~~~~~~


def newS(request):
    new = News.objects.all
    query = request.session['query']
    return render(request, 'new.html', {'new': new, 'query': query})


# check the nan value for the message credibility
def changeNan(list):
    newlist = []
    for i in range(len(list)):
        if math.isnan(list[i]) is True:
            list[i] = 0
            newlist.append(list[i])
        else:
            newlist.append(list[i])
    print('the new list is here \n', newlist)
    return newlist
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Online news Scoring ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def newsScoring(request):
    news = News.objects.all()
    today = date.today()
    data = gettimedata(news)
    documents = getDoc(news)
    query = request.session['query']

    similarity_docs = rank_similarity_docs(documents, query)
    similarity_docs = list(flatten(similarity_docs))
    gr = website_cred(news)
    cosi_sim = changeNan(similarity_docs)
    print('Similarity dox: ', cosi_sim)
    # print('global rank: ', gr)

    for n, s, g in zip(news, cosi_sim, gr):
        idn = n.id
        title = n.news_title
        dop = n.news_dop
        fdop = (today-dop).days
        fdopnormalized = normalization(data, fdop)
        sim_doc = float(s)
        glob = normalization(gr, g)

        print("id: ", idn)
        print("title: ", title)
        print("difference date: ", fdop)
        # normalized date of publish
        print("normalized time score: ", fdopnormalized)
        print('type of this variable is: ', type(fdopnormalized))
        print('similarity: ', sim_doc)
        print('Global Rank: ', glob)
        print('type of this variable is: ', type(glob))
        print('~'*35, 'news', '~'*35)

        score = Scoring(
            nTitle=title,
            tc=fdopnormalized,
            mc=sim_doc,
            gr=glob,
            l='0'
        )
        # score.save()

    # website_cred(news)
    return render(request, 'news.html', {'news': news, 'query': query})

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Time Credibility ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# get the difference of time between the day of the research
# and the date of publish for data normalization


def gettimedata(data):
    listnews = []
    today = date.today()
    for dt in data:
        d = dt.news_dop
        fdop = (today-d).days
        listnews.append(fdop)
        # return fdata
    return listnews


# normalization
def normalization(data, x):
    try:
        norm = (x-min(data))/(max(data)-min(data))
    except ZeroDivisionError:
        norm = 0
    return norm


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~end of time credibility ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ message credibility ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# getting all preprocessed title as document
def getDoc(data):
    documents = []
    for n in data:
        new = n.pp_title
        documents.append(new)
    return documents


# term frequency of DocumentLS
def termFrequency(term, documents):
    normalizeDocument = documents.split()
    return normalizeDocument.count(term)/float(len(normalizeDocument))


def compute_normalizedtf(documents):
    tf_doc = []
    for txt in documents:
        sentence = txt.split()
        norm_tf = dict.fromkeys(set(sentence), 0)
        for word in sentence:
            norm_tf[word] = termFrequency(word, txt)
        tf_doc.append(norm_tf)
        df = pd.DataFrame([norm_tf])
        idx = 0
        new_col = ["Normalized TF"]
        df.insert(loc=idx, column='Document', value=new_col)
        print(df)
    return tf_doc


# Inverse Document frequency
def inverseDocumentFrequemcy(term, allDocuments):
    numDocumentsWithThisTerm = 0
    for doc in range(0, len(allDocuments)):
        if term in allDocuments[doc].split():
            numDocumentsWithThisTerm += 1
    if numDocumentsWithThisTerm > 0:
        return 1.0 + math.log(float(len(allDocuments)) / numDocumentsWithThisTerm)
    else:
        return 1.0


def compute_idf(documents):
    idf_dict = {}
    for doc in documents:
        sentence = doc.split()
        for word in sentence:
            idf_dict[word] = inverseDocumentFrequemcy(word, documents)
    print(idf_dict)
    return idf_dict


# tf_idf of  all document for the query
def compute_tfidf_with_alldocs(documents, query):
    tf_idf = []
    tf_doc = compute_normalizedtf(documents)
    idf_dict = compute_idf(documents)
    index = 0
    query_tokens = query.split()
    df = pd.DataFrame(columns=['doc']+query_tokens)
    for doc in documents:
        df['doc'] = np.arange(0, len(documents))
        doc_num = tf_doc[index]
        sentence = doc.split()
        for word in sentence:
            for text in query_tokens:
                if(text == word):
                    tf_idf_score = doc_num[word]*idf_dict[word]
                    tf_idf.append(tf_idf_score)
                    df.iloc[index, df.columns.get_loc(word)] = tf_idf_score
        index += 1
    df.fillna(0, axis=1, inplace=True)
    return tf_idf, df


# get the normalized termfrequency for the query
def compute_query_tf(query):
    query_norm_tf = {}
    tokens = query.split()
    for word in tokens:
        query_norm_tf[word] = termFrequency(word, query)
    return query_norm_tf


# inverse document frequency of the Query
def compute_query_idf(query, documents):
    idf_dict_qry = {}
    sentence = query.split()
    for word in sentence:
        idf_dict_qry[word] = inverseDocumentFrequemcy(word, documents)
    return idf_dict_qry


# tf_idf of the query input usser
def compute_query_tfidf(query, documents):
    tfidf_dict_query = {}
    query_norm_tf = compute_query_tf(query)
    idf_dict_qry = compute_query_idf(query, documents)
    sentence = query.split()
    for word in sentence:
        tfidf_dict_query[word] = query_norm_tf[word] * idf_dict_qry[word]
    return tfidf_dict_query


# cosine similarity of the query with overall documents
def cosine_similarity(tfidf_dict_qry, df, query, doc_num):
    dot_product = 0
    qry_mod = 0
    doc_mod = 0
    tokens = query.split()

    for keyword in tokens:
        dot_product += tfidf_dict_qry[keyword] * \
            df[keyword][df['doc'] == doc_num]
        # ||Query||
        qry_mod += tfidf_dict_qry[keyword] * tfidf_dict_qry[keyword]
        # ||Document||
        doc_mod += df[keyword][df['doc'] == doc_num] * \
            df[keyword][df['doc'] == doc_num]
    qry_mod = np.sqrt(qry_mod)
    doc_mod = np.sqrt(doc_mod)

    denominator = qry_mod*doc_mod
    cos_sim = dot_product/denominator
    return cos_sim


# flatten the array into one dimention
def flatten(lis):
    for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
            for x in flatten(item):
                yield x
        else:
            yield item


# ranking the similarity of the document with user query
def rank_similarity_docs(data, query):
    cos_sim = []
    tf_idf, df = compute_tfidf_with_alldocs(data, query)
    tfidf_dict_query = compute_query_tfidf(query, data)
    for doc_num in range(0, len(data)):
        cos_sim.append(cosine_similarity(
            tfidf_dict_query, df, query, doc_num).tolist())
    return cos_sim


# getting all the links from the database
def getLinks(data):
    news_links = []
    for n in data:
        new = n.news_link
        news_links.append(new)
    return news_links

# website credibility


def website_cred(news):
    gr = []

    news_link = getLinks(news)

    for nl in news_link:
        domain = tldextract.extract(nl)
        domain = domain.registered_domain
        API = '43e3f76942394e5b812d3821f7ac50a2'
        print('the registered domain is: ', domain)

        api = f"https://api.similarweb.com/v1/similar-rank/{domain}/rank?api_key={API}"
        data = requests.get(api).json()

        print(data)
        rank = data['similar_rank']
        frank = rank['rank']
        gr.append(frank)
        print('Global rank of the websource is: ', frank)

    return gr
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ end of  message credibility ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ labellization of the news ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def label(request):
    score = Scoring.objects.all()
    label = score
    print(label)
    return render(request, 'label.html', {'label': label})


def edit(request, id):
    object = Scoring.objects.get(id=id)
    return render(request, 'edit.html', {'object': object})


def update(request, id):
    object = Scoring.objects.get(id=id)
    # form = scoreform(request.POST, instance=object)
    if request.method == 'POST':
        label = request.POST['label']
        if label != '':
            object.l = label
            object.save()

            msg = (id, ' news labelling succed')
            messages.info(request, msg)
            return redirect('label')
        else:
            messages.info(request, 'some input is empty, please check again')
            return redirect('edit')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End of labellization of the news ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def classify(request):
    objects = Scoring.objects.all()  # .order_by('-id')
    score_dataframe = pd.DataFrame.from_records(objects.values())
    print(score_dataframe)
    X = score_dataframe.drop(['l', 'nTitle', 'id'], axis=1)
    kmeans = KMeans(2)
    kmeans.fit(X)
    identified_cluster = kmeans.fit_predict(X)
    print('clustering result: ', identified_cluster)
    # y = score_dataframe['l']
    # y = pd.DataFrame(identified_cluster, columns=['l'])
    y = identified_cluster
    print(X)
    print(y)
    acc = 0
    prec = 0
    recl = 0
    list1 = objects.values()
    mylist = zip(list1, y)
    res_tcMcWc = []
    p = PrettyTable()
    try:
        # split the data into 70% training and 30% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=0)
    # creating the svm classifier
        clf = svm.SVC(kernel='rbf')
    # training the model using training sets
        clf.fit(X_train, y_train)
    # predict the answer for test sets
        y_pred = clf.predict(X_test)
    # print example
        print('training sample: \n', X_train)
        print('testing sample: \n', X_test)
    # evaluation of the model getting acuracy, precision and recall
        print('Confussion matrix result: ', confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recl = recall_score(y_test, y_pred)
        res_tcMcWc.append(prec)
        res_tcMcWc.append(recl)
        res_tcMcWc.append(acc)
        # parsing testing dataset into json object
        json_records = X_test.reset_index().to_json(orient='records')
        data = []
        data = json.loads(json_records)
        # parsing Ytesting and yPredict into Json object
        df = pd.DataFrame({'RealValues': y_test, 'PredictedValues': y_pred})
        json_records = df.reset_index().to_json(orient='records')
        dt = []
        dt = json.loads(json_records)
        class_result = zip(data, dt)
        # return res_tcMcWc
    except:
        messages.info(
            request, ' The testing data is unbalanced, \n the classifier detect only one class of data.')
    ftc = Vtc(objects)
    fmc = Vmc(objects)
    fwc = Vwc(objects)
    ftcMc = tcMc(objects)
    ftcWc = tcWc(objects)
    fmcWc = mcWc(objects)
    ftcMcWc = res_tcMcWc
    ft = [['TC', 'Mc', 'Wc', 'Tc&Mc', 'Tc&Wc', 'Mc&Wc', 'TcMcWc']]
    rows = ['Precision', 'Recall', 'Accuracy']
    for r, tc, mc, wc, tm, tw, mw, tmw in zip(rows, ftc, fmc, fwc, ftcMc, ftcWc, fmcWc, ftcMcWc):
        feature = r
        vtc = tc
        vmc = mc
        vwc = wc
        vtm = tm
        vtw = tw
        vmw = mw
        vtmw = tmw
        # ft.append((str(feature), str(vtc), str(vmc), str(
        #     vwc), str(vtm), str(vtw), str(vmw), str(vtmw)))
        p.add_row([feature, vtc, vmc, vwc, vtm, vtw, vmw, vtmw])

    p.field_names = ["Feature", "TC", "MC", "Wc",
                     "Tc&Mc", "Tc&Wc", "Mc&Wc", "TcMcWc"]
    print(p)
    return render(request, 'classify.html', {'acc': acc, 'prec': prec, 'recl': recl, 'mylist': mylist, 'testing': class_result})


def reset(request):
    Keyword.objects.all().delete()
    News.objects.all().delete()
    Scoring.objects.all().delete()

    return redirect('/')


# code upgrade after Thesis exam
def Vtc(objects):
    score_dataframe = pd.DataFrame.from_records(objects.values())
    X = score_dataframe.drop(['nTitle', 'l', 'id', 'mc', 'gr'], axis=1)
    kmeans = KMeans(2)
    kmeans.fit(X)
    result_cluster = kmeans.fit_predict(X)
    y = result_cluster
    res_tc = []
    try:
        # split the data into 70% training and 30% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=0)
    # creating the svm classifier
        clf = svm.SVC(kernel='rbf')
    # training the model using training sets
        clf.fit(X_train, y_train)
    # predict the answer for test sets
        y_pred = clf.predict(X_test)
        print('Confusion matrix for Time credibility only:\n')
        print('Confussion matrix result: ', confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recl = recall_score(y_test, y_pred)
        res_tc.append(prec)
        res_tc.append(recl)
        res_tc.append(acc)
    except:
        print('data cannot be predicted')
    return res_tc
# get message credibility only


def Vmc(objects):
    score_dataframe = pd.DataFrame.from_records(objects.values())
    X = score_dataframe.drop(['nTitle', 'l', 'id', 'tc', 'gr'], axis=1)
    kmeans = KMeans(2)
    kmeans.fit(X)
    result_cluster = kmeans.fit_predict(X)
    y = result_cluster
    res_mc = []
    try:
        # split the data into 70% training and 30% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=0)
    # creating the svm classifier
        clf = svm.SVC(kernel='rbf')
    # training the model using training sets
        clf.fit(X_train, y_train)
    # predict the answer for test sets
        y_pred = clf.predict(X_test)
        print('confusion matrix for Message credibility only: \n')
        print('Confussion matrix result: ', confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recl = recall_score(y_test, y_pred)
        res_mc.append(prec)
        res_mc.append(recl)
        res_mc.append(acc)
    except:
        print('data cannot be predicted')
    return res_mc

    # getting wc only


def Vwc(objects):
    score_dataframe = pd.DataFrame.from_records(objects.values())
    X = score_dataframe.drop(['nTitle', 'l', 'id', 'tc', 'mc'], axis=1)
    kmeans = KMeans(2)
    kmeans.fit(X)
    result_cluster = kmeans.fit_predict(X)
    y = result_cluster
    res_wc = []
    try:
        # split the data into 70% training and 30% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=0)
        # creating the svm classifier
        clf = svm.SVC(kernel='rbf')
        # training the model using training sets
        clf.fit(X_train, y_train)
        # predict the answer for test sets
        y_pred = clf.predict(X_test)
        print('Confusion matrix for Website credibility: \n')
        print('Confussion matrix result: ', confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recl = recall_score(y_test, y_pred)
        res_wc.append(prec)
        res_wc.append(recl)
        res_wc.append(acc)
    except:
        print('data cannot be predicted')
    return res_wc


def tcMc(objects):
    score_dataframe = pd.DataFrame.from_records(objects.values())
    X = score_dataframe.drop(['nTitle', 'l', 'id', 'gr'], axis=1)
    kmeans = KMeans(2)
    kmeans.fit(X)
    result_cluster = kmeans.fit_predict(X)
    y = result_cluster
    res_tcMc = []
    try:
        # split the data into 70% training and 30% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=0)
        # creating the svm classifier
        clf = svm.SVC(kernel='rbf')
        # training the model using training sets
        clf.fit(X_train, y_train)
        # predict the answer for test sets
        y_pred = clf.predict(X_test)
        print('confusion matrix for TC and MC')
        print('Confussion matrix result: ', confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recl = recall_score(y_test, y_pred)
        res_tcMc.append(prec)
        res_tcMc.append(recl)
        res_tcMc.append(acc)
    except:
        print('data cannot be predicted')
    return res_tcMc


def tcWc(objects):
    score_dataframe = pd.DataFrame.from_records(objects.values())
    X = score_dataframe.drop(['nTitle', 'l', 'id', 'mc'], axis=1)
    kmeans = KMeans(2)
    kmeans.fit(X)
    result_cluster = kmeans.fit_predict(X)
    y = result_cluster
    res_tcWc = []
    try:
        # split the data into 70% training and 30% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=0)
        # creating the svm classifier
        clf = svm.SVC(kernel='rbf')
        # training the model using training sets
        clf.fit(X_train, y_train)
        # predict the answer for test sets
        y_pred = clf.predict(X_test)
        print('confusion matrix for TC & WC: \n')
        print('Confussion matrix result: ', confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recl = recall_score(y_test, y_pred)
        res_tcWc.append(prec)
        res_tcWc.append(recl)
        res_tcWc.append(acc)
    except:
        print('data cannot be predicted')
    return res_tcWc


def mcWc(objects):
    score_dataframe = pd.DataFrame.from_records(objects.values())
    X = score_dataframe.drop(['nTitle', 'l', 'id', 'tc'], axis=1)
    kmeans = KMeans(2)
    kmeans.fit(X)
    result_cluster = kmeans.fit_predict(X)
    y = result_cluster
    res_mcWc = []
    try:
        # split the data into 70% training and 30% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=0)
        # creating the svm classifier
        clf = svm.SVC(kernel='rbf')
        # training the model using training sets
        clf.fit(X_train, y_train)
        # predict the answer for test sets
        y_pred = clf.predict(X_test)
        print('Confusion matrix for MC & WC: \n')
        print('Confussion matrix result: ', confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        recl = recall_score(y_test, y_pred)
        res_mcWc.append(prec)
        res_mcWc.append(recl)
        res_mcWc.append(acc)
    except:
        print('data cannot be predicted')
    return res_mcWc
