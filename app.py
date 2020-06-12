import nltk
nltk.download('punkt')
nltk.download('wordnet')
import numpy
import random
from nltk.stem import WordNetLemmatizer
stemmer=WordNetLemmatizer()
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import requests
import datetime as dt
import pandas_datareader.data as web

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
symbols=pd.read_csv('NASDAQ.txt', sep='\t', names=['Symbol', 'Company'])

def get_name(msg):
    for i in range(len(symbols['Symbol'])):
        if msg==symbols['Symbol'][i]:
            return symbols['Company'][i]
    else:
        return msg

def pct_change(open_price,current_price):
    pct = ((current_price-open_price)/open_price)*100
    return float(pct)

def get_details(msg):
    start_date = dt.datetime(2015, 1, 1)
    end_date = dt.datetime.now()
    df = web.DataReader(msg, 'yahoo', start_date, end_date)
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)
    return jsonify(
        name=get_name(msg),
        website='https://amstock.herokuapp.com/%' + msg,
        close=float(df['Close'][-1]),
        open=float(df['Open'][-1]),
        high=float(df['High'][-1]),
        low=float(df['Low'][-1]),
        volume=float(df['Volume'][-1]),
        percent=pct_change(df['Open'][-1], df['Close'][-1])
    )

def check_symbol(sym):
    for i in symbols['Symbol']:
        if sym==i:
            return True

def check_fake(msg):
    r = requests.post(url='https://amfakeclass.herokuapp.com/predict/'+msg)
    if r.text=='True':
        return 'Fake'
    else:
        return 'Real'

def check_spam(msg):
    r = requests.post(url='https://amspamclass.herokuapp.com/predict/' + msg)
    if r.text == 'True':
        return True
    else:
        return False

def check(msg):
    if msg[0]=='_':
        if 'stock' in msg[1:]:
            return 'Stock'
        elif 'fake' in msg[1:]:
            return 'Fake'
        elif 'spam'in msg[1:]:
            return 'Spam'
    else:
        return 'Chat'

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.lemmatize(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

@app.route('/predict/<msg>',methods=['GET','POST'])
def predict(msg):

    if check(msg)=='Chat':
        data = pd.read_json('chat_intents.json')
        words = []
        labels = []
        docs_x = []
        docs_y = []

        stemmer = WordNetLemmatizer()

        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                wrds = nltk.word_tokenize(pattern)
                words.extend(wrds)
                docs_x.append(wrds)
                docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                labels.append(intent["tag"])

        words = [stemmer.lemmatize(w.lower()) for w in words if w != "?"]
        words = sorted(list(set(words)))

        labels = sorted(labels)
        results = model.predict([bag_of_words(msg, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']

        return (random.choice(responses))
    elif check(msg)=='Stock':
        if check_symbol(msg[6:]):
            return get_details(msg[6:])
    elif check(msg)=='Fake':
        if check_fake(msg):
            return ('True')
        else:
            return ('False')
    elif check(msg)=='Spam':
        if check_spam(msg):
            return ('Spam')
        else:
            return ('Ham')

if __name__ == "__main__":
    app.run(debug=True)
