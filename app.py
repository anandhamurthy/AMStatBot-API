import pandas_datareader.data as web
import datetime as dt
import requests
import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template
from nltk.stem import WordNetLemmatizer
import random
import numpy
import nltk
from flask_cors import CORS

nltk.download('punkt')
nltk.download('wordnet')
stemmer = WordNetLemmatizer()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
                             #['localhost', 'https://amstatbot.netlify.app']}})

model = pickle.load(open('model.pkl', 'rb'))
symbols = pd.read_csv('Symbols.csv')


def get_name(msg):
    for i in range(len(symbols['Symbol'])):
        if msg == symbols['Symbol'][i]:
            return symbols['Company'][i]
    else:
        return msg


def pct_change(open_price, current_price):
    pct = ((current_price-open_price)/open_price)*100
    return float(pct)


def get_details(msg):
    start_date = dt.datetime(2015, 1, 1)
    end_date = dt.datetime.now()
    df = web.DataReader(msg, 'yahoo', start_date, end_date)
    df.reset_index(inplace=True)
    df.set_index("Date", inplace=True)

    stock_response = {
        "name": get_name(msg),
        "website": 'https://amstock.herokuapp.com/' + msg,
        "close": float(df['Close'][-1]),
        "open": float(df['Open'][-1]),
        "high": float(df['High'][-1]),
        "low": float(df['Low'][-1]),
        "volume": float(df['Volume'][-1]),
        "percent": pct_change(df['Close'][-2], df['Close'][-1]),
        "symbol": msg
    }

    response = {
        'status': 200,
        'message': 'OK',
        'response': stock_response

    }
    return jsonify(response)


def check_symbol(sym):
    for i in symbols['Symbol']:
        if sym == i:
            return True


# def check(msg):
#     if msg[0] == '_':
#         if 'stock' in msg[1:]:
#             return 'Stock'
#         elif 'fake' in msg[1:]:
#             return 'Fake'
#         elif 'spam' in msg[1:]:
#             return 'Spam'
#     else:
#         return 'Chat'


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.lemmatize(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    msg = request.form.get('message', type=str)
    type_id = request.form.get('type', type=int)
    print(msg, type_id)
    if type_id == 1:
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

        response = {
            'status': 200,
            'message': 'OK',
            'response': random.choice(responses)
        }
        return jsonify(response)
    elif type_id == 2:
        if check_symbol(msg[6:]):
            return get_details(msg[6:])
        else:
            return 'Invalid'


if __name__ == "__main__":
    app.run(debug=False)
