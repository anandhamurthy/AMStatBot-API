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

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
symbols=pd.read_csv('tickers.csv')

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
    r = requests.post(url='https://amspamclaa.herokuapp.com/predict/' + msg)
    if r.text == 'True':
        return 'Spam'
    else:
        return 'Ham'

def check(msg):
    if msg[0]=='%':
        if check_symbol(msg[1:]):
            return 'Stock'
        else:
            return 'Invalid Symbol'
    elif msg[0]=='_':
        if 'fake' in msg[1:]:
            if check_fake(msg[5:]):
                return 'Fake'
            else:
                return 'Real'
        elif 'spam'in msg[1:]:
            if check_fake(msg[5:]):
                return 'Fake'
            else:
                return 'Real'
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

@app.route('/predict/<msg>',methods=['POST'])
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
        return ('https://amstock.herokuapp.com/'+msg[1:])
    elif check(msg)=='Fake':
        if check_fake(msg):
            return ('True')
        else:
            return ('False')
    elif check(msg)=='Spam':
        if check_spam(msg):
            return ('True')
        else:
            return ('False')

if __name__ == "__main__":
    app.run(debug=True)
