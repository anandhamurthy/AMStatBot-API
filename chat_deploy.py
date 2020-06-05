import nltk
import numpy
import random
from nltk.stem import WordNetLemmatizer
stemmer=WordNetLemmatizer()
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.lemmatize(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

@app.route('/predict/<msg>')
def predict(msg):
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

if __name__ == "__main__":
    app.run(debug=True)