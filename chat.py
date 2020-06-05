import nltk
#import tflearn
from nltk.stem import WordNetLemmatizer
import numpy
#import tensorflow as tf
import random
import pandas as pd

data=pd.read_json('chat_intents.json')
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
print(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.lemmatize(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

from sklearn.tree import DecisionTreeRegressor

model=DecisionTreeRegressor()
model.fit(training,output)
print(model)
import pickle
pickle.dump(model,open('model.pkl','wb'))


# tf.reset_default_graph()
#
# net = tflearn.input_data(shape=[None, len(training[0])])
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, 8)
# net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
# net = tflearn.regression(net)
#
# model = tflearn.DNN(net)
# model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
#
# # model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
# # model.save("model.tflearn")
#
# try:
#     model.load("model.tflearn")
# except:
#     model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
#     model.save("model.tflearn")
#
#
