from flask import Flask, render_template, session, redirect, url_for, session






#Note que cada uma das intenções possui:
#tag: um nome único;
#patterns: sequência de frases ditas pelo usuário;
#responses: respostas possíveis para o chatbot.
#Feito isso o próximo passo é importar nossas dependências:
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
#import tensorflow as tf
from tensorflow.python.framework import ops
import random

import json
with open('intents/intents.json') as file:
    data = json.load(file)








# Com nosso arquivo JSON devidamente carregado,
# podemos então começar a organiza-lo separando
# os documentos, respostas e classes.

words = []
classes = []
documents = []
ignore_words = ['?']


# loop through each sentence in our intents patterns
for intent in data['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        # add to our words list
        words.extend(w)
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicates
classes = sorted(list(set(classes)))

print (len(documents), "documents")
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)






# Note que nessa parte também removemos algumas
# duplicadas com a finalidade deixar a base limpa.
# Nosso próximo passo agora é transformar esses
# documentos em dados numéricos para o tensorflow.
# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    # create our bag of words array
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])








# Observando atentamente repare que transformamos os textos em duas
# listas (intenção, classe) que servirão de treinamento para nossa
# rede neural, com os seguintes parâmetros:
#       entrada (input): será o vetor train_x;
#       camadas ocultas: 2 com 8 nós ;
#       saída (output): train_y;
#       épocas: 1000 (aumente caso necessário)


# reset underlying graph data
# tf.reset_default_graph()
ops.reset_default_graph()


# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# Start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')











# Finalizado o treinamento devemos salvar nosso modelo para a parte 2
# desse tutorial onde faremos a predição da classe e das respostas do chatbot.

# save all of our data structures
import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open( "training_data", "wb" ) )







# Para testar o modelo treinado iremos realizar uma pequena demonstração
# de predição que será trabalhada na parte 2 desse post.
def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

p = bow("Voces aceitam cartao de credito?", words)
print (p)
print (classes)

#print prediction 
print(model.predict([p]))






















































app = Flask(__name__)
# Configure a secret SECRET_KEY
# We will later learn much better ways to do this!!
app.config['SECRET_KEY'] = 'someRandomKey'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
#flower_model = load_model("final_iris_model.h5")
#flower_scaler = joblib.load("iris_scaler.pkl")



@app.route('/', methods=['GET', 'POST'])
def index():
    return 'index page'


@app.route('/prediction')
def prediction():
    return 'prediction page'


if __name__ == '__main__':
    app.run(debug=True)
