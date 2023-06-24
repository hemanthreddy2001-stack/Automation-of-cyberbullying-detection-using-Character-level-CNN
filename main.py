# -*- coding: utf-8 -*-
"""
Created on Fri Apr 16 14:54:35 2021

@author: mahesh pala
"""

from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
import numpy as np
import os

from string import punctuation
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model
from keras.utils.np_utils import to_categorical
import pickle
from keras.models import model_from_json

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from nltk.corpus import wordnet
from keras.models import Sequential
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix


main = tkinter.Tk()
main.title("Cyberbullying Detection Character Level") #designing main screen
main.geometry("1300x1200")

global model
global filename
global word_precision, word_recall, word_fmeasure, word_accuracy
global char_precision, char_recall, char_fmeasure, char_accuracy
# global synonym_precision, synonym_recall, synonym_fmeasure, synonym_accuracy
word_trainX = []
word_trainy = []
char_trainX = []
char_trainy = []
global word_model
global char_model
global word_tokenizer
global char_tokenizer
global word_length
global char_length
global bullying_counts
global non_bullying_counts

def uploadDataset(): #function to upload tweeter profile
    text.delete('1.0', END)
    global filename
    filename = filedialog.askopenfilename(initialdir = "dataset")
    text.delete('1.0', END)
    text.insert(END,filename+' dataset loaded\n')

def clean_doc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens)
    #print(tokens)
    return tokens

def getCharacters(tokens):
    chars = list(tokens)    #char base uncomment
    tokens = ' '.join(chars)#char based uncomment
    #print(tokens)
    return tokens

def cleancodeVec():
    global word_trainX
    global word_trainy
    global char_trainX
    global char_trainy
    global bullying_counts
    global non_bullying_counts
    word_trainX.clear()
    word_trainy.clear()
    char_trainX.clear()
    char_trainy.clear()
    bullying_counts = 0
    non_bullying_counts = 0

    train = pd.read_csv(filename,encoding='iso-8859-1',sep='\t')
    for i in range(len(train)):
        label = train.get_value(i, 'label_bullying')
        msg = train.get_value(i, 'text_message')
        msg = msg.strip()
        tokens = clean_doc(msg)
        if len(tokens) > 0:
            word_trainX.append(tokens)
            word_trainy.append(label)
            char_trainy.append(label)
            characters = getCharacters(tokens)
            char_trainX.append(characters)
            if label == 0:
                non_bullying_counts = non_bullying_counts + 1
            if label == 1:
                bullying_counts = bullying_counts + 1

    word_trainy = to_categorical(word_trainy)
    char_trainy = to_categorical(char_trainy)
    text.delete('1.0', END)
    text.insert(END,"Length of text messages for word array : "+str(len(word_trainX))+"\n")
    text.insert(END,"Length of text messages for char array : "+str(len(char_trainX))+"\n")

# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])

# encode a list of lines
def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    return padded
    
    
def create_tokenizer(lines):
    word_tokenizer = Tokenizer()
    word_tokenizer.fit_on_texts(lines)
    return word_tokenizer


def calculateWordMetrics():
    global word_precision, word_recall, word_fmeasure, word_accuracy
    testX = []
    y_test = []
    for i in range(0,2000):
        testX.append(word_trainX[i])
    for i in range(0,2000):
        y_test.append(word_trainy[i])
    y_test = np.array(y_test)
    y_test = np.argmax(y_test, axis=1)
    testX = np.array(testX, dtype='float32')
    print(str(word_trainX.shape)+" "+str(testX.shape))
    y_predicted = word_model.predict(testX, batch_size=128)
    y_predicted = np.argmax(y_predicted, axis=1)
    word_precision = precision_score(y_test, y_predicted,average='macro') * 100
    word_recall = recall_score(y_test, y_predicted,average='macro') * 100
    word_fmeasure = f1_score(y_test, y_predicted,average='macro') * 100
    
    f = open('WordModel/history.pckl', 'rb')
    acc = pickle.load(f)
    f.close()
    acc = acc['accuracy']
    word_accuracy = np.amax(acc) * 100

    test_bullying = 0
    test_non_bullying = 0
    for i in range(len(y_predicted)):
        if y_predicted[i] == 0:
            test_non_bullying = test_non_bullying + 1
        if y_predicted[i] == 1:
            test_bullying = test_bullying + 1    

    # cm = confusion_matrix(y_test,y_predicted)
    #text.delete('1.0', END)
    text.insert(END,'\n\nWord Based Accuracy  : '+str(word_accuracy)+'\n')
    text.insert(END,'Word Based Precision : '+str(word_precision)+'\n')
    text.insert(END,'Word Based Recall    : '+str(word_recall)+'\n')
    text.insert(END,'Word Based FMeasure  : '+str(word_fmeasure)+"\n")
    text.insert(END,'Train Non Cyber Bullying Count   : '+str(non_bullying_counts)+"\n")
    text.insert(END,'Train Cyber Bullying Count       : '+str(bullying_counts)+"\n")
    text.insert(END,'Test Non Cyber Bullying Count    : '+str(test_non_bullying)+"\n")
    text.insert(END,'Test Cyber Bullying Count        : '+str(test_bullying)+"\n")
       
def wordCNN():
    global word_precision, word_recall, word_fmeasure, word_accuracy
    global word_model
    global word_length
    global word_tokenizer
    global word_trainX
    global word_trainy
    
    text.delete('1.0', END)
    word_tokenizer = create_tokenizer(word_trainX)
    word_length = max_length(word_trainX)
    vocab_size = len(word_tokenizer.word_index) + 1
    word_trainX = np.asarray(word_trainX)
    word_trainX = encode_text(word_tokenizer, word_trainX, word_length)
    print("==================="+str(word_trainX.shape))
    text.insert(END,'Max document length : '+str(word_trainX.shape[0])+"\n")
    text.insert(END,'Word Vocabulary size : '+str(word_trainX.shape[1])+"\n")
    text.insert(END,'Documents & Word Length : '+str(word_trainX.shape)+"\n")
    

    if os.path.exists('WordModel/model.json'):
        with open('WordModel/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            word_model = model_from_json(loaded_model_json)
        word_model.load_weights("WordModel/model_weights.h5")
        word_model._make_predict_function()   
        print(word_model.summary())
        calculateWordMetrics()
        text.insert(END,'Word Based CNN Model Generated. See black console to view layers of CNN')
    else:
        word_model = Sequential()
        word_model.add(Dense(512, input_shape=(length,)))
        word_model.add(Activation('relu'))
        word_model.add(Dropout(0.3))
        word_model.add(Dense(512))
        word_model.add(Activation('relu'))
        word_model.add(Dropout(0.3))
        word_model.add(Dense(2))
        word_model.add(Activation('softmax'))
        word_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(word_model.summary())
        #model.fit([trainX,trainX,trainX], array(trainy), epochs=10, batch_size=16,validation_split=0.2, shuffle=True, verbose=2)#best work
        hist = word_model.fit(trainX, trainy, epochs=10, batch_size=128,validation_split=0.2, shuffle=True, verbose=2)
        text.insert(END,"Word Based CNN Model Generated. See black console to view layers of CNN")
        f = open('WordModel/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        word_model.save_weights('WordModel/model_weights.h5')            
        model_json = word_model.to_json()
        with open("WordModel/model.json", "w") as json_file:
            json_file.write(model_json)
        calculateWordMetrics()

def calculateCharMetrics():
    global char_precision, char_recall, char_fmeasure, char_accuracy
    testX = []
    y_test = []
    for i in range(0,2000):
        testX.append(char_trainX[i])
    for i in range(0,2000):
        y_test.append(char_trainy[i])
    y_test = np.array(y_test)
    y_test = np.argmax(y_test, axis=1)
    testX = np.array(testX, dtype='float32')
    print(str(char_trainX.shape)+" "+str(testX.shape))
    y_predicted = char_model.predict(testX, batch_size=128)
    y_predicted = np.argmax(y_predicted, axis=1)
    char_precision = precision_score(y_test, y_predicted,average='macro') * 100
    char_recall = recall_score(y_test, y_predicted,average='macro') * 100
    char_fmeasure = f1_score(y_test, y_predicted,average='macro') * 100
    char_accuracy = accuracy_score(y_test,y_predicted)*100

    test_bullying = 0
    test_non_bullying = 0
    for i in range(len(y_predicted)):
        if y_predicted[i] == 0:
            test_non_bullying = test_non_bullying + 1
        if y_predicted[i] == 1:
            test_bullying = test_bullying + 1    
    
    #text.delete('1.0', END)
    text.insert(END,'\n\nChar Based Accuracy      : '+str(char_accuracy)+'\n')
    text.insert(END,'Char Based Precision         : '+str(char_precision)+'\n')
    text.insert(END,'Char Based Recall            : '+str(char_recall)+'\n')
    text.insert(END,'Char Based FMeasure          : '+str(char_fmeasure)+"\n")
    text.insert(END,'Train Non Cyber Bullying Count   : '+str(non_bullying_counts)+"\n")
    text.insert(END,'Train Cyber Bullying Count       : '+str(bullying_counts)+"\n")
    text.insert(END,'Test Non Cyber Bullying Count    : '+str(test_non_bullying)+"\n")
    text.insert(END,'Test Cyber Bullying Count        : '+str(test_bullying)+"\n")


def create_char_tokenizer(lines):
    char_tokenizer = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    char_tokenizer.fit_on_texts(lines)
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1
    char_tokenizer.word_index = char_dict.copy() 
    char_tokenizer.word_index[char_tokenizer.oov_token] = max(char_dict.values()) + 1
    return char_tokenizer    
    
    
def charCNN():
    global char_precision, char_recall, char_fmeasure, char_accuracy
    global char_model
    global char_length
    global char_tokenizer
    global char_length
    global char_trainX
    global char_trainy
    text.delete('1.0', END)

    char_tokenizer = create_char_tokenizer(char_trainX)
    char_length = max_length(char_trainX)
    vocab_size = len(char_tokenizer.word_index) + 1
    char_trainX = encode_text(char_tokenizer, char_trainX, char_length)
    char_trainX = np.array(char_trainX, dtype='float32')
    text.insert(END,'Max document length : '+str(char_trainX.shape[0])+"\n")
    text.insert(END,'Char Vocabulary size : '+str(char_trainX.shape[1])+"\n")
    text.insert(END,'Documents & Word Length : '+str(char_trainX.shape)+"\n")
    vocab_size = len(char_tokenizer.word_index)

    if os.path.exists('CharModel/model.json'):
        with open('CharModel/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            char_model = model_from_json(loaded_model_json)
        char_model.load_weights("CharModel/model_weights.h5")
        char_model._make_predict_function()   
        print(char_model.summary())
        calculateCharMetrics()
        text.insert(END,'Char Based CNN Model Generated. See black console to view layers of CNN')
    else:
        embedding_weights = [] #(71, 70)
        embedding_weights.append(np.zeros(vocab_size)) # first row is pad
        for char, i in char_tokenizer.word_index.items(): # from index 1 to 70
            onehot = np.zeros(vocab_size)
            onehot[i-1] = 1
            embedding_weights.append(onehot)
        embedding_weights = np.array(embedding_weights)
        input_size = char_length
        embedding_size = 69
        conv_layers = [[256, 7, 3], [256, 7, 3], [256, 3, -1], [256, 3, -1], [256, 3, -1], [256, 3, 3]]
        fully_connected_layers = [1024, 1024]
        num_of_classes = 2
        dropout_p = 0.5
        optimizer = 'adam'
        loss = 'categorical_crossentropy'
        embedding_layer = Embedding(vocab_size+1, embedding_size+1, input_length=input_size, weights=[embedding_weights])
        inputs = Input(shape=(input_size,), name='input', dtype='int64')  # shape=(?, 1014)
        x = embedding_layer(inputs)
        for filter_num, filter_size, pooling_size in conv_layers:
            x = Conv1D(filter_num, filter_size)(x) 
            x = Activation('relu')(x)
            if pooling_size != -1:
                x = MaxPooling1D(pool_size=pooling_size)(x) # Final shape=(None, 34, 256)
        x = Flatten()(x) # (None, 8704)
        for dense_size in fully_connected_layers:
            x = Dense(dense_size, activation='relu')(x) # dense_size == 1024
            x = Dropout(dropout_p)(x)
        predictions = Dense(num_of_classes, activation='softmax')(x)
        char_model = Model(inputs=inputs, outputs=predictions)
        char_model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) # Adam, categorical_crossentropy
        print(char_model.summary())
        indices = np.arange(char_trainX.shape[0])
        np.random.shuffle(indices)
        x_train = char_trainX[indices]
        y_train = char_trainy[indices]
        hist = char_model.fit(x_train, y_train, validation_split=0.2, batch_size=128, epochs=10, verbose=1)
        f = open('CharModel/history.pckl', 'wb')
        pickle.dump(hist.history, f)
        f.close()
        char_model.save_weights('CharModel/model_weights.h5')
        model_json = char_model.to_json()
        with open("CharModel/model.json", "w") as json_file:
            json_file.write(model_json)
        calculateCharMetrics()
        text.insert(END,'Char Based CNN Model Generated. See black console to view layers of CNN')
    

def extension_clean_doc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens) #here upto for word based
    arr = tokens.split(" ")
    allwords = ''
    dup = []
    for i in range(len(arr)):
        arr[i] = arr[i].lower().strip()
        if arr[i] not in dup:
            dup.append(arr[i])
            for syn in wordnet.synsets(arr[i]):
                for lm in syn.lemmas():
                    if lm.name not in dup:
                        dup.append(lm.name)
                        allwords+=lm.name()+" "
    allwords = allwords.strip();
    print(tokens+" === "+allwords)
    chars = list(allwords)    #char base uncomment
    tokens = ' '.join(chars)#char based uncomment
    #print(str(tokens))
    return tokens


def process_text(textdata):
    documents = list()
    data = clean_doc(textdata)
    data = getCharacters(data)
    documents.append(data)
    return documents

def predictBullying():
    text.delete('1.0', END)
    input_sentence = simpledialog.askstring("Enter your sentence here  to detect cyberbullying", "Enter your sentence here to detect cyberbullying")
    testX = process_text(input_sentence)
    print(testX)
    testX = encode_text(char_tokenizer, testX, char_length)
    print(testX)
    ypred = char_model.predict(testX)
    predict = np.argmax(ypred)
    if predict == 0:
        text.insert(END,input_sentence+' DOES NOT CONTAINS Cyberbullying Words')
    else:
        text.insert(END,input_sentence+' CONTAINS Cyberbullying Words')
        
def accuracyGraph():
    f = open('WordModel/history.pckl', 'rb')
    word_loss = pickle.load(f)
    f.close()

    f = open('CharModel/history.pckl', 'rb')
    char_loss = pickle.load(f)
    f.close()

    wordloss = word_loss['accuracy']
    charloss = char_loss['accuracy']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(wordloss, 'ro-', color = 'indigo')
    plt.plot(charloss, 'ro-', color = 'green')
    plt.legend(['Word Based CNN Accuracy', 'Char Based CNN Accuracy'], loc='upper left')
    # plt.xticks(wordloss.index)
    plt.title('Word Vs Char Accuracy Comparison Graph')
    plt.show()

def precisionGraph():
    height = [word_precision,char_precision]
    bars = ('Word Based Precision','Char Based Precision')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def recallGraph():
    height = [word_recall,char_recall]
    bars = ('Word Based Recall','Char Based Recall')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def measureGraph():
    height = [word_fmeasure,char_fmeasure]
    bars = ('Word Based FMeasure','Char Based FMeasure')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

def lossGraph():
    f = open('WordModel/history.pckl', 'rb')
    word_loss = pickle.load(f)
    f.close()

    f = open('CharModel/history.pckl', 'rb')
    char_loss = pickle.load(f)
    f.close()

    wordloss = word_loss['val_loss']
    charloss = char_loss['val_loss']
    plt.figure(figsize=(10,6))
    plt.grid(True)
    plt.xlabel('Epoch')
    plt.ylabel('Focal Loss')
    plt.plot(wordloss, 'ro-', color = 'indigo')
    plt.plot(charloss, 'ro-', color = 'green')
    # plt.xticks(wordloss.index)
    plt.legend(['Word Based CNN Loss', 'Char Based CNN Loss'], loc='upper left')
    plt.title('Word Vs Char Focal Loss Comparison Graph')
    plt.show()
    
    

font = ('times', 16, 'bold')
title = Label(main, text='Cyberbullying Detection in Social Media Text Based on Character-Level Convolutional Neural Network with Shortcuts')
title.config(bg='lavender', fg='tomato')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Text Dataset", command=uploadDataset, bg='#ffb3fe')
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

trainButton = Button(main, text="Clean & Convert Text to Code Vector", command=cleancodeVec, bg='#ffb3fe')
trainButton.place(x=240,y=550)
trainButton.config(font=font1) 

wordButton = Button(main, text="Generate Word CNN", command=wordCNN, bg='#ffb3fe')
wordButton.place(x=560,y=550)
wordButton.config(font=font1) 

charButton = Button(main, text="Generate Char CNN Model", command=charCNN, bg='#ffb3fe')
charButton.place(x=760,y=550)
charButton.config(font=font1) 

accuracyButton = Button(main, text="Accuracy Comparison", command=accuracyGraph, bg='#ffb3fe')
accuracyButton.place(x=1000,y=550)
accuracyButton.config(font=font1) 

precisionButton = Button(main, text="Precision Comparison", command=precisionGraph, bg='#ffb3fe')
precisionButton.place(x=50,y=600)
precisionButton.config(font=font1)

recallButton = Button(main, text="Recall Comparison", command=recallGraph, bg='#ffb3fe')
recallButton.place(x=260,y=600)
recallButton.config(font=font1)

measureButton = Button(main, text="FMeasure Comparison", command=measureGraph, bg='#ffb3fe')
measureButton.place(x=450,y=600)
measureButton.config(font=font1)

lossButton = Button(main, text="Focal Loss Comparison", command=lossGraph, bg='#ffb3fe')
lossButton.place(x=670,y=600)
lossButton.config(font=font1)

predictButton = Button(main, text="Predict Bullying from Text", command=predictBullying, bg='#ffb3fe')
predictButton.place(x=900,y=600)
predictButton.config(font=font1)

main.config(bg='palegreen')
main.mainloop()



