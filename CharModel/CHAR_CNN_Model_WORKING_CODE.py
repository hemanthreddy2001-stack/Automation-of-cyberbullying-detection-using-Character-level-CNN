from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
from string import punctuation
from os import listdir
from nltk.corpus import stopwords
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from keras.layers import Input, Embedding, Activation, Flatten, Dense
from keras.layers import Conv1D, MaxPooling1D, Dropout
from keras.models import Model
from keras.utils.np_utils import to_categorical
import pickle
import matplotlib.pyplot as plt
from keras.models import model_from_json

main = tkinter.Tk()
main.title("Convolutional Neural Network Based Text Steganalysis")
main.geometry("1300x1200")

global filename
trainX = []
trainy = []
dup = []
global testX
global model
global tokenizer
global length
global accuracy

def uploadDataset():
    global filename
    filename = filedialog.askopenfilename(initialdir="dataset")
    pathlabel.config(text=filename)
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");
    

    


def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text

# turn a doc into clean tokens
def clean_doc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens) #here upto for word based
    chars = list(tokens)    #char base uncomment
    tokens = ' '.join(chars)#char based uncomment
    #print(str(tokens))
    return tokens

# load all docs in a directory
def process_docs(directory, is_trian):
    documents = list()
    for filename in listdir(directory):
        path = directory + '/' + filename
        doc = load_doc(path)
        tokens = clean_doc(doc)
        documents.append(tokens)
    return documents



def process_text(text):
    documents = list()
    tokens = clean_doc(text)
    documents.append(tokens)
    return documents



def preprocess():
    global trainX
    global trainy
    trainX.clear()
    trainy.clear()
    #docs1 = process_docs(filename+'/topic1', True)
    #docs2 = process_docs(filename+'/topic2', True)
    #trainX = docs1 + docs2
    #trainy = [0 for _ in range(len(docs1))] + [1 for _ in range(len(docs2))]
    train = pd.read_csv(filename,encoding='iso-8859-1',sep='\t')
    for i in range(len(train)):
        label = train.get_value(i, 'label_bullying')
        msg = train.get_value(i, 'text_message')
        msg = msg.strip()
        tokens = clean_doc(msg)
        if len(tokens) > 0:
            trainX.append(tokens)
            trainy.append(label)

    trainy = to_categorical(trainy)
    #print(trainX)
    print(trainy)
    text.delete('1.0', END)
    text.insert(END,"Features from dataset\n\n");
    #text.insert(END,str(trainX))
    
    
def create_tokenizer(lines):
    tokenizer = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
    tokenizer.fit_on_texts(lines)
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    char_dict = {}
    for i, char in enumerate(alphabet):
        char_dict[char] = i + 1
    tokenizer.word_index = char_dict.copy() 
    tokenizer.word_index[tokenizer.oov_token] = max(char_dict.values()) + 1
    return tokenizer

# calculate the maximum document length
def max_length(lines):
    return max([len(s.split()) for s in lines])

# encode a list of lines
def encode_text(tokenizer, lines, length):
    encoded = tokenizer.texts_to_sequences(lines)
    padded = pad_sequences(encoded, maxlen=length, padding='post')
    #padded = to_categorical(padded)
    print(padded)
    return padded



def cnn():
    global accuracy
    global trainX
    global trainy
    global model
    global tokenizer
    global length
    text.delete('1.0', END)
    tokenizer = create_tokenizer(trainX)
    length = max_length(trainX)
    vocab_size = len(tokenizer.word_index) + 1
    text.insert(END,'Max document length : '+str(length)+"\n")
    text.insert(END,'Vocabulary size : '+str(vocab_size)+"\n")
    trainX = encode_text(tokenizer, trainX, length)
    trainX = np.array(trainX, dtype='float32')

    print(tokenizer.word_index)
    vocab_size = len(tokenizer.word_index)

    embedding_weights = [] #(71, 70)
    embedding_weights.append(np.zeros(vocab_size)) # first row is pad

    for char, i in tokenizer.word_index.items(): # from index 1 to 70
        onehot = np.zeros(vocab_size)
        onehot[i-1] = 1
        embedding_weights.append(onehot)
    embedding_weights = np.array(embedding_weights)

    print(embedding_weights.shape) # first row all 0 for PAD, 69 char, last row for UNK
    print(embedding_weights)

    input_size = length
    embedding_size = 69
    conv_layers = [[256, 7, 3], 
               [256, 7, 3], 
               [256, 3, -1], 
               [256, 3, -1], 
               [256, 3, -1], 
               [256, 3, 3]]

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
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy']) # Adam, categorical_crossentropy
    print(model.summary())

    indices = np.arange(trainX.shape[0])
    np.random.shuffle(indices)

    x_train = trainX[indices]
    y_train = trainy[indices]

    hist = model.fit(x_train, y_train, validation_split=0.2, batch_size=128, epochs=10, verbose=1)

    f = open('history.pckl', 'wb')
    pickle.dump(hist.history, f)
    f.close()

    # retrieve:    
    f = open('history.pckl', 'rb')
    history = pickle.load(f)
    f.close()

    model.save_weights('model_weights.h5')            
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
   

    
    
def predictSteg(vec1, vec2):
    vector1 = np.asarray(vec1)
    vector2 = np.asarray(vec2)
    return dot(vector1, vector2)/(norm(vector1)*norm(vector2))

def predict():
    global testX
    text.delete('1.0', END)
    input_sentence = simpledialog.askstring("Enter your sentence here  for steg analysis detection", "Enter your sentence here  for steg analysis detection")
    testX = process_text(input_sentence)
    testX = encode_text(tokenizer, testX, length)
    ypred = model.predict(testX)
    print(ypred)
    '''
    result = 0
    classname = -1
    for i in range(len(trainX)):
        score = predictSteg(trainX[i],testX[0])
        if score > result:
            result = score
            classname = i
    if trainy[classname] == 0:
        text.insert(END,input_sentence+" contains no steg text")
    elif trainy[classname] == 1:
        text.insert(END,input_sentence+" contains steg text")
    else:
        text.insert(END,"Unable to understand. Given sentence out of trained model")
    '''    

def graph():
    height = [accuracy,90]
    bars = ('Propose LC-CNN Accuracy', 'Existing T-LEX Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()   
                    
    
font = ('times', 16, 'bold')
title = Label(main, text='Convolutional Neural Network Based Text Steganalysis',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Text File with Sentences", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

modelButton = Button(main, text="Preprocess and Convert Text To Indexes", command=preprocess)
modelButton.place(x=50,y=200)
modelButton.config(font=font1)

svmButton = Button(main, text="Run CNN Algorithm & Embeded Words", command=cnn)
svmButton.place(x=50,y=250)
svmButton.config(font=font1)

naiveButton = Button(main, text="Predict Steg Analsysis from sentence", command=predict)
naiveButton.place(x=50,y=300)
naiveButton.config(font=font1)

naiveButton = Button(main, text="Accuracy Graph", command=graph)
naiveButton.place(x=50,y=350)
naiveButton.config(font=font1)

font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=500,y=100)
text.config(font=font1)


main.config(bg='magenta3')
main.mainloop()
