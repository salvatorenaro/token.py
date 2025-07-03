import numpy as np
import string
from tensorflow.keras.preprocessing.text import Tokenizer


x = ['The cat sat on the mat', 'the dog ate my homework']


token = {}
for i in x:
    for word in i.split():
        if word not in token:
            token[word] = len(token)+1



max_lenght = 10

result = np.zeros((len(x),max_lenght,max(token.values())+1))

for key,value in enumerate(x):
    for j,char in list(enumerate(value.split()))[:max_lenght]:
        index = token.get(char)
        result[key,j,index] = 1


characters = string.printable
token_index = dict(zip(range(1,len(characters)+1),characters))
max_lenghts = 50
results = np.zeros((len(x),max_lenghts, max(token_index.keys())+1)) 

for key,value in enumerate(x):
    for  j, char in enumerate(value):
        index_ = token_index.get(char)
        results[key,j,index] = 1

tokenizer = Tokenizer(num_words = 10000)
tokenizer.fit_on_texts(x)
sequences = tokenizer.texts_to_sequences(x)
one_hot = tokenizer.texts_to_matrix(x,mode = 'binary')
word_index  = tokenizer.word_index


dimensionality = 1000
max_lenght_f  = 10
matrix = np.zeros((len(x),max_lenght_f,dimensionality))
for key,value in enumerate(x):
    for j,char in list(enumerate(value.split()))[:max_lenght_f]:
        ind_ex = abs(hash(char)) % dimensionality
        matrix[key,j,ind_ex] = 1
