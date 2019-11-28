import keras
import pandas as pd
df=pd.read_csv(r"C:\Users\User\Desktop\PROJECT\Work\Test batches\Test1(3500).csv",encoding="ISO-8859-1")

new_X=list(df['sentence'].values)
Y=list(df['emotion'].values)



from import_nltk import t_process
X=[]
use_X=[]
for sen in new_X:
    tmp=t_process(str(sen))
    X.extend(tmp)
    use_X.append(' '.join(tmp))
    
    


unique_words=list(set(X))
le=len(unique_words)
print("Vocab size:::"+str(le))



##from keras.preprocessing.text import one_hot
##embedded_sentences = [one_hot(sent, le) for sent in use_X]




import sklearn
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(use_X,Y, test_size=0.1,shuffle=True)

print(X_train[:5])
print(Y_train)

from keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer(le)
tokenizer.fit_on_texts(use_X)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
diction=tokenizer.word_index.items()


Y_train=keras.utils.to_categorical(Y_train,5)
Y_test=keras.utils.to_categorical(Y_test,5)


from keras.preprocessing.sequence import pad_sequences
maxlen=50
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

print(X_train[:5])


import numpy
X_train=numpy.array(X_train)
X_test=numpy.array(X_test)
Y_train=numpy.array(Y_train)
Y_test=numpy.array(Y_test)


print(X_train[0,:])

from keras.models import Sequential
from keras import layers
from keras.layers import Dropout

from keras import regularizers

embedding_dim = 50

model = Sequential()
model.add(layers.Embedding(input_dim=le, \
                           output_dim=embedding_dim, \
                           input_length=maxlen))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(64,kernel_regularizer=regularizers.l2(0.01),\
                       activation='relu'))
model.add(Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


model.fit(X_train,Y_train,epochs=10,verbose=2,shuffle=True,batch_size=50)

loss,accuracy= model.evaluate(X_test,Y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))


##saving the model and making predictions
##saving
model.save("test")
print("Model Saved")
##
####loading
##from keras.models import load_model
##model=load_model('test1')
##model.summary()
##
#### I think this stage means that the it is finished till to the model.fit point
##
####Now only the predict point is left


##Texts to sequences

tmp_ls=[]
pre_str="She thinks I'm ugly"
new_str=' '.join(t_process(str(pre_str)))
print(new_str)
tmp_ls.append(new_str)
print(tmp_ls)
sq_str=tokenizer.texts_to_sequences(tmp_ls)
pd_str=pad_sequences(sq_str, padding='post', maxlen=maxlen)
print(pd_str)

Xnew=numpy.array(pd_str)
Ynew=model.predict_classes(Xnew)

print("X=%s, Predicted=%s" % (Xnew[0], Ynew[0]))




