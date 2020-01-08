import keras
import pandas as pd
df=pd.read_csv(r"Test1(3500).csv",encoding="ISO-8859-1")

##new_X=list(df['sentence'].values)
##Y=list(df['emotion'].values)

##For second dataset
new_X=list(df['content'].values)
Y=list(df['sentiment'].values)

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

##print(X_train[:5])

from keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer(le)
tokenizer.fit_on_texts(use_X)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
diction=tokenizer.word_index.items()





from keras.preprocessing.sequence import pad_sequences
maxlen=20
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

##print(X_train[:5])


import numpy
X_train=numpy.array(X_train)
X_test=numpy.array(X_test)
Y_train=numpy.array(Y_train)
Y_test=numpy.array(Y_test)

print(X_train[:5])

print(Y_train[:5])


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train,Y_train)


Y_pred = clf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(Y_test, Y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(Y_test,Y_pred)
print("Accuracy:",result2)



##Texts to sequences

tmp_ls=[]
pre_str="I am really angry at you."
new_str=' '.join(t_process(str(pre_str)))

tmp_ls.append(new_str)

sq_str=tokenizer.texts_to_sequences(tmp_ls)
pd_str=pad_sequences(sq_str, padding='post', maxlen=maxlen)



Xnew=numpy.array(pd_str)
print(Xnew)



res=clf.predict(Xnew)

print(res)

