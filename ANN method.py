import keras
import pandas as pd
df=pd.read_csv(r"Test1(3500).csv",encoding="ISO-8859-1")

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


from keras.preprocessing.text import Tokenizer
tokenizer=Tokenizer(le)
tokenizer.fit_on_texts(use_X)
diction=tokenizer.word_index.items()


from keras.preprocessing.sequence import pad_sequences
maxlen=50



from keras.models import load_model
model=load_model('m1')
model.summary()


## Needed for testing
import numpy


tmp_ls=[]
pre_str="I am angry at this man"
new_str=' '.join(t_process(str(pre_str)))
##print(new_str)
tmp_ls.append(new_str)
##print(tmp_ls)
sq_str=tokenizer.texts_to_sequences(tmp_ls)
pd_str=pad_sequences(sq_str, padding='post', maxlen=maxlen)
##print(pd_str)
Xnew=numpy.array(pd_str)
Ynew=model.predict_classes(Xnew)
print("X=%s, Predicted=%s" % (Xnew[0], Ynew[0]))


##Testing on JB
##with open('jb_20.txt') as f:
##    content=f.readlines()

##Testig on newyork times
with open('test.txt') as f:
    content=f.readlines()

for x in content:
    tmp_ls=[]
    pre_str=x.strip()
    new_str=' '.join(t_process(str(pre_str)))
    ##print(new_str)
    tmp_ls.append(new_str)
    ##print(tmp_ls)
    sq_str=tokenizer.texts_to_sequences(tmp_ls)
    pd_str=pad_sequences(sq_str, padding='post', maxlen=maxlen)
    ##print(pd_str)
    Xnew=numpy.array(pd_str)
    Ynew=model.predict_classes(Xnew)
    print("X=%s, Predicted=%s" % (x.strip(), Ynew[0]))
    


