##import tweepy
##import csv
##import json
##
##
##consumer_key = 'Tw1MPMHsa1b9aZWfQdJe5ojFo'
##consumer_secret = '9GJ8f5f99pj2s2QaMeY2Vi5dlPMjFXCl94Jm2X5oTcne6DJxMQ'
##access_key = '1173899410628997122-RYt62ymT1SJeJbkeLz1X3E2oJvSKnH'
##access_secret = 'HVErnHCAMKDCG7efNU6BJmqFK4D8adv1oHpjZd0oyvDjZ'
##
##
##auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
##auth.set_access_token(access_key, access_secret)
##api = tweepy.API(auth,wait_on_rate_limit=True)
####api=tweepy.API(auth)
##
##sn="TheMyanmarTimes"


##all_the_tweets = []
##
### We will get the tweets with multiple requests of 200 tweets each
##new_tweets = api.user_timeline(screen_name=sn, count=200)
##
### saving the most recent tweets
##all_the_tweets.extend(new_tweets)
##
### save id of 1 less than the oldest tweet
##oldest_tweet = all_the_tweets[-1].id - 1
##
### grabbing tweets till none are left
##while len(new_tweets)>0:
##    # The max_id param will be used subsequently to prevent duplicates
##    new_tweets = api.user_timeline(screen_name=sn,count=200, max_id=oldest_tweet)
##    # save most recent tweets
##    all_the_tweets.extend(new_tweets)
##    # id is updated to oldest tweet - 1 to keep track
##    oldest_tweet = all_the_tweets[-1].id - 1
##    print ('...%s tweets have been downloaded so far' % len(all_the_tweets))
##
### transforming the tweets into a 2D array that will be used to populate the csv
##outtweets = [[tweet.id_str, tweet.created_at,tweet.text.encode('utf-8')] for tweet in all_the_tweets]
##
## ##writing to the csv file
##
##with open(sn + '_tweets.csv', 'w', encoding='utf8') as f:
##    writer = csv.writer(f)  
##    writer.writerow(['id', 'created_at', 'text'])
##    writer.writerows(outtweets)

##Getting replies part
##repliesss={}
##tmp=[]
##for tweet in all_the_tweets:
##    res=tweepy.Cursor(api.search,q='to:HtethtetHtwe3',since_id=tweet.id_str,timeout=1000).items()
##    for item in res:
##        if item.in_reply_to_status_id== tweet.id_str:
##            tmp.append(item.text)
##        replies[tweet.id_str]=tmp
##        tmp.clear()

##
##file=open("testfile.txt","w")
##
##replies=[]
##non_bmp_map = dict.fromkeys(range(0x10000,0x10FFFF), 0xfffd)
##for full_tweets in tweepy.Cursor(api.user_timeline,screen_name=sn,timeout=999999).items(20):
##  for tweet in tweepy.Cursor(api.search,q='to:'+sn,result_type='recent',timeout=999999).items(100):
##    if hasattr(tweet, 'in_reply_to_status_id_str'):
##      if (tweet.in_reply_to_status_id_str==full_tweets.id_str):
##        replies.append(tweet.text)
##  try:
####      print("Tweet :",full_tweets.text.translate(non_bmp_map))
##      file.write(str(full_tweets.text.translate(non_bmp_map))+"\n")
##  except:
##      pass
##  for elements in replies:
##      try:
####          print("Replies :",elements.encode('utf-8'))
##          file.write(str(elements.encode('utf-8'))+"\n")
##      except:
##          pass
##             
##  replies.clear()
##
##file.close()



##Prediciting part
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




## Needed for testing
import numpy




from keras.models import load_model
model=load_model('m1')
model.summary()


from import_nltk import t_process



##This is the keyword method 
import pandas as pd
df=pd.read_csv(r"C:\Users\User\Desktop\PROJECT\Work\DataSets\sen_res.csv",encoding="ISO-8859-1")


term=list(df['term'].values)
score=list(df['score'].values)
AffectDimension=list(df['AffectDimension'].values)


##dictionary generation
diction=[['term','score','AD']]
i=0
for x in term:
    diction.append([x,score[i],AffectDimension[i]])
    i=i+1

##print(len(term))
##print(diction['outraged'])

print(diction[:5])
print(len(diction))


#### Some NLP

def aff_intensity(inp):
    

    s1="This is the test's sentence? right!!.."

    ##Actual sentence now
    s1="I am happy and sad at the same time ughhhh"

    
    s1=inp



    ##lowercase
    s1=s1.lower()

    ##removing punctuations
    import string
    s1=s1.translate(str.maketrans('', '', string.punctuation))


    ##print(s1)

    ##Tokenizing
    import nltk
    tokenizer=nltk.tokenize.TreebankWordTokenizer()
    s1=tokenizer.tokenize(s1)

    s2=[]
    stemer2=nltk.stem.WordNetLemmatizer()
    for word in s1:
        s2.append(stemer2.lemmatize(word))

    from nltk.corpus import stopwords
    stop_words=set(stopwords.words("english"))
    filtered_sentence=[w for w in s2 if not w in stop_words]

    

##    print(filtered_sentence)


    #### Classying emotion


    classification=[]
    for word in filtered_sentence:
        for row in diction:
            if row[0] == word:
                
                classification.append(row)
                break

##    print(classification)


    ## Post processing
    final_score=0
       
    for rows in classification:
        if rows[2]=='anger':
            final_score+=rows[1]

    print("anger score:"+str(final_score))

    final_score=0
    for rows in classification:
        if rows[2]=='sadness':
            final_score+=rows[1]

    print("sadness score:"+str(final_score))

    final_score=0
    for rows in classification:
        if rows[2]=='joy':
            final_score+=rows[1]

    print("joy score:"+str(final_score))

    final_score=0
    for rows in classification:
        if rows[2]=='fear':
            final_score+=rows[1]

    print("fear score:"+str(final_score))


##This is the end of keyword-method. 





##Testig on newyork times
with open('testfile.txt') as f:
    content=f.readlines()

dn={'4':'angry(negative)','0':'neutral','3':'hate','2':'worry','1':'love(positive)'}

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
    
    
    
    print("\n")
    
    print("Sentence=%s,\nFrom ANN method   >>>\nPredicted=%s" % (x.strip(), dn[str(Ynew[0])]))

    
    print("\nFrom keyword method >>>")
    
    aff_intensity(x)














        




