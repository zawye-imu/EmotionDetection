import pandas as pd
df=pd.read_csv(r"sen_res.csv",encoding="ISO-8859-1")


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

    

    print(filtered_sentence)


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



##Testing the results part
with open('test.txt') as f:
    content=f.readlines()


for sentence in content:
    print("\n\n"+sentence)
    aff_intensity(sentence)
##aff_intensity("The dogs are happy")
