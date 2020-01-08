def t_process(s):
    
##    s1="\'This is a test's string with test words such \
##    as: not beautiful, played, cats, feet, 400, 15 etc..\'"
    s1=s

               
    ##lowercase
    s1=s1.lower()

    ##removing punctuations
    import string
    s1=s1.translate(str.maketrans('', '', string.punctuation))


##    print(s1)

    ##Tokenizing
    import nltk
    tokenizer=nltk.tokenize.TreebankWordTokenizer()
    s1=tokenizer.tokenize(s1)


    ##Negation hadling 
    from nltk.corpus import wordnet
    if 'not' or '\'t' in s1:
        
        for w in s1:
            if w == 'not' or w == '\'t':

                try:
                    
                    key=w
                    dex=s1.index(key)
                    key_word=s1[dex+1]

                    
                    net_word=wordnet.synsets(key_word)
                    net_word2=net_word[0].lemmas()
                    ant=net_word2[0].antonyms()[0].name()
                    
                    if ant !=[]:
                        
                        del s1[dex]
                        del s1[dex]
                        s1.insert(dex,ant)
                    else:
                        pass
                except Exception as e:
                    pass
                
                    

                    
                    
               
                
                  
    else:
        pass
        
                  

            
            
            
         

    


    ##number formats
    from num2words import num2words
    for words in s1:
        if words.isdigit():
            dex=s1.index(words)
            w=words
            del s1[dex]
            s1.insert(dex,num2words(w))
            
##    print(s1)

    ##Stemming
    s2=[]
    stemer=nltk.stem.PorterStemmer()
    for words in s1:
        s2.append(stemer.stem(words))


    ##lemmatizing
##    stemer2=nltk.stem.WordNetLemmatizer()
##    for word in s1:
##        s2.append(stemer2.lemmatize(word))
    
##    print(s2)


    ##Stopwrods
    from nltk.corpus import stopwords
    stop_words=set(stopwords.words("english"))
    filtered_sentence=[w for w in s2 if not w in stop_words]
##    print(stop_words)

##checking if is contaion empty strings and deleting them    
    for words in filtered_sentence:
        if words=='':
            dex=filtered_sentence.index(words)
            del filtered_sentence[dex]

    return filtered_sentence
    

##print(t_process("How are you doing? papa la la la la not angry"))
