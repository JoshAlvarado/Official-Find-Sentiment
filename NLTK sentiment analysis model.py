from time import time
import random, re, string
from nltk import text
import pandas as pd 
import matplotlib.pyplot as plt 
from nltk.tokenize import TweetTokenizer
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import classify
from nltk import NaiveBayesClassifier
import pickle 
from pickle import dump

STOP_WORDS = stopwords.words('english')

####IMPORT DATA####
ds_raw = pd.read_csv(r'C:\Users\Joshu\OneDrive\Documents\Python\NLTKmodel\datasetraw.csv', encoding= "ISO-8859-1", header=None) #https://www.kaggle.com/kazanova/sentiment140 download link
ds_raw.columns = ["label" , "id", "date", "query", "username", "tweet"] #turn csv to DS
#Get rid of ID, DATE, QUERY, USERNAME
ds = ds_raw[['label','tweet']] #Only take label and tweet from raw DS
ds.head()
positive_tweets = ds[ds['label'] == 4]
negative_tweets = ds[ds['label'] == 0]
print(len(positive_tweets), len(negative_tweets))
ds_pos = positive_tweets.iloc[:int(len(positive_tweets))] # dividing data depending on machine power
ds_neg = negative_tweets.iloc[:int(len(negative_tweets))] # dividing data depending on machine power
print(len(ds_pos), len(ds_neg))
ds = pd.concat([ds_pos, ds_neg])
len(ds)
print("data split")


tk = TweetTokenizer(reduce_len=True)

data = []

X = ds['tweet'].tolist()
Y = ds['label'].tolist()
#MODEL PREDICTIONS
for x, y in zip(X, Y):
    if y == 4:
        data.append((tk.tokenize(x), "Positive"))
    else:
        data.append((tk.tokenize(x), "Negative"))


data[:5]
###LABELING N/V/A AND REMOVING INFLECTIONAL ENDINGS (example: moving -- move)
def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    lemmatized_sentence = []
    for word, tag in pos_tag(tokens):
        if tag.startswith('NN'):
            pos = 'n'  #noun
        elif tag.startswith('VB'):
            pos = 'v' #verb
        else:
            pos = 'a' #adj
        lemmatized_sentence.append(lemmatizer.lemmatize(word, pos))
    return lemmatized_sentence

#REMOVING COMMON SLANG
def remove_slang(token):
    if token == 'u':
        return 'you'
    if token == 'urs':
        return 'yours'
    if token == 'r':
        return 'are'
    if token == 'pls':
        return 'please'
    if token == 'plz':
        return 'please'
    if token == '2day':
        return 'today'
    if token == 'some1':
        return 'someone'
    if token == 'yrs':
        return 'years'
    if token == 'hrs':
        return 'hours'
    if token == 'mins':
        return 'minutes'
    if token == '4got':
        return 'forgot'
    return token

def cleantokens(tweet_tokens):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):

        #remove links
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+','', token)
        #remove mentions
        token = re.sub("(@[A-Za-z0-9_]+)","", token)
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        cleaned_token = remove_slang(token.lower())
        if cleaned_token not in string.punctuation and len(cleaned_token) > 2 and cleaned_token not in STOP_WORDS:
            cleaned_tokens.append(cleaned_token)
    return cleaned_tokens



def list_to_dict(cleaned_tokens):
    return dict([token, True] for token in cleaned_tokens)

cleaned_tokens_list = []

for tokens, label in data:
    cleaned_tokens_list.append((cleantokens(tokens), label))
 


final_data = []

for tokens, label in cleaned_tokens_list:
    final_data.append((list_to_dict(tokens), label))



final_data[:5]
#determine accuracy compared to NLTK's
random.Random(140).shuffle(final_data)
trim_index = int(len(final_data) * 0.9)
train_data = final_data[:trim_index]
test_data = final_data[trim_index:]

classifier = NaiveBayesClassifier.train(train_data)


print('Accuracy on train data:', classify.accuracy(classifier, train_data))
print('Accuracy on test data:', classify.accuracy(classifier, test_data))
print(classifier.show_most_informative_features(20))


custom_tweet = "I love league of legends so much."

custom_tokens = cleantokens(tk.tokenize(custom_tweet))

print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))  # prints the custom_tweet and whats its been classified.

print("Saving")
	

f = open('my_classifier.pickle', 'wb')
pickle.dump(classifier, f)
f.close()