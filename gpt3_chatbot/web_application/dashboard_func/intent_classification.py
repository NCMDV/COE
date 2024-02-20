import pandas as pd
import joblib
import re
import json
# from spellchecker import SpellChecker
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('stopwords')
nltk.download('wordnet')

# list of intents
intents = ['greeting', 'live_agent', 'coe']
files = dict()

#load models and vectorizers (current directory must be in web_application)
files["greeting_vectorizer"] = joblib.load("./dashboard_files/vectorizers/greeting_MLPClassifier_9952.joblib")
files["greeting_model"] = joblib.load("./dashboard_files/models/greeting_MLPClassifier_9952.joblib")

files["live_agent_vectorizer"] = joblib.load("./dashboard_files/vectorizers/live_agent_MLPClassifier_9923.joblib")
files["live_agent_model"] = joblib.load("./dashboard_files/models/live_agent_MLPClassifier_9923.joblib")

files["coe_vectorizer"] = joblib.load("./dashboard_files/vectorizers/coe_MLPClassifier_0.joblib")
files["coe_model"] = joblib.load("./dashboard_files/models/coe_MLPClassifier_0.joblib")


#load objects for preprocessing
stopwords_eng = set(stopwords.words('english'))
negation_set = {'no', 'nor', 'not', 't', 'can', "don't", 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn',"mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
stopwords_eng = stopwords_eng - negation_set


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    try:
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}

        return tag_dict.get(tag, wordnet.NOUN)
    except:
        print(type(word))
        return wordnet.NOUN


def preprocess_text(text):
    """
    Time in seconds for each step:
    {'cleaning': 0.12903165817260742, 'spelling': 117.63137936592102, 'stopwords': 0.0, 'lemmatization': 0.6576428413391113}
    """

    #to remove punctuations only
    text = text.lower()
    text = re.sub('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+', '', text)
    
    #to remove punctuations and special characters (other languages)
    text = re.sub(r'[^\w\s]+', '', text) 

    word_tokens = word_tokenize(text)

    # #correct misspelled (english)    
    # spell = SpellChecker()
    # misspelled = spell.unknown(word_tokens)
    # word_tokens = [w if w not in misspelled else spell.correction(w) for w in word_tokens]

    #removing stopwords
    word_tokens = [w for w in word_tokens if not w in stopwords_eng]

    #lemmatization
    lemmatizer = WordNetLemmatizer()
    word_tokens = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) if w is not None else "" for w in word_tokens]

    clean_text = ' '.join(word_tokens)
    return clean_text


def predict_intent(text):

    text = preprocess_text(text)

    prediction_probabilites = dict()

    for intent in intents: 
    #check if greeting
        vectorizer = files[intent + "_vectorizer"]
        model = files[intent + "_model"]
        x_values_list = vectorizer.transform([text]).toarray()
        x_ = pd.DataFrame(x_values_list,columns = vectorizer.get_feature_names_out())
        prediction_probabilites[intent] = round(model.predict_proba(x_)[0][0], 4)

    print("prediction probabilities:", prediction_probabilites)
    max_intent = max(prediction_probabilites, key=prediction_probabilites.get)
    if prediction_probabilites[max_intent] > 0.5:
        return max_intent
    
    else:
        return "others"
    
    # #check if live_agent
    # x_values_list = live_agent_vectorizer.transform([text]).toarray()
    # x_ = pd.DataFrame(x_values_list,columns = live_agent_vectorizer.get_feature_names_out())
    # pred = live_agent_model.predict(x_)
    # if pred == "live_agent":
    #     return "live_agent"
    
    # return "others"


