import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

"""
==========================NLP Preprocessing=============================
Cleaning texts.

"""

def clean_sentence(txt):
    txt=re.sub(r"@.+","",txt)
    txt=re.sub(r"\\n", "", txt)
    txt=re.sub(r"\s", " ", txt)
    txt=re.sub(r"In making a purchase.+","",txt)
    return txt


def remove_punctuation(txt):

    txt=re.sub(r"@.+","",txt)
    txt=re.sub(r"\\n", "", txt)
    txt=re.sub(r"\s", " ", txt)
    txt=re.sub(r"In making a purchase.+","",txt)
    p=re.compile('[\'!"#$%&\\()*+,-.\/:;<=>?@[\\]^_`{|}~]')
    txt=re.sub(p, "", txt)

    return txt


nltk.download('punkt')
nltk.download('wordnet')

def tokenize(txt):
    stemmer = WordNetLemmatizer()
    return [stemmer.lemmatize(w) for w in word_tokenize(txt)]

def query_clean(txt):
    return tokenize(remove_punctuation(txt))

def preprocessor(df):
    df["clean_text"]=df["product_description"].apply(clean_sentence)
    print("✅added clean text column")

    df["text_tokens"]=df["product_description"].apply(remove_punctuation)
    print("✅remove_punctuation")

    df["text_tokens"]=df["text_tokens"].apply(tokenize)
    print("✅words tokenized")

    return df
