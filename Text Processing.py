# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: NLP Learning
#     language: python
#     name: nlp_learning
# ---

# # Description
# This notebook is focusing on different ways of text processing.
#
# We use the headline sentences as illustration.

# +
import json
datastore = []
with open("./Sarcasm_Headlines_Dataset.json/Sarcasm_Headlines_Dataset.json", "r") as f:
    for line in f:
        datastore.append(json.loads(line))
sentences = []
labels = []

for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
demo_examples = sentences[:5]
# Adding some repeating words 
demo_examples.append("the wife wants the son to listen")
demo_examples
# -

# ## Text Pre-cleaning
# Normally we will need to clean the text with the following steps:
# 1. Remove punctuations and replace it with space " "
# 2. Remove html formatting
# 3. Remove leading and trailing spaces
# 4. Convert to lower case
# 5. Remove stopwords

import re
import bs4
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

# ### Customized functions

eng_stopwords = stopwords.words('english')


# +
def remove_duplicate_space(text):
    # Remove extra spaces - cleaning
    return " ".join(text.split())

def remove_numbers(text):
    pattern = r'[0-9]'
    return re.sub(pattern, '', text)

def remove_punctuation(text):
    #pattern = r'[^\w\s]+'
    pattern = '^\s+|\W+|[0-9]|\s+$'
    out = re.sub(pattern, ' ', text)
    return out 

def remove_underscore(text):
    return text.replace("_", " ")

def lower_case(text):
    return text.lower()

# Step 1 - Remove HTML formatting
def remove_html(text):
    soup = BeautifulSoup(text, features="html.parser")
    cleaned_text = soup.get_text()
    return cleaned_text

def text_preprocessing(text):
    text = remove_html(text)
    text = remove_numbers(text)
    text = remove_punctuation(text)
    text = remove_underscore(text)
    text = remove_duplicate_space(text)
    text = lower_case(text)
    cleaned_text = ' '.join([i for i in text.split() if i not in eng_stopwords])
    
    return cleaned_text


# -

test1 = "<p>I've been looking into job markets in the past 3 months, but nothing happened</p>"
text_preprocessing(test1)

demo_examples_cleaned = [text_preprocessing(i) for i in demo_examples]
demo_examples_cleaned

# ## One hot encoding and bag of words

# For example, given the following cleaned texts:  
#
# `["dog jump fence", "dog break fence"]`
#
#
# To create a one hot representation the strings must be broken down into a list of words:  
#
# `[['dog', 'jump', 'fence'], ['dog', 'break', 'fence']]`  
#
# Once the string has been broken down into a list words, the list will need to be flattened to create a mapping between a unique word and a unique one hot representation:  
#
# `['dog', 'jump', 'fence', 'dog', 'break', 'fence']`
#
# Feeding the flattened list to fit the One Hot Encoder will create the following mapping:  
#
#  `[1., 0., 0., 0.]` $\rightarrow$ break  
#  `[0., 1., 0., 0.]` $\rightarrow$ dog    
#  `[0., 0., 1., 0.]` $\rightarrow$ fence  
#  `[0., 0., 0., 1.]` $\rightarrow$ jump   
#
# Now that we have the mapping, we can get the one hot representation of `"dog jump fence"`:
#
# `[[0., 1., 0., 0.],
#   [0., 0., 0., 1.], 
#   [0., 0., 1., 0.]]`  
#
# The one hot representation can be summed along the rows to create the bag of words representation for `"dog jump fence"`:
#
# `[0., 1., 1., 1.]`

from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

demo_examples_cleaned 

# One-hot-encoding is to convert each "category" in the feature to be a binary feature with 1 indicating belonging to the category and 0 indicating not belonging. In the context of NLP, we need to have the full vocab first. 

# Split each sentence into words first
data_split = [i.split() for i in demo_examples_cleaned]
data_split

data_flattern = [i for sublist in data_split for i in sublist]
data_flattern_array = np.array(data_flattern)
data_flattern_array

print("Total number of words:", len(data_flattern_array))
print("Total number of unique words:", len(set(data_flattern_array)))

unique_array = np.array(list(set(data_flattern_array)))
unique_array.reshape(-1,1)

# Initialize the one hot encoder 
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
one_hot_encoder.fit(unique_array.reshape(-1,1))
# This step trains the "one_hot_encoder" to learn what words are in the "universe"

# The trained module or the corresponding location of each "category" is below
one_hot_encoder.categories_

test2 = "happy wife happy life"
test2_for_transform = np.array(test2.split()).reshape(-1,1)
test2_for_transform

test2_encoded = one_hot_encoder.transform(test2_for_transform)

np.sum(test2_encoded, axis=0)

# **Here we see that the 43rd word is "wife" (appeared once) and 15th word is "happy" (appeared twice). The word "life" is unknown so it is ignored.**

for i in range(len(demo_examples_cleaned)):
    sentence_i = demo_examples_cleaned[i]
    print("Encoding sentence: ", sentence_i)
    sentence_for_transform = np.array(sentence_i.split()).reshape(-1,1)
    sentence_transformed = one_hot_encoder.transform(sentence_for_transform)
    sentence_array = np.sum(sentence_transformed, axis=0)
    if i==0:
        encoded_demo_examples = sentence_array
    else:
        encoded_demo_examples = np.vstack([encoded_demo_examples, sentence_array])


encoded_demo_examples.shape

np.sum(encoded_demo_examples, axis=0)

# +
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
text = ["I have an apple", "The apple is red", "I like red like like"]
tfidf_vectorizer.fit(text)
X = tfidf_vectorizer.transform(text).toarray()
X
# -

tfidf_vectorizer.get_feature_names_out()
