from slackclient import SlackClient
import pandas as pd
import csv
import nltk
import re

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
from nltk.tokenize import word_tokenize

token = 'xoxp-3102231083-26874464368-160898230868-7d1cf5e7bbe53fadc5483de22d5aa502'

sc = SlackClient(token)

users = sc.api_call("users.list")

# create user keys
users = pd.DataFrame(users['members'])
# filter
users = users[['deleted', 'is_bot', 'real_name', 'id']]
users = users[(users.deleted == False) & (users.is_bot == False) & users.real_name.str.len() > 0]

# get history of messages

# list channels
channels = sc.api_call("channels.list")
channels = pd.DataFrame(channels['channels'])
# grab messages from each channel
stemmer = SnowballStemmer('english')
messages = []
for channel_id in channels.id:
    # extract txt only
    result = sc.api_call('channels.history', channel=channel_id)
    out = []
    for mesg in result['messages']:
        if 'text' in mesg:
            strg = mesg['text']
            # clean
            strg = re.sub("[^a-zA-Z]", " ", strg.strip())
            strg = ' '.join([stemmer.stem(plural) for plural in strg.split(' ')])
            out.append(strg)
    messages.extend(out)
# clean data

# things to look at:
# number of messages per month per person
# most reacted post
# average number of channels each user is connected to
# network graph of words said in slack

