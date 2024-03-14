import pandas as pd
import pickle
from openai import OpenAI
import os
from chatbot import ChatBot
import ast


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    embeddings = client.embeddings.create(input = [text], model=model).data[0].embedding
    return embeddings


pd.set_option('display.max_colwidth', None)

openai_api = os.environ['OPENAI_API_KEY']

client = OpenAI(api_key=openai_api)

df = pd.read_excel('reply.xlsx').astype('str')
text = df['Customer question']


# 用openai的embedding将问题向量化


embeddings_ = text.apply(get_embedding)

df['embeddings'] = embeddings_


test = pd.read_csv('text.csv')


# print(ChatBot.calc_similarity(pickle.loads(test['embeddings'][0]),pickle.loads(test['embeddings'][1])))
astr = test['embeddings'][0]

if not astr.endswith(']'):
    astr += ']'


print(len(eval(astr)))



