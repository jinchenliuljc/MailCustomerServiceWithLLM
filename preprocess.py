import pandas as pd
import pickle
from openai import OpenAI
import os
from chatbot import ChatBot
import ast


# def get_embedding(text, model="text-embedding-3-small"):
#     text = text.replace("\n", " ")
#     embeddings = client.embeddings.create(input = [text], model=model).data[0].embedding
#     return embeddings

#这个文件中定义的方法接受excel并将问题转换为embedding再暂存，读数据时可以使用read_data方法。

def preprocess(filename):

    pd.set_option('display.max_colwidth', None)

    openai_api = os.environ['OPENAI_API_KEY']

    client = OpenAI(api_key=openai_api)

    df = pd.read_excel(filename).astype('str')
    text = df['Customer question']


    # 用openai的embedding将问题向量化


    embeddings_ = text.apply(lambda x : ChatBot.get_embedding(x,client))

    df['embeddings'] = embeddings_

    new_filename = filename.split('.')[0]+'.csv'
    df.to_csv(new_filename)
    return new_filename


def read_data(filename):
    new_filename = filename.split('.')[0]+'.csv'
    csv = pd.read_csv(new_filename)
    embeddings = csv['embeddings']
    csv['embeddings'] = embeddings.apply(eval)
    return csv





# print(len(eval(astr)))



