from preprocess import preprocess, read_data
from chatbot import ChatBot
import numpy as np
from openai import OpenAI
import pandas as pd
import pickle
import os

# 这个文件是用来运行测试问题的
if __name__ == "__main__":
    openai_api = os.environ['OPENAI_API_KEY']


    knowledge_base_filename = 'data/knowledgebase.xlsx'
    try: 
        df = read_data(knowledge_base_filename)     
    except Exception:
        new_filename = preprocess(knowledge_base_filename)
        df = read_data(new_filename)

    print(df)


    with open('prompt_for_answer.txt','r') as f:
        prompt_for_answer = f.read()

    client = OpenAI(api_key=openai_api)
    bot = ChatBot(client, prompt_for_answer)

    test_set = pd.read_excel('data/testset.xlsx')

    bot_answer = []
    for id, row in test_set.iterrows():
        
        input = {'name': row['ID'], 'question': row['Customer Question']}
        result = bot.main(input, df)
        bot_answer.append(result[0])

    bot_answer = pd.Series(bot_answer, name = 'Bot Answer')

    test_set['Bot Answer'] = bot_answer

    
    test_set.to_excel('res1.xlsx')
    
    # input = {'name': 'Sevana Hasty', 'question': '''Yes, here is the date and the order number.

    # Order Placed: August 17, 2023
    # Amazon.com order number: 114-0079533-7630677
    
    # Thanks,
    # -Sevana Hasty'''}
    # result = bot.main(input, df)