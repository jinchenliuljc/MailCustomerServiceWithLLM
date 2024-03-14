from chatbot import ChatBot
import pickle
import os


class Classifier(ChatBot):
    def __init__(self, api_key, system_for_question_generation, system_for_answer, match_param=3, match_threshold=0.5):
        super().__init__(api_key, system_for_question_generation, system_for_answer, match_param, match_threshold)

    def main(self, input, df):
        return self.main(input, df)
    
    


openai_api = os.environ['OPENAI_API_KEY']
print(openai_api)


# pickle文件是一个dataframe，和源excel的区别是多出一列包含所有问题的embedding，在未来应该是在存储知识到知识库时就做好embedding
with open('customer_rep.pickle','rb') as f:
    df = pickle.load(f)

with open('prompt_for_answer.txt','r') as f:
    prompt_for_answer = f.read()

with open('prompt_for_question.txt','r') as f:
    prompt_for_question = f.read()

bot = ChatBot(openai_api, prompt_for_question, prompt_for_answer)

input = {'name': 'Christie Bohnstingl', 'question': '''Hello, \
Unfortunately my Schenley steam mom has stopped working. \
Order number from Amazon \
111-6479641-6445847 \
I would appreciate assistance getting it replaced. \
Thank you so much for your time.\
\
Kindly, \
Christie Bohnstingl '''}
result = bot.main(input, df)

print('---------------------------\n',result)
