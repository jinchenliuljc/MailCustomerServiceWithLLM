from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
import pandas as pd
import pickle
import os



class ChatBot():

  def __init__(self, client, system_for_answer, match_param=100, match_threshold=0.5):
    # self.api_key = api_key

    # 用于生成测试问题的prompt
    # self.system_for_question_generation = system_for_question_generation


    # 用于生成回复的prompt，让模型结合上下文和参考问答回答新问题
    self.system_for_answer = system_for_answer

    self.client = client

    self.match_param = match_param
    self.match_threshold = match_threshold

  def get_embedding(self,text, client, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding


  def calc_similarity(self,vec1,vec2):
    vec1 = np.array(vec1).reshape(1,-1)
    vec2 = np.array(vec2).reshape(1,-1)
    return cosine_similarity(vec1,vec2)[0][0]


  def template(self,the_type,role,content):
    # 针对不同case生成不同的template
    if the_type == 'context':
      return {"role": role, "content": content}
    if the_type == 'reference':
      if role == 'user':
        return {'question' : content}
      if role == 'assistant':
        return {'answer' : content}
    if the_type == 'test_question':
      if role == 'user':
        return {"role": 'assistant', "content": content}
      if role == 'assistant':
        return {"role": 'user', "content": content}


  def form_template(self,the_type, dialogs:np.ndarray):
    '''
    将问答对格式和openai sdk的get_completion输入对齐，这里假设未来的数据库中新记录从尾部插入，因为样例数据中是这样
    当然不是也没关系，可以保存个时间戳数据这里再按时间戳sort
    '''
    if dialogs.shape[0] == 0:
      return []

    messages = []
    # asked = []
    for dialog in dialogs:
      # if not dialog[0] in asked:
        # messages.append({"role": "user", "content": dialog[0]})
      messages.append(self.template(the_type,'user',dialog[0]))
      # asked.append(dialog[0])

      messages.append(self.template(the_type,'assistant',dialog[1]))
    return messages


  #调用api根据prompt和上下文生成回复
  def get_completion(self, client, messages, model="gpt-4-0125-preview", temperature=0):
    response = client.chat.completions.create(model=model, messages = messages, temperature=temperature)
    return response.choices[0].message.content, response.usage


  def get_refs(self,question,df):
    # 将问题embed到向量
    target = self.get_embedding(question,self.client)

    # 匹配top k个最相似的问答对，k在类初始化时定义
    df['similarity'] = df['embeddings'].apply(self.calc_similarity,vec2=target)
    refs = df.nlargest(self.match_param,'similarity')[['Case','Customer question','Customer Service Reply']]
    # 加一个阈值筛选，如果相似度全都太小就不要reference
    # refs = refs[refs['similarity']>self.match_threshold][['Customer question','Customer Service Reply']].values
    print('\n---------------------', refs)
    return refs

  # OPENAI_API_KEY = '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'


  # 这里不清楚未来的input是什么格式，这边先用一个字典（json）
  def main(self, input, df):
    # print((df['Case']==input['name']).any())
    # if (df['Case']==input['name']).any():

    # 取出与该客户的所有邮件往来历史
    dialogs = df[df['Case']==input['name']][['Customer question','Customer Service Reply']].values
    answer_context = self.form_template('context',dialogs)
    print(answer_context,'\n----------------------------')


    refs = self.get_refs(input['question'], df)
    # refs = df[['Customer question','Customer Service Reply']].values
    # print(refs.head())
    # with open('dataframe.txt','w') as f:
    #   f.write(refs)
    # 插入system消息
    refs_ = self.form_template('reference',refs)

    # print(refs_)
    answer_context.insert(0,{"role": "system", "content": self.system_for_answer.format(ref=refs_)})
    # print(answer_context)
    # answer_context.insert(0,{"role": "system", "content": self.system_for_answer})
    # 插入新问题
    answer_context.append({"role": "user", "content": input['question']})

    print(answer_context[1])

    # 生成回答
    answer,usage = self.get_completion(self.client,answer_context)
    # answer_context = self.form_template('context',dialogs)

    return answer,usage

  # with open('/content/drive/MyDrive/customer_rep.pickle','rb') as f:
  #   df = pickle.load(f)

  def test(self, df):
  #存储测试问答
    test_samples = {'name':[],'question':[],'answer':[]}
    with open('prompt_for_question.txt','r') as f:
      prompt_for_question = f.read()

    # 基于每一个客户的所有邮件（此处用姓名因为还没有id）往来历史生成一个可能的新问题，再用bot回复，测试bot行为是否可靠
    for name in df['Case'].unique():
      # print(name)
      #取出单一客户的所有上下文
      dialogs = df[df['Case']==name][['Customer question','Customer Service Reply']].values


      test_question_context = self.form_template('test_question',dialogs)

      # 插入system消息
      test_question_context.insert(0,{"role": "system", "content": prompt_for_question})
      print(test_question_context)
      # 生成测试问题
      question = self.get_completion(self.client, test_question_context)
      # print(question)

      answer = self.main({'name':name,'question':question},df)


      #保存测试问答对
      test_samples['name'].append(name)
      test_samples['question'].append(question)
      test_samples['answer'].append(answer)

      break
    return test_samples



class Classifier(ChatBot):
    def __init__(self, client, system_for_answer, match_param=3, match_threshold=0.5):
      super().__init__(client, system_for_answer, match_param, match_threshold)

    def main(self, input, df):
        return self.main(input, df)


if __name__ == "__main__":
  openai_api = os.environ['OPENAI_API_KEY']


  # pickle文件是一个dataframe，和源excel的区别是多出一列包含所有问题的embedding，在未来应该是在存储知识到知识库时就做好embedding
  # with open('customer_rep.pickle','rb') as f:
  #   df = pickle.load(f)

  df = pd.read_excel('reply.xlsx').astype('str')

  with open('prompt_for_answer.txt','r') as f:
    prompt_for_answer = f.read()




  client = OpenAI(api_key=openai_api)
  bot = ChatBot(client, prompt_for_answer)

  # input = {'name': 'Ashley White', 'question': '''Hello, \
  # Unfortunately my Schenley steam mom has stopped working. \
  # Order number from Amazon \
  # 111-6479641-6445847 \
  # I would appreciate assistance getting it replaced. \
  # Thank you so much for your time.\
  # \
  # Kindly, \
  # Ashley White '''}
  # result = bot.main(input, df)

  input = {'name': 'Narumit', 'question': '''Thank you for your reply. I actually bought 2 of them from Amazon.com, but I only kept one of the original package. 

My name is Narumit Saksrisanguan 
My address is 1812 151st St SW, Lynnwood, WA 98087

Yes, I can ship them back. Where and how do I ship it to you? Do you provide the return label?

Thank you,
Narumit '''}
  result = bot.main(input, df)


  # input = {'name': 'Mary Kelly','question':'''
  # Hello,

  # I've been using your steam cleaner mop with accessories since September 2023.

  # It worked wonderfully.
  # Until tonight. I was using the steam cleaner to mop my bathroom floor. There was a loud pop noise and the steam cleaner stopped working. 

  # I also got electrocuted.

  # The steam cleaner turns on but no steam comes out.

  # I have only used tap water, NYC excellent tap water, to fill the steam cleaner reservoir when I'm using it.

  # The reservoir was filled with water when it stopped working.
  # I emptied the water, waited a few minutes and filled the reservoir with tap water again.
  # I tried plugging it into a different outlet even though the steamer turns on. 

  # I looked to see if anything was obstructing the steam flow.

  # Then I called Amazon and I trouble shooted with a Customer Service Representative to try and get the Schenley steam cleaner working again. 

  # Nothing worked. The steam cleaner turned on but didn't produce steam.

  # Also , in the past month the Schenley steam cleaner started dripping water from time to time from the bottom of the reservoir. Not a constant dripping of water but from time to time while using the steam cleaner a few drops of water would splash out of it. It was dripping from the bottom of the reservoir. The steam cleaner was working though so I didn't really notice
  # I noticed it now. IT ELECTROCUTED ME!

  # There are no cracks or holes in the reservoir except for the hole where the water goes into the reservoir.
  # I've never dropped the steam cleaner either. 

  # I just tried to use the steam cleaner again. 
  # I think that the reservoir has a leak around the edges that I can't see. Water is running out of the edges on the of the reservoir as well as the area where the plug comes out in a thin but steady stream now.

  # That's when I got electrocuted. The water leaked onto the handle of the steam cleaner when I pointed it upwards and I got electrocuted. Also, NOW  the steam cleaner won't turn on.
  # It made a bunch of noises and shut off.

  # I wouldn't touch it again anyway.
  # I'm not risking electrocution again.

  # Please provide me with a new steam cleaner or one that works or a refund. 

  # I  have health issues. I need the steam cleaner for my health. It cleaned my apartment very well by killing dust mites and bacteria and disinfecting my apartment. Once I use the steam cleaner I breathe better. But now it's broken completely. AND I got electrocuted!

  # I almost bought a different steam cleaner but I thought that your steam cleaner was better even though it cost more money. Please don't make me regret my choice of buying your product. I can't believe that I only had it for about 3 1/2 months and already it's broken. I can't believe that I got electrocuted!!!

  # Please find attached a screenshot of proof of my purchase of the Schenley steam cleaner from Amazon. 

  # I look forward to hearing from you.
  # Best regards,
  # Mary Kelly
  # '''}

  # result = bot.main(input,df)

  print('---------------------------\n',result)
