#!/usr/bin/env python
# coding: utf-8

# # Math23k Analysis Report

# ## Data Description
# | Field             | Annotation                                          |
# | --------          | --------------------------------------------------- |
# | id                | Id of the problem |
# | original_text	    | Original text of the problem |
# | equation          | Solution to the problem |
# | segmented_text    | Chinese word segmentation of the problem |
# 

# In[2]:


import numpy as np
import pandas as pd
import jieba

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go


# In[3]:


path1 = "../dataset/math23k/raw/train23k.json"
path2 = "../dataset/math23k/raw/test23k.json"
path3 = "../dataset/math23k/raw/valid23k.json"

data  = pd.read_json(path1, orient='records')
data2 = pd.read_json(path2, orient='records')
data3 = pd.read_json(path3, orient='records')
data = pd.concat([data, data2, data3])


# ## Record Examples

# In[4]:


data.head()


# ## The number of problems

# In[5]:


len(data['id'].unique())


# ## Part of missing values for every column

# In[6]:


data.isnull().sum() / len(data)


# ## Cut words and find verbs in problems

# Verbs may be quite useful for solving math word problems, because sometimes a verb means an operator in equation. 

# In[7]:


import jieba.posseg as pseg
def cut_word(text):
    return jieba.lcut(text)

def find_verbs(text):
    words = pseg.cut(text)
    return [word for word,flag in words if flag == 'v']

data['content']=data['original_text'].apply(cut_word)
data['verbs']=data['original_text'].apply(find_verbs)
data.head()


# ## Count of words of problems

# In[8]:


def getsize(ser):
    return len(ser)

data['word_count']=data['content'].apply(getsize)
data.head()


# ## The length of problems

# This picture shows that the length of most problems are about 20 to 40 chinese words.It may be helpful for design of model's input 

# In[9]:


cnt = data['word_count'].value_counts().reset_index()
cnt.columns = [ 'word_count' , 'problem_count']

fig = px.bar(
    cnt , x = 'word_count', y = 'problem_count' ,
    title = 'The length of problems'
)
fig.show()


# ## Delete stopword

# In[ ]:





# In[10]:


def get_stopword():
    s = set()
    with open("../dataset/stopword/stopword.txt","r",encoding="UTF-8") as f:
        for line in f:
            s.add(line.strip())
    return s

def delete_stopword(words):
    return [w for w in words if (w not in stopword)]

stopword=get_stopword()
data['content']=data['content'].apply(delete_stopword)
data.head()


# ## The keywords

# Keywords may show us the topic of problem sometimes.They are useful for our analysis.
# This report use textrank algorithm in 'jieba'. because length of problem are usually short ,TF/IDF may be not suitable for this dataset.

# In[ ]:


import jieba.analyse
def get_keyword(text):
    topk = min(3,len(text))
    keyword = [word for word in jieba.analyse.textrank(text, topK = topk)]
    return keyword

data['keywords'] = data['original_text'].apply(get_keyword)
data.head()


# ## Topic Prediction

# Classify problems by their topics may be helpful for models and analyse the result of models in different fields.
# Because there're no labels in original data, unsupervised algorithm LDA may be suitable.

# In[ ]:


from gensim import corpora, models

all_words = []
for text in data['content']:
    all_words.append(text)
#print(all_words)
dictionary = corpora.Dictionary(all_words)
corpus = [dictionary.doc2bow(text) for text in all_words]

lda = models.ldamodel.LdaModel(corpus = corpus, id2word = dictionary, num_topics = 5)

print('keywords of topics')
for topic in lda.print_topics(num_words = 5):
    print(topic)


# In[ ]:


topic = []
for i,values in enumerate(lda.inference(corpus)[0]):
    topic_val = 0
    topic_id = 0
    for tid, val in enumerate(values):
        if val > topic_val:
            topic_val = val
            topic_id = tid
    topic.append(topic_id)
data['topic'] = topic
data.head(10)


# In[ ]:


output = data['topic'].value_counts().reset_index()
output.columns=['topic_id','number of problems']

fig = px.pie(
    output,
    names = 'topic_id',
    values = 'number of problems',
    title = 'Topic of problems'    
)

fig.show("svg")


# ## Number of operators

# If you know how many operators are there in equations, it may be much easier for you to solve math word problems.Especially when your algorithm are based on equation templates.

# In[ ]:


def num_of_operators(equation):
    cnt = 0
    for op in equation:
        if op =='(' or op == '+' or op == '-' or op == '*' or op == '/' or op == '^':
            cnt += 1
    return cnt

tmp = data.loc[:,['equation']]
tmp['operators_cnt'] = tmp['equation'].apply(num_of_operators)
cnt = tmp['operators_cnt'].value_counts().reset_index()
output = cnt.head(10)
other_sum = cnt['operators_cnt'].sum() - output['operators_cnt'].sum()

output = output.sort_values(['operators_cnt'])
output.loc[10] = ['other', other_sum]

output.columns=['number of operators','number of problems']

fig = px.pie(
    output,
    names = 'number of operators',
    values = 'number of problems',
    title = 'Number of operators'    
)

fig.show("svg")


# ## Evaluate difficulty

# Different problems have different difficulty.People may choose different way to solve problems when difficulty of problems are different,and so is AI.To evaluate difficulty of problems, the kinds of operators in equations may be useful.Value of them are as follows.

# In[ ]:


def calc_difficulty(equation):
    difficulty = 0

    def eval(x):
        if x == '+' : return 2
        elif x == '-' : return 3
        elif x == '*' : return 5
        elif x == '/' : return 7
        elif x == '(' : return 8
        elif x == '%' : return 5
        elif x == '^' : return 6
        else : return 0

    for op in equation:
        difficulty += eval(op)
    return difficulty


data['difficulty'] = data['equation'].apply(calc_difficulty)

cnt = data['difficulty'].value_counts().reset_index()
cnt.columns = [ 'difficulty' , 'problem_count']

fig = px.bar(
    cnt , x = 'difficulty', y = 'problem_count' ,
    title = 'The difficulty of problems'
)
fig.show()


# ## The most difficult problems

# In[ ]:


tmp = data[['id','original_text','difficulty']]
tmp = tmp.sort_values(['difficulty']).tail(10)
tmp ['id'] = tmp['id'] . astype(str)
fig = px.bar(
    tmp , x = 'difficulty', y = 'id' ,
    orientation = 'h',
    title = 'The difficulty of problems'
)
fig.show()


# ## Simplify expressions
# 
# Algorithm based on templates will find templates in equations at first.To find templates, we should simplify expressions first.'+' means operator '+' or '-', '\*' means operator '\*' or '/', 'n' means a number.
# 

# In[ ]:


from pythonds.basic.stack import Stack


def simplify(expr): 
    n = len(expr)
    output = ''
    flag = True
    for i in range(2,n):
        if flag and (expr[i].isdigit() or expr[i] == '.' or expr[i] == '%'):
            output = output + 'n'
            flag = False
        if not (expr[i].isdigit() or expr[i] == '.' or expr[i] == '%'):
            if expr[i] == '[' or expr[i] == '{':
                output = output + '('
            elif expr[i] == ']' or expr[i] == '}':
                output = output + ')'
            elif expr[i] == '-':
                output = output + '+'
            elif expr[i] == '/':
                output = output + '*'
            else: output = output + expr[i]
            flag = True
    return output
    
data['post_expression'] = data['equation'].apply(simplify)


# ## Count of numbers in equations

# In[ ]:


def CountNum(expr):
    cnt = 0
    for x in expr:
        if x == 'n':
            cnt = cnt + 1
    return cnt


tmp = data.loc[:,['post_expression','original_text']]
tmp['number_cnt'] = tmp['post_expression'].apply(CountNum)


cnt = tmp['number_cnt'].value_counts().reset_index()
output = cnt.head(10)
other_sum = cnt['number_cnt'].sum() - output['number_cnt'].sum()

output = output.sort_values(['number_cnt'])
output.loc[10] = ['other', other_sum]

output.columns=['count of numbers','number of problems']

fig = px.pie(
    output,
    names = 'count of numbers',
    values = 'number of problems',
    title = 'Count of numbers in equations'    
)

fig.show("svg")    




# ## Are numbers in equations as many as in problems?
# This result shows that about half of problems have useless parameters or potential parameters in problems

# In[ ]:


def NuminProb(text):
    prob = str(text)
    cnt = 0
    flag = True
    
    for w in prob:
        if w.isdigit() or w == '.' or w == '%':
            if flag:
                cnt += 1
                flag = False
        else:
            flag =True
    return cnt

def isSame(a, b):
    if a == b:
        return True
    else:
        return False

tmp['num_in_prob'] = tmp['original_text'].apply(NuminProb)
tmp['same count'] = tmp.apply(lambda row: isSame(row['number_cnt'], row['num_in_prob']), axis=1)
same = tmp['same count'].value_counts().reset_index()

fig = px.pie(
    same,
    names = 'index',
    values = 'same count',
    title = 'Are numbers in equation as many as in problems?'    
)
fig.show("svg")   


# ## Postfix expressions
# Some algorithm need postfix expressions instead of infix expressions.The reasons for that may be postfix expressions can help us build expression trees,and there are no brackets in postfix expressions,so postfix expressions can merge some template.

# In[ ]:


def InfixToPostfix(infixexpr):
    prec = {}
    prec['^'] = 4
    prec["*"] = 3
    prec["/"] = 3
    prec["+"] = 2
    prec["-"] = 2
    prec["("] = 1

    opstack = Stack()
    postfixList = []

    for token in infixexpr:
        if token == 'n':
            postfixList.append(token)
        elif token == "(":
            opstack.push(token)
        elif token == ")":
            topstack = opstack.pop()
            while topstack != "(":
                postfixList.append(topstack)
                if opstack.isEmpty():
                    print(infixexpr)
                else :
                    topstack = opstack.pop()
        else:
            while (not opstack.isEmpty()) and (prec[opstack.peek()] >= prec[token]):
                postfixList.append(opstack.pop())
            opstack.push(token)
    while not opstack.isEmpty():
        postfixList.append(opstack.pop())
    return ''.join(postfixList)


data['post_expression'] = data['post_expression'].apply(InfixToPostfix)

data.head()


# ## Templates of postfix expressions

# Template may be useful to solve math word problems. In fact,many algorithms are based on them.The result shows that 15 kinds of postfix templates can help us solve about 70% of problems.

# In[ ]:


ds = data['post_expression'].value_counts().reset_index()
ds = ds.sort_values(['post_expression'])

output = ds.tail(15)
other_sum = ds['post_expression'].sum() - output['post_expression'].sum()

output.columns = [
    'post_expression',
    'percent'
]

output = output.sort_values(['percent'])
output.loc[15] = ['others', other_sum]

fig = px.pie(
    output,
    names = 'post_expression',
    values = 'percent',
    title = 'Templates of postfix expressions',
)

fig.show("svg")


# ## Reference
# @inproceedings{Liu2019TreestructuredDF,
#   title={Tree-structured Decoding for Solving Math Word Problems},
#   author={Qianying Liu and Wenyv Guan and Sujian Li and Daisuke Kawahara},
#   booktitle={EMNLP/IJCNLP},
#   year={2019}
# }
# 

# @inproceedings{Xie2019AGT,
#   title={A Goal-Driven Tree-Structured Neural Model for Math Word Problems},
#   author={Zhipeng Xie and Shichao Sun},
#   booktitle={IJCAI},
#   year={2019}
# }

# @inproceedings{Wang2019TemplateBasedMW,
#   title={Template-Based Math Word Problem Solvers with Recursive Neural Networks},
#   author={Lei Wang and D. Zhang and Jipeng Zhang and Xing Xu and L. Gao and B. Dai and H. Shen},
#   booktitle={AAAI},
#   year={2019}
# }

# @article{Lee2020SolvingAW,
#   title={Solving Arithmetic Word Problems with a Templatebased Multi-Task Deep Neural Network (T-MTDNN)},
#   author={D. Lee and G. Gweon},
#   journal={2020 IEEE International Conference on Big Data and Smart Computing (BigComp)},
#   year={2020},
#   pages={271-274}
# }

