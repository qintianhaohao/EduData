#!/usr/bin/env python
# coding: utf-8

# # EdNet-KT1 Data Analysis
# 
# ## Columns Description
# 
# |  Field   | Annotation  |
# |  ----  | ----  |
# | user_id  | student's id |
# | timestamp  |  the moment the question was given, represented as Unix timestamp in milliseconds |
# | solving_id  | represents each learning session of students corresponds to each bunle. It is a form of single integer, starting from 1 |
# | question_id  | the ID of the question that given to student, which is a form of q{integer} |
# | user_answer  | the answer that the student submitted, recorded as a character between a and d inclusively |
# | elapsed_time  | the time that the students spends on each question in milliseconds |

# ## Statement for Our Data Set

# There are 784309 tables in our data set. Each table describes a student's question-solving log. There is no difference in the information dimension between the tables. Each table contains the `timestamp`,`solving_id`,`question_id`,`user_answer` and `elapsed_time` as described in the above `Columns Description` section.

# In[1]:


import numpy as np
import pandas as pd
import os
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

# 定义处理缺失值时使用的常量
NOT_CHOSEN = '未选择'
def analyze_and_process_dataframe(df):
    """
    对数据框进行基本的探索性分析和处理。

    参数:
    df (pandas.DataFrame): 需要分析和处理的数据框。

    返回:
    pandas.DataFrame: 处理后的数据框。
    """

    # 显示数据框的基本统计信息
    print("数据框的基本统计信息：")
    print(df.describe())

    # 计算并打印唯一问题ID的数量
    unique_question_ids = len(df.question_id.unique())
    print(f"总共有 {unique_question_ids} 个不同的问题。")

    # 检查并打印每列中缺失值的比例
    print('各列缺失值比例：')
    print(df.isnull().sum() / len(df))

    # 填充'user_answer'列中的缺失值
    df.fillna({'user_answer': NOT_CHOSEN}, inplace=True)

    return df


TOP_STUDENTS_COUNT = 40
CHART_TITLE = 'Top 40 active students'
def plot_top_active_students(data, top_n=TOP_STUDENTS_COUNT):
    """
    Plots the top N most active students based on their count.

    :param data: DataFrame containing user counts
    :param top_n: Number of top active students to display
    """

    # Sort by 'count' in descending order and select top N
    top_students = data.sort_values(by='count', ascending=False).head(top_n)

    # Create and show the bar chart
    fig = px.bar(
        top_students,
        x='count',
        y='user_id',
        orientation='h',
        title=CHART_TITLE
    )
    fig.show()


if __name__ == '__main__':
    # ## Record Example
    # We randomly selected 5000 tables from all the students for analysis,which accounted for about 0.64% of the total data set, and added a column named `user_id` to the original table

    path='../../dataset/EdNet/EdNet-KT1/KT1'
    d=[]
    table_list=[]
    s=pd.Series(os.listdir(path))
    file_selected=s.sample(5000).to_numpy()
    for file_name in file_selected:
        data_raw=pd.read_csv(path+'\\'+file_name,encoding = "ISO-8859-15")
        data_raw['user_id']=pd.Series([file_name[:-4]]*len(data_raw))
        d.append([file_name[:-4],len(data_raw)])
        data=pd.DataFrame(data_raw,columns=['user_id']+data_raw.columns.to_list()[:-1])
        table_list.append(data)
    df=pd.concat(table_list)
    pd.set_option('display.max_rows',10)
    df=df.reset_index(drop=True)

    df = analyze_and_process_dataframe(df)

    user_count_table = pd.DataFrame(d, columns=['user_id', 'count'])
    plot_top_active_students(user_count_table)


    # We use the number of questions that students have done as an indicator of whether a student is active. This figure shows the 40 most active students.

    # In[8]:

    ds=df.loc[:,['user_id','elapsed_time']].groupby('user_id').mean()
    ds=ds.reset_index(drop=False)
    ds.columns=['user_id','avg_elapsed_time']
    ds_tail=ds.sort_values(by=['avg_elapsed_time'],axis=0).tail(40)

    fig_tail = px.bar(
        ds_tail,
        x = 'avg_elapsed_time',
        y = 'user_id',
        orientation='h',
        title='Bottom 40 fast-solving students '
    )
    fig_tail.show()
    ds_head=ds.sort_values(by=['avg_elapsed_time'],axis=0).head(40)
    fig_head = px.bar(
        ds_head,
        x = 'avg_elapsed_time',
        y = 'user_id',
        orientation='h',
        title='Top 40 fast-solving students'
    )
    fig_head.show()


    # We take the average time it takes students to do a question as an indicator of how fast students do it. These two figures respectively show the fastest and slowest students among the 5000 students, and the average time they spent doing the problems.

    # Note that some students spend very little time doing the questions, and the time is almost zero. We can almost judge that these students did not do the questions at all, and they chose blindly. We remove these students and rearrange them

    # In[9]:


    bound=5000 # If the average time of doing the topic is less than 5000, it means that the student is most likely to be bad
    ds=df.loc[:,['user_id','elapsed_time']].groupby('user_id').mean()
    ds=ds.reset_index(drop=False)
    ds.columns=['user_id','avg_elapsed_time']
    bad_user_ids=ds[ds['avg_elapsed_time']<bound]['user_id'].to_list()
    df_drop=df.drop(df[df['user_id'].isin(bad_user_ids)].index)
    print('bad students number is ',len(bad_user_ids))
    print('length of table after dropping is ',len(df_drop))


    # ### After dropping

    # In[10]:


    ds=df_drop['user_id'].value_counts().reset_index(drop=False)
    ds.columns=['user_id','count']
    ds_tail=ds.sort_values(by=['count'],axis=0).tail(40)
    fig_tail = px.bar(
        ds_tail,
        x = 'count',
        y = 'user_id',
        orientation='h',
        title='Top 40 active students after dropping some students'
    )
    fig_tail.show()


    # This figure shows the 40 most active students after dropping some bad students.

    # In[11]:


    ds=df_drop.loc[:,['user_id','elapsed_time']].groupby('user_id').mean()
    ds=ds.reset_index(drop=False)
    ds.columns=['user_id','avg_elapsed_time']

    ds_head=ds.sort_values(by=['avg_elapsed_time'],axis=0).head(40)
    fig_head = px.bar(
        ds_head,
        x = 'avg_elapsed_time',
        y = 'user_id',
        orientation='h',
        title='Top 40 fast-solving students after dropping some students'
    )
    fig_head.show()


    # This figure respectively show the more reasonable fastest students among the 5000 students than before, and the average time they spent doing the problems.

    # ## Sort question_id

    # In[12]:


    ds=df.loc[:,['question_id','elapsed_time']].groupby('question_id').mean()
    ds=ds.reset_index(drop=False)
    ds_tail=ds.sort_values(by=['elapsed_time'],axis=0).tail(40)
    fig_tail = px.bar(
        ds_tail,
        x = 'elapsed_time',
        y = 'question_id',
        orientation='h',
        title='Top 40 question_id by the average of elapsed_time'
    )
    fig_tail.show("svg")
    ds_head=ds.sort_values(by=['elapsed_time'],axis=0).head(40)
    fig_head = px.bar(
        ds_head,
        x = 'elapsed_time',
        y = 'question_id',
        orientation='h',
        title='Bottom 40 question_id by the average of elapsed_time'
    )
    fig_head.show("svg")


    # We can judge the difficulty of this question from the average time spent on a question.
    # These two figures reflect the difficulty of the questions and shows the ids of the 40 most difficult and 40 easiest questions.s

    # ## Appearence of Questions

    # In[13]:


    ds=df['question_id'].value_counts().reset_index(drop=False)
    ds.columns=['question_id','count']
    ds_tail=ds.sort_values(by=['count'],axis=0).tail(40)
    fig_tail = px.bar(
        ds_tail,
        x = 'count',
        y = 'question_id',
        orientation='h',
        title='Top 40 question_id by the number of appearance'
    )
    fig_tail.show("svg")
    ds_head=ds.sort_values(by=['count'],axis=0).head(40)
    fig_head = px.bar(
        ds_head,
        x = 'count',
        y = 'question_id',
        orientation='h',
        title='Bottom 40 question_id by the number of appearance'
    )
    fig_head.show("svg")


    # These two images reflect the 40 questions that were drawn the most frequently and the 40 questions that were drawn the least frequently

    # In[14]:


    ds2=df['question_id'].value_counts().reset_index(drop=False)
    ds2.columns=['question_id','count']
    def convert_id2int(x):
        return pd.Series(map(lambda t:int(t[1:]),x))
    ds2['question_id']=convert_id2int(ds2['question_id'])
    ds2.sort_values(by=['question_id'])
    fig = px.histogram(
        ds2,
        x = 'question_id',
        y = 'count',
        title='question distribution'
    )
    fig.show("svg")


    # ##  Question's Option Selected Most Frequently

    # In[15]:


    ds=df.loc[:,['question_id','user_answer','user_id']].groupby(['question_id','user_answer']).count()

    most_count_dict={}
    for id in df.question_id.unique():
        most_count=ds.loc[id].apply(lambda x:x.max())[0]
        most_count_dict[id]=most_count
    ds2=ds.apply(lambda x:x-most_count_dict[x.name[0]],axis=1)
    ds2=ds2[ds2.user_id==0]
    ds2=ds2.reset_index(drop=False).loc[:,['question_id','user_answer']]
    ds2.columns=['question_id','most_answer']
    ds2.index=ds2['question_id']
    ds2['most_answer']


    # This shows the most selected options (including `not choose`) for each question.
    # Note that if there are multiple options for a question to be selected most frequently, the table will also contain them.

    # ## Choices Distribution

    # In[16]:


    ds = df['user_answer'].value_counts().reset_index(drop=False)
    ds.columns = ['user_answer', 'percent']

    ds['percent']=ds['percent']/len(df)
    ds = ds.sort_values(by=['percent'])

    fig = px.pie(
        ds,
        names = ds['user_answer'],
        values = 'percent',
        title = 'Percent of Choice'
    )

    fig.show("svg")


    # We use a pie chart to show the distribution of the proportions of `a`, `b`, `c`, `d` and `not choose` among the options selected by the 5000 students.

    # ## Sort By Time Stamp

    # In[17]:


    import time
    import datetime


    # In[18]:


    df_time=df.copy()
    columns=df.columns.to_list()
    columns[1]='time'
    df_time.columns=columns
    df_time['time'] /= 1000
    df_time['time']=pd.Series(map(datetime.datetime.fromtimestamp,df_time['time']))
    df_time


    # This table shows the result of converting unix timestamp to datetime format

    # ### question distribution by time

    # In[19]:


    ds_time_question=df_time.loc[:,['time','question_id']]
    ds_time_question=ds_time_question.sort_values(by=['time'])
    ds_time_question


    # This table shows the given questions in chronological order.And we can see that the earliest question q127 is on May 11, 2017, and the latest question q10846 is on December 3, 2019.

    # In[20]:


    ds_time_question['year']=pd.Series(map(lambda x :x.year,ds_time_question['time']))
    ds_time_question['month']=pd.Series(map(lambda x :x.month,ds_time_question['time']))
    ds=ds_time_question.loc[:,['year','month']].value_counts()

    years=ds_time_question['year'].unique()
    years.sort()
    fig=make_subplots(
        rows=2,
        cols=2,
        start_cell='top-left',
        subplot_titles=tuple(map(str,years))
    )
    traces=[
        go.Bar(
            x=ds[year].reset_index().sort_values(by=['month'],axis=0)['month'].to_list(),
            y=ds[year].reset_index().sort_values(by=['month'],axis=0)[0].to_list(),
            name='Year: '+str(year),
            text=[ds[year][month] for month in ds[year].reset_index().sort_values(by=['month'],axis=0)['month'].to_list()],
            textposition='auto'
        ) for year in years
    ]
    for i in range(len(traces)):
        fig.append_trace(traces[i],(i//2)+1,(i%2)+1)

    fig.update_layout(title_text='Bar of the distribution of the number of question solved in {} years'.format(len(traces)))
    fig.show('svg')


    # 1. These three figures show the distribution of the number of problems solved in each month of 2017, 2018, and 2019.
    # 2. And the number of questions solved is gradually increasing.
    # 3. And the number of questions solved in March, 4, May, and June is generally small.

    # ### user distribution by time

    # In[21]:


    ds_time_user=df_time.loc[:,['user_id','time']]
    ds_time_user=ds_time_user.sort_values(by=['time'])
    ds_time_user


    # This table shows the students who did the questions in order of time.And we can see that the first student who does the problem is u21056, and the last student who does the problem is u9476.

    # In[22]:


    ds_time_user=df_time.loc[:,['user_id','time']]
    ds_time_user=ds_time_user.sort_values(by=['time'])
    ds_time_user['year']=pd.Series(map(lambda x :x.year,ds_time_user['time']))
    ds_time_user['month']=pd.Series(map(lambda x :x.month,ds_time_user['time']))
    ds_time_user.drop(['time'],axis=1,inplace=True)
    ds=ds_time_user.groupby(['year','month']).nunique()

    years=ds_time_user['year'].unique()
    years.sort()
    fig=make_subplots(
        rows=2,
        cols=2,
        start_cell='top-left',
        subplot_titles=tuple(map(str,years))
    )
    traces=[
        go.Bar(
            x=ds.loc[year].reset_index()['month'].to_list(),
            y=ds.loc[year].reset_index()['user_id'].to_list(),
            name='Year: '+str(year),
            text=[ds.loc[year].loc[month,'user_id'] for month in ds.loc[year].reset_index()['month'].to_list()],
            textposition='auto'
        ) for year in years
    ]
    for i in range(len(traces)):
        fig.append_trace(traces[i],(i//2)+1,(i%2)+1)

    fig.update_layout(title_text='Bar of the distribution of the number of active students in {} years'.format(len(traces)))
    fig.show('svg')


    # 1. These three graphs respectively show the number of students active on the system in each month of 2017, 2018, and 2019.
    # 2. And we can see that the number of active students in 2019 is generally more than that in 2018, and there are more in 2018 than in 2017, indicating that the number of users of the system is gradually increasing.
    # 3. Note that the number of students is not repeated here


