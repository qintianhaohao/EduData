
#!/usr/bin/env python
# coding: utf-8

# # KDD Cup 2010 —— Data Analysis on algebra_2006_2007_train

import pandas as pd
import plotly.express as px
# ## Data Description

# ### Column Description

# | Attribute | Annotaion |
# |:--:|---|
# |Row|The row number|
# | Anon Student Id             | Unique, anonymous identifier for a student    |
# | Problem Hierarchy           | The hierarchy of curriculum levels containing the problem |
# | Problem Name                | Unique identifier for a problem |
# | Problem View                | The total number of times the student encountered the problem so far |
# | Step Name                   | Unique identifier for one of the steps in a problem |
# | Step Start Time             | The starting time of the step (Can be null) |
# | First Transaction Time      | The time of the first transaction toward the step |
# | Correct Transaction Time    | The time of the correct attempt toward the step, if there was one |
# | Step End Time               | The time of the last transaction toward the step |
# | Step Duration (sec)         | The elapsed time of the step in seconds, calculated by adding all of the durations for transactions that were attributed to the step (Can be null if step start time is null) |
# | Correct Step Duration (sec) | The step duration if the first attempt for the step was correct |
# | Error Step Duration (sec)   | The step duration if the first attempt for the step was an error (incorrect attempt or hint request) |
# | Correct First Attempt       | The tutor's evaluation of the student's first attempt on the step—1 if correct, 0 if an error |
# | Incorrects                  | Total number of incorrect attempts by the student on the step |
# | Hints                       | Total number of hints requested by the student for the step |
# | Corrects                    | Total correct attempts by the student for the step (only increases if the step is encountered more than once) |
# | KC(KC Model Name)           | The identified skills that are used in a problem, where available |
# | Opportunity(KC Model Name)  | A count that increases by one each time the student encounters a step with the listed knowledge component |
# || Additional KC models, which exist for the challenge data sets, will appear as additional pairs of columns (KC and Opportunity columns for each model) |

# For the test portion of the challenge data sets, values will not be provided for the following columns:

# &diams; Step Start Time
# 
# &diams; First Transaction Time
# 
# &diams; Correct Transaction Time
# 
# &diams; Step End Time
# 
# &diams; Step Duration (sec)
# 
# &diams; Correct Step Duration (sec)
# 
# &diams; Error Step Duration (sec)
# 
# &diams; Correct First Attempt
# 
# &diams; Incorrects
# 
# &diams; Hints
# 
# &diams; Corrects

# In[1]:


def load_data():
    # 加载数据集
    path = "../../dataset/KDD_Cup_2010/algebra_2006_2007/algebra_2006_2007_train.txt"
    data = pd.read_table(path, encoding="ISO-8859-15", low_memory=False)
    return data

def show_record_examples(data):
    pd.set_option('display.max_column', 500)

def show_record_examples(data):
    # 展示数据样本，描述统计和缺失值比例
    pd.set_option('display.max_column', 500)
    print(data.head())
    print(data.describe())
    print("Part of missing values for every column")
    print(data.isnull().sum() / len(data))
    print("the number of records:")
    print(len(data))
    print("how many students are there in the table:")
    print(len(data['Anon Student Id'].unique()))
    print("how many problems are there in the table:")
    print(len(data['Problem Name'].unique()))

def sort_by_student_id(data):
    ds = data['Anon Student Id'].value_counts().reset_index()

def sort_by_student_id(data):
    # 按学生ID排序并显示前40名学生的问题步骤数量
    ds = data['Anon Student Id'].value_counts().reset_index()
    ds.columns = ['Anon Student Id', 'count']
    ds['Anon Student Id'] = ds['Anon Student Id'].astype(str) + '-'
    ds = ds.sort_values('count').tail(40)
    fig = px.bar(ds, x='count', y='Anon Student Id', orientation='h', title='Top 40 students by number of steps they have done')
    fig.show()

def calculate_percentages(data):
    count_corrects = data['Corrects'].sum()

def calculate_percentages(data):
    # 计算正确、提示和错误尝试的百分比
    count_corrects = data['Corrects'].sum()
    count_hints = data['Hints'].sum()
    count_incorrects = data['Incorrects'].sum()
    total = count_corrects + count_hints + count_incorrects
    percent_corrects = count_corrects / total
    percent_hints = count_hints / total
    percent_incorrects = count_incorrects / total
    dfl = [['corrects', percent_corrects], ['hints', percent_hints], ['incorrects', percent_incorrects]]
    df = pd.DataFrame(dfl, columns=['transaction type', 'percent'])
    fig = px.pie(df, names=['corrects', 'hints', 'incorrects'], values='percent', title='Percent of corrects, hints and incorrects')
    fig.show()

def sort_by_problem_name(data):
    storeProblemCount = [1]

def sort_by_problem_name(data):
    # 按问题名称排序并显示最有用的40个问题
    storeProblemCount = [1]
    storeProblemName = [data['Problem Name'][0]]
    currentProblemName = data['Problem Name'][0]
    currentStepName = [data['Step Name'][0]]
    lastIndex = 0
    for i in range(1, len(data), 1):
        pbNameI = data['Problem Name'][i]
        stNameI = data['Step Name'][i]
        if pbNameI != data['Problem Name'][lastIndex]:
            currentStepName = [stNameI]
            currentProblemName = pbNameI
            if pbNameI not in storeProblemName:
                storeProblemName.append(pbNameI)
                storeProblemCount.append(1)
            else:
                storeProblemCount[storeProblemName.index(pbNameI)] += 1
            lastIndex = i
        elif stNameI not in currentStepName:
            currentStepName.append(stNameI)
            lastIndex = i
        else:
            currentStepName = [stNameI]
            storeProblemCount[storeProblemName.index(pbNameI)] += 1
            lastIndex = i
    dfData = {'Problem Name': storeProblemName, 'count': storeProblemCount}
    df = pd.DataFrame(dfData).sort_values('count').tail(40)
    df["Problem Name"] += '-'
    fig = px.bar(df, x='count', y='Problem Name', orientation='h', title='Top 40 useful problem')
    fig.show()

def analyze_problem_correct_rate(data):
    data['total transactions'] = data['Incorrects'] + data['Hints'] + data['Corrects']

def analyze_problem_correct_rate(data):
    # 分析每个问题的正确率
    data['total transactions'] = data['Incorrects'] + data['Hints'] + data['Corrects']
    df1 = data.groupby('Problem Name')['total transactions'].sum().reset_index()
    df2 = data.groupby('Problem Name')['Corrects'].sum().reset_index()
    df1['Corrects'] = df2['Corrects']
    df1['Correct rate'] = df1['Corrects'] / df1['total transactions']
    df1 = df1.sort_values('total transactions')
    count = 0
    standard = 500
    for i in df1['total transactions']:
        if i > standard:
            count += 1
    df1 = df1.tail(count)
    df1 = df1.sort_values('Correct rate')
    df1['Problem Name'] = df1['Problem Name'].astype(str) + "-"
    df_px = df1.tail(20)
    fig = px.bar(df_px, x='Correct rate', y='Problem Name', orientation='h', title='Correct rate of each problem (top 20)  (total transactions of each problem are required to be more than 500)', text='Problem Name')
    fig.update_layout(title_font_size=10)
    fig.show()
    df_px = df1.head(20)
    fig = px.bar(df_px, x='Correct rate', y='Problem Name', orientation='h', title='Correct rate of each problem (bottom 20)  (total transactions of each problem are required to be more than 500)', text='Problem Name')
    fig.update_layout(title_font_size=10)
    fig.show()


def analyze_kc_correct_rate(data):
    data.dropna(subset=['KC(Default)'], inplace=True)

def analyze_kc_correct_rate(data):
    # 分析每个知识组件(KC)的正确率
    data.dropna(subset=['KC(Default)'], inplace=True)
    data['total transactions'] = data['Corrects'] + data['Hints'] + data['Incorrects']
    df1 = data.groupby('KC(Default)')['total transactions'].sum().reset_index()
    df2 = data.groupby('KC(Default)')['Corrects'].sum().reset_index()
    df1['Corrects'] = df2['Corrects']
    df1['correct rate'] = df1['Corrects'] / df1['total transactions']
    count = 0
    standard = 300
    for i in df1['total transactions']:
        if i > standard:
            count += 1
    df1 = df1.sort_values('total transactions').tail(count)
    df1 = df1.sort_values('correct rate')
    df1['KC(Default)'] = df1['KC(Default)'].astype(str) + '-'
    df_px = df1.tail(20)
    fig = px.bar(df_px, x='correct rate', y='KC(Default)', orientation='h', title='Correct rate of each KC(Default) (top 20)  (total transactions of each KC are required to be more than 300)', text='KC(Default)')
    fig.update_yaxes(visible=False)
    fig.update_layout(title_font_size=10)
    fig.show()
    df_px = df1.head(20)
    fig = px.bar(df_px, x='correct rate', y='KC(Default)', orientation='h', title='Correct rate of each KC(Default) (bottom 20)  (total transactions of each KC are required to be more than 300)', text='KC(Default)')
    fig.update_yaxes(visible=False)
    fig.update_layout(title_font_size=10)
    fig.show()

if __name__ == '__main__':
    data = load_data()
    show_record_examples(data)
    sort_by_student_id(data)
    calculate_percentages(data)
    sort_by_problem_name(data)
    analyze_problem_correct_rate(data)
    analyze_kc_correct_rate(data)




