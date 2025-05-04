#!/usr/bin/env python
# coding: utf-8

# # ASSISTments2017 Data Analysis
# <https://edudata.readthedocs.io/en/latest/build/blitz/ASSISTments/ASSISTments2009-2010.html>

# ## Data Description
# 
# ### Column Description
# 
# 
# | Field    | Annotation                                          |
# | :- | :- |
# | student id | 	a deidentified ID/tag used for identifying an individual student |
# | SY ASSISTments Usage | the academic years the student used ASSISTments |
# | AveKnow |  average student knowledge level (according to Bayesian Knowledge Tracing algorithm -- cf. Corbett & Anderson, 1995) |
# | AveCarelessness | average student carelessness (according to San Pedro, Baker, & Rodrigo, 2011 model) |
# | AveCorrect | average student correctness |
# | NumActions | total number of student actions in system |
# | AveResBored |  average student affect: boredom (see Pardos, Baker, San Pedro, Gowda, & Gowda, 2014) |
# | AveResEngcon | average student affect:engaged concentration (see Pardos, Baker, San Pedro, Gowda, & Gowda, 2014)|
# | AveResConf | average student affect:confusion (see Pardos, Baker, San Pedro, Gowda, & Gowda, 2014) |
# | AveResFrust | average student affect:frustration (see Pardos, Baker, San Pedro, Gowda, & Gowda, 2014) |
# | AveResOfftask | average student affect: off task (see Pardos, Baker, San Pedro, Gowda, & Gowda, 2014 and also Baker, 2007) |
# | AveResGaming | average student affect:gaming the system  (see Pardos, Baker, San Pedro, Gowda, & Gowda, 2014 and also Baker Corbett Koedinger & Wagner, 2004) |
# | actionId | the unique id of this specific action | 
# | skill |  a tag used for identifying the cognitive skill related to the problem (see Razzaq, Heffernan, Feng, & Pardos, 2007) |
# | problemId | a unique ID used for identifying a single problem |
# | assignmentId | a unique ID used for identifying an assignment |
# | assistmentId |  a unique ID used for identifying an assistment (a instance of a multi-part problem) |
# | startTime | when did the student start the problem (UNIX time, seconds) |
# |  endTime | when did the student end the problem (UNIX time, seconds) |
# | timeTaken | Time spent on the current step |
# | correct | Answer is correct |
# | original | Problem is original not a scaffolding problem |
# |  hint | Action is a hint response |
# | hintCount | Total number of hints requested so far |
# | hintTotal | total number of hints requested for the problem |
# | scaffold | Problem is a scaffolding problem |
# | bottomHint | Bottom-out hint is used |
# | attemptCount | Total problems attempted in the tutor so far. |
# | problemType | the type of the problem |
# | frIsHelpRequest | First response is a help request |
# | frPast5HelpRequest | Number of last 5 First responses that included a help request |
# | frPast8HelpRequest | Number of last 8 First responses that included a help request |
# | stlHintUsed | Second to last hint is used an indicates a hint that gives considerable detail but is not quite bottom-out |
# | past8BottomOut | Number of last 8 problems that used the bottom-out hint. |
# | totalFrPercentPastWrong | Percent of all past problems that were wrong on this KC. |
# | totalFrPastWrongCount | Total first responses wrong attempts in the tutor so far. |
# | frPast5WrongCount | Number of last 5 First responses that were wrong |
# | frPast8WrongCount | Number of last 8 First responses that were wrong |
# | totalFrTimeOnSkill | Total first response time spent on this KC across all problems |
# | timeSinceSkill | Time since the current KC was last seen. |
# | frWorkingInSchool | First response Working during school hours (between 7:00 am and 3:00 pm) |
# | totalFrAttempted | Total first responses attempted in the tutor so far. |
# | totalFrSkillOpportunities | Total first response practice opportunities on this KC so far. |
# | responseIsFillIn | Response is filled in (No list of answers available) | 
# | responseIsChosen | Response is chosen from a list of answers (Multiple choice, etc). |
# | endsWithScaffolding | Problem ends with scaffolding |
# | endsWithAutoScaffolding  | Problem ends with automatic scaffolding |
# | frTimeTakenOnScaffolding | First response time taken on scaffolding problems |
# | frTotalSkillOpportunitiesScaffolding | Total first response practice opportunities on this skill so far |
# | totalFrSkillOpportunitiesByScaffolding | Total first response scaffolding opportunities for this KC so far |
# | frIsHelpRequestScaffolding | First response is a help request Scaffolding |
# | timeGreater5Secprev2wrong | Long pauses after 2 Consecutive wrong answers |
# | sumRight | NaN |
# | helpAccessUnder2Sec | Time spent on help was under 2 seconds |
# | timeGreater10SecAndNextActionRight | Long pause after correct answer |
# | consecutiveErrorsInRow | Total number of 2 wrong answers in a row across all the problems |
# | sumTime3SDWhen3RowRight | NaN |
# | sumTimePerSkill | NaN |
# | totalTimeByPercentCorrectForskill | Total time spent on this KC across all problems divided by percent correct for the same KC |
# | prev5count | NaN |
# | timeOver80 | NaN |
# | manywrong | NaN |
# | confidence(BORED) | the confidence of the student affect prediction: bored |
# | confidence(CONCENTRATING) | the confidence of the student affect prediction: concecntrating |
# | confidence(CONFUSED) | the confidence of the student affect prediction: confused |
# | confidence(FRUSTRATED) | the confidence of the student affect prediction: frustrated  |
# | confidence(OFF TASK) | the confidence of the student affect prediction: off task |
# | confidence(GAMING) | the confidence of the student affect prediction: gaming |
# | RES_BORED | rescaled of the confidence of the student affect prediction: boredom |
# | RES_CONCENTRATING | rescaled of the confidence of the student affect prediction: concentration |
# | RES_CONFUSED | rescaled of the confidence of the student affect prediction: confusion |
# | RES_FRUSTRATED | rescaled of the confidence of the student affect prediction: frustration |
# | RES_OFFTASK | rescaled of the confidence of the student affect prediction: off task |
# | RES_GAMING | rescaled of the confidence of the student affect prediction: gaming |
# | Ln-1 | baysian knowledge tracing's knowledge estimate at the previous time step |
# | Ln | baysian knowledge tracing's knowledge estimate at the time step |
# | schoolID | the id (anonymized) of the school the student was in during the year the data was collected |
# | MCAS | Massachusetts Comprehensive Assessment System test score. In short, this number is the student's state test score (outside ASSISTments) during that year. -999 represents the data is missing |


import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go


def load_data(path):
    return pd.read_csv(path, encoding="ISO-8859-15", low_memory=False)


def describe_data(data):
    pd.set_option('display.max_columns', 500)
    print('data head: \n', data.head())
    print('data description: \n', data.describe())
    print("The number of records: " + str(len(data['action_num'].unique())))
    print('Part of missing values for every column')
    print(data.isnull().sum() / len(data))
    print(f"Unique students: {len(data.studentId.unique())}")
    print(f"Unique middle schools: {len(data.MiddleSchoolId.unique())}")


def plot_bar(data, column, title, orientation='h', tail=None):
    ds = data[column].value_counts().reset_index()
    ds.columns = [column, 'count']
    if orientation == 'h':
        ds[column] = ds[column].astype(str) + '-'
    ds = ds.sort_values(['count']).tail(tail) if tail else ds.sort_values(['count'])
    fig = px.bar(ds, x='count', y=column, orientation=orientation, title=title)
    fig.show()


def plot_pie(data, column, title):
    ds = data[column].value_counts().reset_index()
    ds.columns = [column, 'percent']
    ds['percent'] /= len(data)
    ds = ds.sort_values(['percent'])
    fig = px.pie(ds, names=column, values='percent', title=title)
    fig.show()


def plot_histogram(data, column, title):
    ds = data[column].value_counts().reset_index()
    ds.columns = [column, 'count']
    ds = ds.sort_values(column)
    fig = px.histogram(ds, x=column, y='count', title=title)
    fig.show()

def main():

    # 主程序
    path = "../../../dataset/anonymized_full_release_competition_dataset/anonymized_full_release_competition_dataset.csv"
    data = load_data(path)

    describe_data(data)

    # 绘制学生ID相关图表
    plot_bar(data, 'studentId', '按照行动次数排名的前40名学生', tail=40)
    plot_histogram(data, 'studentId', 'User action distribution')

    # 绘制中学ID相关图表
    plot_pie(data, 'MiddleSchoolId', 'Percent of schools')

    # 绘制正确答案相关图表
    plot_pie(data, 'correct', 'Percent of correct answers')

    # 绘制问题ID相关图表
    plot_bar(data, 'problemId', 'Top 40 useful problem ids', tail=40)
    plot_histogram(data, 'problemId', 'Problem ID action distribution')

    # 绘制问题类型相关图表
    plot_pie(data, 'problemType', 'Percent of problem types')

    ds = data['problemType'].value_counts().reset_index()
    ds.columns = ['problemType', 'percent']
    ds['percent'] /= len(data)
    ds = ds.sort_values(['percent']).tail(6)
    fig = make_subplots(rows=3, cols=2)
    traces = [
        go.Bar(
            x=['wrong', 'right'],
            y=[
                len(data[(data['problemType'] == item) & (data['correct'] == 0)]),
                len(data[(data['problemType'] == item) & (data['correct'] == 1)])
            ],
            name=f'Type: {item}',
            text=[f'Wrong: {len(data[(data["problemType"] == item) & (data["correct"] == 0)])}',
                  f'Right: {len(data[(data["problemType"] == item) & (data["correct"] == 1)])}']
        ) for item in ds['problemType']
    ]
    for i, trace in enumerate(traces):
        fig.add_trace(trace, row=(i // 2) + 1, col=(i % 2) + 1)
    fig.update_layout(height=600, width=800, title_text="Problem Types Breakdown")
    fig.show()

if __name__ == "__main__":
    main()