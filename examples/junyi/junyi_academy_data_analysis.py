# # Junyi
# [data source](https://pslcdatashop.web.cmu.edu/Files?datasetId=1198)
#
# ### Authorization
# 任何商业用途都是不允许的！
# 如果您发表您的工作，请引用以下论文：
#
# Haw-Shiuan Chang, Hwai-Jung Hsu and Kuan-Ta Chen,
# "Modeling Exercise Relationships in E-Learning: A Unified Approach,"
# International Conference on Educational Data Mining (EDM), 2015.
#
# ### Introduction
# 该数据集包含了Junyi Academy（http://www.junyiacademy.org/）的问题日志和练习相关信息。此外，我们收集的用于构建模型的练习关系注释也包含在内。
#
# ## 数据描述
# ### 列描述
# #### junyi_Exercise_table.csv:
# | 字段 | 注释 |
# |------|------|
# | name | 练习名称（名称也是练习的ID，因此每个名称在数据集中是唯一的）。如果你想访问网站上的练习，请在此名称后附加URL，例如 http://www.junyiacademy.org/exercise/similar_triangles_1。请注意，Junyi Academy 像Khan Academy一样不断更改其内容，因此某些练习的URL可能在你访问时不可用。 |
# | live | 该练习是否在2015年1月仍然可以在网站上访问 |
# | prerequisite | 指示其先修练习（知识图谱中的父节点） |
# | h_position | 知识图谱中x轴的坐标 |
# | v_position | 知识图谱中y轴的坐标 |
# | creation_date | 该练习创建的日期 |
# | seconds_per_fast_problem | 如果学生回答问题的时间少于这个时间，则网站会判断学生完成练习很快。这个数字是由Junyi Academy的专家手动分配的。 |
# | pretty_display_name | 在知识图谱中显示的练习中文名称（请使用UTF-8解码中文字符） |
# | short_display_name | 练习的另一个中文名称（请使用UTF-8解码中文字符） |
# | topic | 每个练习的主题，在知识图谱中会显示为较大的节点 |
# | area | 每个练习的领域（每个领域包含多个主题） |

import numpy as np
import dask.dataframe as dd
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go


def read_and_process_data(path, encoding="utf-8", low_memory=False):
    data = pd.read_csv(path, encoding=encoding, low_memory=low_memory)
    return data


def plot_exercises_distribution(data, color_column, title):
    fig = px.scatter(
        data,
        x='h_position',
        y='v_position',
        color=color_column,
        title=title
    )
    fig.show()


def makeplot(title, groupByItem, data):
    ds = data.groupby(groupByItem, as_index=False).agg(exercise_count=('topic', 'count'))
    ds = ds.sort_values('exercise_count')

    fig = px.bar(
        ds,
        x='exercise_count',
        y=groupByItem,
        orientation='h',
        title=title
    )
    fig.show()


def main():
    # 读取并处理数据
    path = "../../dataset/junyi/junyi_Exercise_table.csv"
    data = read_and_process_data(path)

    # 显示数据前几行
    print(data.head())

    # 描述性统计
    print(data.describe())

    # 处理area列
    data["area"] = [item if item != "null" and item != 'nan' else "unknown" for item in data["area"].apply(str)]

    # 绘制练习分布图
    plot_exercises_distribution(data, 'area', 'Exercises distribution on area in knowledge map')

    # 处理topic列
    data["topic"] = [item if item != "null" and item != 'nan' else "unknown" for item in data["topic"].apply(str)]

    # 绘制练习分布图
    plot_exercises_distribution(data, 'topic', 'Exercises distribution on topics in knowledge map')

    # 绘制条形图
    makeplot(title='Exercise count on area', groupByItem='area', data=data)
    makeplot(title='Exercise count on topics', groupByItem='topic', data=data)

    # 其他数据集的处理
    path = "../../dataset/junyi/relationship_annotation_training.csv"
    data = dd.read_csv(path, encoding="utf-8", low_memory=False)
    print(data.head())
    print(data.describe().compute())

    # junyi_ProblemLog_original.csv
    path = "../../dataset/junyi/junyi_ProblemLog_original.csv"
    data = dd.read_csv(path, encoding="utf-8", low_memory=False, dtype={'hint_time_taken_list': 'object'})
    print(data.head())
    print(data.describe().compute())
    print(data['user_id'].nunique().compute())
    total_count = len(data)
    print(total_count)

    # 处理earned_proficiency列
    ds = data['earned_proficiency'].value_counts().reset_index().compute()
    ds.columns = ['earned_proficiency', 'percent']
    ds['percent'] /= total_count
    ds = ds.sort_values(['percent'])
    fig = px.pie(ds, names=['mastered', 'not mastered'], values='percent', title='Percent of mastered exercises')
    fig.show()

    # 处理correct列
    ds = data['correct'].value_counts().reset_index().compute()
    ds.columns = ['correct', 'percent']
    ds['percent'] /= total_count
    ds = ds.sort_values(['percent'])
    fig = px.pie(ds, names=['wrong', 'correct'], values='percent', title='Percent of answer correctly at first attempt')
    fig.show()

    # junyi_ProblemLog_for_PSLC.txt
    path = "../../dataset/junyi/junyi_ProblemLog_for_PSLC.txt"
    data = dd.read_csv(path, sep='\t', encoding="utf-8")
    pd.set_option('display.max_columns', 2000)
    print(data.head())

    # 分析
    print(len(data))

    # 每个用户的session数
    ds = data.groupby('Anon Student Id').agg({'Session Id': 'count'}).describe().compute()
    print(ds)

    # 1%抽样
    data1 = data.sample(frac=0.01).compute()

    # 每个session对应的练习次数、知识点数(1%抽样)
    nunique = dd.Aggregation(
        name="nunique",
        chunk=lambda s: s.apply(lambda x: list(set(x))),
        agg=lambda s0: s0.obj.groupby(level=list(range(s0.obj.index.nlevels))).sum(),
        finalize=lambda s1: s1.apply(lambda final: len(set(final))),
    )
    ds = data1.groupby('Session Id').agg(
        {'KC (Exercise)': 'nunique', 'KC (Topic)': 'nunique', 'Time': lambda x: x.max() - x.min()})
    print(ds.describe())


if __name__ == "__main__":
    main()
