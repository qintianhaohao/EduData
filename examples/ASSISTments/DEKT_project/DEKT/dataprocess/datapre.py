
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold
import csv


def load_data():
    """加载数据并进行预处理"""
    data = pd.read_csv(
        '../../data/anonymized_full_release_competition_dataset/test.csv',
        usecols=['startTime', 'timeTaken', 'studentId', 'skill', 'problemId', 'problemType', 'correct', 'BORED', 'CONCENTRATING', 'CONFUSED', 'FRUSTRATED']
    ).dropna(subset=['skill', 'problemId']).sort_values('startTime')
    data.timeTaken = data.timeTaken.astype(int)
    return data




def extract_unique_values(data):
    """提取数据中的唯一值"""
    stuids = data.studentId.unique().tolist()
    skills = data.skill.unique().tolist()
    problems = data.problemId.unique().tolist()
    at = data.timeTaken.unique()
    types = data.problemType.unique().tolist()
    return stuids, skills, problems, at, types





# question id from 1 to #num_skill

def create_id_mappings(stuids, skills, problems, at, types):
    """创建ID映射字典"""
    stu2id = {s: i for i, s in enumerate(stuids)}
    skill2id = {p: i for i, p in enumerate(skills)}
    problem2id = {p: i for i, p in enumerate(problems)}
    at2id = {a: i for i, a in enumerate(at)}
    type2id = {t: i for i, t in enumerate(types)}
    return stu2id, skill2id, problem2id, at2id, type2id


def calculate_interval_time(data):
    """计算时间间隔并创建映射"""
    it = set()
    for u in data.studentId.unique():
        startTime = np.array(data[data.studentId == u].startTime)
        for i in range(1, len(startTime)):
            item = (startTime[i] - startTime[i - 1]) // 60
            if item > 43200:
                item = 43200
            it.add(item)
    it2id = {a: i for i, a in enumerate(it)}
    return it2id


def calculate_pre_and_attempt_time(data):
    """计算前置时间和尝试次数并创建映射"""
    pretime = set()
    attmptime = set()
    for u in data.studentId.unique():
        dict_sq = {}
        idx = data[(data.studentId == u)].index.tolist()
        temp = data.iloc[idx]
        for xxx in np.array(temp):
            if xxx[3] not in dict_sq:
                dict_sq[xxx[3]] = [xxx[0], 1]
                pretime.add(43200)
                attmptime.add(0)
            else:
                tt = (xxx[0] - dict_sq[xxx[3]][0]) // 60
                if tt > 43200:
                    tt = 43200
                pretime.add(tt)
                attmptime.add(dict_sq[xxx[3]][1])
                dict_sq[xxx[3]][0] = xxx[0]
                dict_sq[xxx[3]][1] += 1
    pre2id = {p: i for i, p in enumerate(pretime)}
    attemp2id = {a: i for i, a in enumerate(attmptime)}
    return pre2id, attemp2id





def print_statistics(stuids, skills, problems, at, it, types, pretime, attmptime):
    """打印统计数据"""
    print("number of stus: %d" % len(stuids))
    print("number of skills: %d" % len(skills))
    print("number of problems: %d" % len(problems))
    print("number of answer time: %d" % len(at))
    print("number of interval time: %d" % len(it))
    print("number of problemType: %d" % len(types))
    print("number of pretime : %d" % max(pretime))
    print("number of attmptime: %d" % max(attmptime))





def map_problem_to_skill(data, problem2id, skill2id):
    """将问题映射到技能"""
    problem2skill = {}
    for s, p in zip(np.array(data.skill), np.array(data.problemId)):
        problem2skill[problem2id[p]] = skill2id[s]
    with open('../../data/anonymized_full_release_competition_dataset/problem2skill', 'w', encoding='utf-8') as f:
        f.write(str(problem2skill))



############---------------------------------------############


def calculate_skill_difficulty(data, skills):
    """计算技能难度"""
    sdifficult2skill = {}
    nonesk = []  # dropped with less than 30 answer records
    for i in tqdm(skills):
        tttt = []
        idx = data[(data.skill == i)].index.tolist()
        temp1 = data.iloc[idx]
        if len(idx) < 5:
            sdifficult2skill[i] = 10
            continue
        for xxx in np.array(temp1):
            tttt.append(xxx[6])
        avg = int(np.mean(tttt) * 100) + 1
        sdifficult2skill[i] = avg
    print("技能难度的最大值：", max(sdifficult2skill.values()))
    print("技能难度的最小值：", min(sdifficult2skill.values()))
    return sdifficult2skill

# Question difficulty

def calculate_question_difficulty(data, problems):
    """计算问题难度"""
    qdifficult2problem = {}
    nones = []
    for i in tqdm(problems):
        tttt = []
        idx = data[(data.problemId == i)].index.tolist()
        temp1 = data.iloc[idx]
        if len(idx) < 10:
            qdifficult2problem[i] = 5
            continue
        for xxx in np.array(temp1):
            tttt.append(xxx[6])
        avg = int(np.mean(tttt) * 100) + 1
        qdifficult2problem[i] = avg
    print("问题难度的最大值：", max(qdifficult2problem.values()))
    print("问题难度的最小值：", min(qdifficult2problem.values()))
    return qdifficult2problem

############---------------------------------------############




############---------------------------------------############


def parse_all_sequences(data, students, stu2id, skill2id, problem2id, it2id, at2id, qdifficult2problem, sdifficult2skill, type2id):
    """解析所有学生序列"""
    all_sequences = []
    for student_id in tqdm(students, 'parse student sequence:\t'):
        student_sequence = parse_student_sequence(data[data.studentId == student_id], stu2id, skill2id, problem2id, it2id, at2id, qdifficult2problem, sdifficult2skill, type2id)
        all_sequences.extend([student_sequence])
    return all_sequences



def parse_student_sequence(student, stu2id, skill2id, problem2id, it2id, at2id, qdifficult2problem, sdifficult2skill, type2id):
    """解析单个学生序列"""
    seq = student
    stu = [stu2id[st] for st in seq.studentId.tolist()]
    s = [skill2id[q] for q in seq.skill.tolist()]
    a = seq.correct.tolist()
    p = [problem2id[p] for p in seq.problemId.tolist()]
    it = [0]
    startTime = np.array(seq.startTime)
    for i in range(1, len(startTime)):
        item = (startTime[i] - startTime[i - 1]) // 60
        if item > 43200:
            item = 43200
        it.append(it2id[item])
    at = [at2id[int(x)] for x in seq.timeTaken.tolist()]
    bor = [bor for bor in seq.BORED.tolist()]
    conc = [conc for conc in seq.CONCENTRATING.tolist()]
    conf = [conf for conf in seq.CONFUSED.tolist()]
    fru = [fru for fru in seq.FRUSTRATED.tolist()]
    qd = [qdifficult2problem[p] for p in seq.problemId.tolist()]
    sd = [sdifficult2skill[s] for s in seq.skill.tolist()]
    ty = [type2id[y] for y in seq.problemType.tolist()]
    dict_sq = {}
    pretime = []
    attmptime = []
    for xxx in np.array(seq):
        if xxx[3] not in dict_sq:
            dict_sq[xxx[3]] = [xxx[0], 1]
            pretime.append(43200)
            attmptime.append(0)
        else:
            tt = (xxx[0] - dict_sq[xxx[3]][0]) // 60
            if tt > 43200:
                tt = 43200
            pretime.append(tt)
            attmptime.append(dict_sq[xxx[3]][1])
            dict_sq[xxx[3]][0] = xxx[0]
            dict_sq[xxx[3]][1] += 1
    return s, a, p, it, at, bor, conc, conf, fru, qd, sd, ty, stu, pretime, attmptime


def write_sequences_to_file(sequences, trg_path):
    """将序列写入文件"""
    with open(trg_path, 'a', encoding='utf8') as f:
        for seq in tqdm(sequences, 'write data into file: %s' % trg_path):
            s_seq, a_seq, p_seq, it_seq, at_seq, bor_seq, conc_seq, conf_seq, fru_seq, qd_seq, sd_seq, ty_seq, stu_seq, pre_seq, att_seq = seq
            seq_len = len(s_seq)
            f.write(str(seq_len) + '\n')
            f.write(','.join([str(s) for s in s_seq]) + '\n')
            f.write(','.join([str(a) for a in a_seq]) + '\n')
            f.write(','.join([str(p) for p in p_seq]) + '\n')
            f.write(','.join([str(i) for i in it_seq]) + '\n')
            f.write(','.join([str(a) for a in at_seq]) + '\n')
            f.write(','.join([str(bor) for bor in bor_seq]) + '\n')
            f.write(','.join([str(conc) for conc in conc_seq]) + '\n')
            f.write(','.join([str(conf) for conf in conf_seq]) + '\n')
            f.write(','.join([str(fru) for fru in fru_seq]) + '\n')
            f.write(','.join([str(q) for q in qd_seq]) + '\n')
            f.write(','.join([str(s) for s in sd_seq]) + '\n')
            f.write(','.join([str(y) for y in ty_seq]) + '\n')
            f.write(','.join([str(stu) for stu in stu_seq]) + '\n')
            f.write(','.join([str(pre) for pre in pre_seq]) + '\n')
            f.write(','.join([str(att) for att in att_seq]) + '\n')


def main():
    """主函数，执行程序的主要逻辑"""
    data = load_data()
    stuids, skills, problems, at, types = extract_unique_values(data)
    stu2id, skill2id, problem2id, at2id, type2id = create_id_mappings(stuids, skills, problems, at, types)
    it2id = calculate_interval_time(data)
    pre2id, attemp2id = calculate_pre_and_attempt_time(data)
    print_statistics(stuids, skills, problems, at, it2id.keys(), types, pre2id.keys(), attemp2id.keys())
    map_problem_to_skill(data, problem2id, skill2id)
    sdifficult2skill = calculate_skill_difficulty(data, skills)
    qdifficult2problem = calculate_question_difficulty(data, problems)
    sequences = parse_all_sequences(data, data.studentId.unique(), stu2id, skill2id, problem2id, it2id, at2id, qdifficult2problem, sdifficult2skill, type2id)
    train_data, test_data = train_test_split(sequences, test_size=.2, random_state=10)
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    kfold = KFold(n_splits=5, shuffle=True, random_state=10)
    idx = 0
    for train_data_1, valid_data in kfold.split(train_data):
        write_sequences_to_file(train_data[train_data_1], '../../data/anonymized_full_release_competition_dataset/train' + str(idx) + '.txt')
        write_sequences_to_file(train_data[valid_data], '../../data/anonymized_full_release_competition_dataset/valid' + str(idx) + '.txt')
        idx += 1
    write_sequences_to_file(test_data, '../../data/anonymized_full_release_competition_dataset/test.txt')
    print('complete')


if __name__ == "__main__":
    main()
  