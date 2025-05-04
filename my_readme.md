
# 命令


## 启动jupyterlab
```bash
jupyter lab
```

## 将.ipynb文件转换成脚本文件
1. 命令行
```bash
jupyter nbconvert --to script your_notebook.ipynb
```

2. 使用大模型进行代码标准化 
```
prompt：把这个文件转换成标准python文件格式，添加main函数，把所有代码块都转换成函数，并对所有函数添加中文注释
```

## 启动conda环境
```commandline
conda activate myd
```

# 可用代码
## AKT
```commandline
python D:\code\myd_homework\EduKTM\examples\AKT\prepare_dataset.py
python D:\code\myd_homework\EduKTM\examples\AKT\AKT.py
```

## LPKT
```commandline
python D:\code\myd_homework\EduKTM\examples\LPKT\prepare_dataset.py
python D:\code\myd_homework\EduKTM\examples\LPKT\LPKT.py
```

## DEKT
```commandline
python D:\code\myd_homework\EduData\examples\ASSISTments\DEKT-project\DEKT-main\DEKT\main.py
```


# 论文
## 使用junyi数据集的论文
### DKT方法
- Liu, Q., Tong, S., Liu, C., Zhao, H., Chen, E., Ma, H., Wang, S. Exploiting cognitive structure for adaptive learning. arXiv preprint arXiv:1905.12470 (2019)
中文名：《利用认知结构进行自适应学习》
下载链接：https://arxiv.org/abs/1905.12470

## 使用ASSISTments数据集的论文
### DEKT方法
- Dual-State Personalized Knowledge Tracing with Emotional Incorporation
论文：https://arxiv.org/abs/2405.16799
源码：https://github.com/yfz-cloud/DEKT-project/tree/master

## 使用EdNet数据集的论文
### DKT方法
- https://github.com/arshadshk/SAINT-pytorch?tab=readme-ov-file
- https://github.com/arshadshk/SAKT-pytorch/tree/main


# 参考
- edudata的安装教程：
<https://edudata.readthedocs.io/en/latest/tutorial/zh/graph.html>

- 中科大edudata的下载链接：
<http://base.ustc.edu.cn/data>

- 中科大BASE组主页
<https://base.ustc.edu.cn/>

- 关于知识追踪的开源数据集：
<https://github.com/bigdata-ustc/EduData>

- 关于知识追踪的开源模型及代码：
<https://github.com/bigdata-ustc/EduKTM?tab=readme-ov-file>