#this is a guidance for medical data analysis based on adaptive GCN model

原始数据来源于disease-entity.xls 疾病数据库
模型训练数据集为disease_entity_fortrain
index 为 hospitalization_id. demographic 保留 gender and education. 
for gender variable 0-male, 1-female
for education variable 0-illiterate 1-elementary 2-secondary
3-technical secondary 4- high school 5-junior college 6-bachelor

一下代码用于修复冲突
#================ 修复OpenMP冲突 ================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '1'

#================ 修复matplotlib后端 ================
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，不显示窗口，只保存图片
import matplotlib.pyplot as plt

#================ 其他导入 ================
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')