# data_loader.py
import pandas as pd
import torch
import os
import numpy as np
from disease_graph import DiseaseGraphBuilder
from graph_embedding import GraphEmbeddingGenerator

#用的disease-entity数据
class DataLoader:
    DEFAULT_DATA_PATH = "/Users/Kate/Documents/DATA/disease_entity_fortrain.csv"

    @staticmethod
    def load_data(csv_file_path=None, embedding_method='graphsage', embedding_dim=64):

        csv_file_path = csv_file_path or DataLoader.DEFAULT_DATA_PATH
        df = pd.read_csv(csv_file_path, encoding='utf-8')

        # 构建疾病图
        graph_builder = DiseaseGraphBuilder(window_size=2, min_cooccurrence=2)
        cooccurrence_edges, patient_disease_map = graph_builder.extract_cooccurrence_relations(df)

        # 构建图
        graph = graph_builder.build_disease_graph(cooccurrence_edges)
        graph = graph_builder.add_patient_nodes(patient_disease_map)

        # 生成嵌入
        embedding_generator = GraphEmbeddingGenerator(embedding_dim=embedding_dim)
        if embedding_method == 'node2vec':
            node_embeddings = embedding_generator.generate_node2vec_embeddings(graph, workers=1)
        else:
            node_embeddings = embedding_generator.generate_graphsage_embeddings(graph)

        # 提取患者嵌入
        patient_embeddings = embedding_generator.get_patient_embeddings(graph)
        patient_ids = list(patient_embeddings.keys())
        x = torch.FloatTensor([patient_embeddings[pid] for pid in patient_ids])
        y = DataLoader._generate_smart_labels(patient_disease_map, patient_ids)
        edge_index = DataLoader._build_similarity_graph(patient_disease_map, graph_builder.patient_to_idx)

        return x, y, edge_index, patient_disease_map

    @staticmethod
    def _generate_smart_labels(patient_disease_map, patient_ids):
        """按照每个patient患病的数量手动计算标签"""
        labels = []
        for patient_id in patient_ids:
            disease_count = len(patient_disease_map[patient_id])
            # 根据疾病数量分类
            if disease_count <= 2:
                labels.append(0)  # 简单病例
            elif disease_count <= 4:
                labels.append(1)  # 中等复杂度
            elif disease_count <= 6:
                labels.append(2)  # 复杂病例
            else:
                labels.append(3)  # 高度复杂
        return torch.tensor(labels, dtype=torch.long)

    @staticmethod
    def _build_similarity_graph(patient_disease_map, patient_to_idx):
        """基于疾病相似度构建edge"""

        edges = []
        patient_list = list(patient_disease_map.keys())
        n_patients = len(patient_list)

        if n_patients == 0:
            return torch.tensor([], dtype=torch.long).reshape(2, 0)

        # 计算患者间的疾病相似度
        similarity_matrix = DataLoader._calculate_disease_similarity(patient_disease_map, patient_list)

        # 为每个患者选择前k个最相似的患者创建边
        k = min(5, n_patients - 1)  # 每个节点连接k个最相似的邻居

        for i in range(n_patients):
            similarities = []
            for j in range(n_patients):
                if i != j:
                    similarities.append((j, similarity_matrix[i, j]))

            similarities.sort(key=lambda x: x[1], reverse=True)

            for neighbor_idx, similarity in similarities[:k]:
                if similarity > 0:  # 只添加有相似度的边
                    idx1 = patient_to_idx[patient_list[i]]
                    idx2 = patient_to_idx[patient_list[neighbor_idx]]
                    if [idx1, idx2] not in edges and [idx2, idx1] not in edges:
                        edges.append([idx1, idx2])
                        edges.append([idx2, idx1])

        # 如果边太少，添加最小生成树确保连通性
        if len(edges) < (n_patients - 1) * 2:
            edges = DataLoader._ensure_connectivity(edges, patient_list, patient_to_idx, n_patients)

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        print(f" 基于相似度的图边数量: {edge_index.shape[1] // 2} 条无向边")

        return edge_index

    @staticmethod
    def _calculate_disease_similarity(patient_disease_map, patient_list):
        """计算患者间的疾病相似度矩阵"""
        import numpy as np

        n_patients = len(patient_list)
        similarity_matrix = np.zeros((n_patients, n_patients))

        # 预处理患者的疾病集合
        patient_disease_sets = []
        for patient_id in patient_list:
            patient_disease_sets.append(set(patient_disease_map[patient_id]))

        # 计算Jaccard相似度
        for i in range(n_patients):
            for j in range(i + 1, n_patients):
                set1 = patient_disease_sets[i]
                set2 = patient_disease_sets[j]

                if len(set1) == 0 and len(set2) == 0:
                    similarity = 0.0
                else:
                    intersection = len(set1 & set2)
                    union = len(set1 | set2)
                    similarity = intersection / union if union > 0 else 0.0

                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

        return similarity_matrix

    @staticmethod
    def _ensure_connectivity(edges, patient_list, patient_to_idx, n_patients):
        # 使用简单的链式连接确保连通性
        for i in range(n_patients - 1):
            idx1 = patient_to_idx[patient_list[i]]
            idx2 = patient_to_idx[patient_list[i + 1]]

            # 检查边是否已存在
            if [idx1, idx2] not in edges and [idx2, idx1] not in edges:
                edges.append([idx1, idx2])
                edges.append([idx2, idx1])

        return edges

    @staticmethod
    def get_data_info(csv_file_path=None):
        if csv_file_path is None:
            csv_file_path = DataLoader.DEFAULT_DATA_PATH

        try:
            # 检测编码
            encoding = DataLoader.detect_encoding(csv_file_path)
            if encoding is None:
                encoding = 'utf-8'

            df = pd.read_csv(csv_file_path, encoding=encoding)
            info = {
                'num_samples': len(df),
                'columns': df.columns.tolist(),
                'data_types': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict()
            }
            return info
        except Exception as e:
            print(f" 获取数据信息失败: {e}")
            return None

#之前写了数据预览，删了