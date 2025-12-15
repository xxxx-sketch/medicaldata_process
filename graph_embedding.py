# graph_embedding.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from node2vec import Node2Vec

class GraphEmbeddingGenerator:
    """基于图结构生成节点嵌入"""

    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        self.node_embeddings = None
        self.model = None

    def generate_node2vec_embeddings(self, graph, walk_length=30, num_walks=200, workers=4):
        """使用Node2Vec生成节点嵌入"""
        print("\n🎯 使用Node2Vec生成节点嵌入...")

        # 创建Node2Vec模型
        node2vec = Node2Vec(graph, dimensions=self.embedding_dim, walk_length=walk_length,
                            num_walks=num_walks, workers=workers)

        # 训练模型
        self.model = node2vec.fit(window=10, min_count=1, batch_words=4)

        # 获取所有节点的嵌入
        self.node_embeddings = {}
        for node in graph.nodes():
            self.node_embeddings[node] = self.model.wv[node]

        print(f"✅ Node2Vec嵌入完成! 嵌入维度: {self.embedding_dim}")
        return self.node_embeddings

    def generate_graphsage_embeddings(self, graph, num_layers=2, hidden_dim=256):
        """使用GraphSAGE风格的简单图神经网络生成嵌入"""
        print("\n🎯 使用GraphSAGE风格生成节点嵌入...")

        # 构建邻接矩阵
        nodes = list(graph.nodes())
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        num_nodes = len(nodes)

        # 创建初始节点特征（基于节点类型和度）
        initial_features = []
        for node in nodes:
            if graph.nodes[node].get('type') == 'disease':
                # 疾病节点: 使用度作为特征
                feature = [graph.degree(node)] + [0] * 9  # 疾病节点特征
            else:
                # 患者节点: 使用连接疾病数量作为特征
                disease_neighbors = [n for n in graph.neighbors(node)
                                     if graph.nodes[n].get('type') == 'disease']
                feature = [0] + [len(disease_neighbors)] + [0] * 8  # 患者节点特征
            initial_features.append(feature)

        initial_features = torch.FloatTensor(initial_features)

        # 构建邻接矩阵
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if graph.has_edge(node_i, node_j):
                    adj_matrix[i, j] = 1

        # 简单的图卷积层
        class SimpleGraphSAGE(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
                super().__init__()
                self.layers = nn.ModuleList()
                self.layers.append(nn.Linear(input_dim, hidden_dim))

                for _ in range(num_layers - 2):
                    self.layers.append(nn.Linear(hidden_dim, hidden_dim))

                self.layers.append(nn.Linear(hidden_dim, output_dim))
                self.dropout = nn.Dropout(0.1)

            def forward(self, x, adj):
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    if i < len(self.layers) - 1:
                        x = F.relu(x)
                        x = self.dropout(x)
                    # 图传播
                    x = torch.matmul(adj, x)
                return x

        # 训练模型
        model = SimpleGraphSAGE(initial_features.shape[1], hidden_dim, self.embedding_dim, num_layers)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        model.train()
        for epoch in range(100):
            optimizer.zero_grad()
            embeddings = model(initial_features, adj_matrix)

            # 简单的重建损失
            reconstructed_adj = torch.sigmoid(torch.matmul(embeddings, embeddings.t()))
            loss = F.binary_cross_entropy(reconstructed_adj, adj_matrix)

            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                print(f"   Epoch {epoch}, Loss: {loss.item():.6f}")

        # 获取最终嵌入
        model.eval()
        with torch.no_grad():
            final_embeddings = model(initial_features, adj_matrix)
            self.node_embeddings = {node: final_embeddings[i].numpy()
                                    for i, node in enumerate(nodes)}

        print(f"✅ GraphSAGE嵌入完成! 嵌入维度: {self.embedding_dim}")
        return self.node_embeddings

    def get_patient_embeddings(self, graph):
        """提取患者节点的嵌入"""
        patient_embeddings = {}
        for node, embedding in self.node_embeddings.items():
            if graph.nodes[node].get('type') == 'patient':
                patient_embeddings[node] = embedding

        print(f"📊 提取了 {len(patient_embeddings)} 个患者嵌入")
        return patient_embeddings

    def visualize_embeddings(self, patient_embeddings, labels=None, filename="patient_embeddings.png"):
        """可视化患者嵌入"""
        try:
            # 准备数据
            patient_ids = list(patient_embeddings.keys())
            embeddings = np.array(list(patient_embeddings.values()))

            # 使用t-SNE降维
            tsne = TSNE(n_components=2, random_state=42)
            embeddings_2d = tsne.fit_transform(embeddings)

            plt.figure(figsize=(12, 8))

            if labels is not None:
                # 如果有标签，根据标签着色
                scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                      c=labels, cmap='tab10', alpha=0.7, s=50)
                plt.colorbar(scatter, label='Cluster')
            else:
                plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, s=50)

            plt.title("Patient Embeddings Visualization (t-SNE)")
            plt.xlabel("t-SNE Component 1")
            plt.ylabel("t-SNE Component 2")
            plt.grid(True, alpha=0.3)

            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.show()

            print(f"📊 患者嵌入可视化已保存: {filename}")

        except Exception as e:
            print(f"❌ 嵌入可视化失败: {e}")