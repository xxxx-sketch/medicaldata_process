# disease_graph.py
import pandas as pd
import numpy as np
import torch
import networkx as nx
from collections import Counter, defaultdict
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.path import Path
import matplotlib.patches as patches
from node2vec import Node2Vec


class DiseaseGraphBuilder:
    """基于疾病共现关系构建图结构"""

    def __init__(self, window_size=2, min_cooccurrence=2):
        self.window_size = window_size
        self.min_cooccurrence = min_cooccurrence
        self.graph = None
        self.disease_to_idx = {}
        self.idx_to_disease = {}
        self.patient_to_idx = {}
        self.idx_to_patient = {}

    def preprocess_disease_names(self, disease_names):
        """预处理疾病名称 - 修复版本"""
        print("🔧 预处理疾病名称...")

        def clean_disease_name(name):
            if pd.isna(name) or name == '' or name == 'nan':
                return None

            # 转换为字符串并去除特殊字符
            name = str(name).strip()

            # 去除特殊字符和多余空格
            name = re.sub(r'[^\w\u4e00-\u9fff]', ' ', name)
            name = re.sub(r'\s+', ' ', name).strip()

            return name if name else None

        # 批量处理所有疾病名称
        cleaned_names = []
        skipped_count = 0

        for name in disease_names:
            cleaned = clean_disease_name(name)
            if cleaned:
                cleaned_names.append(cleaned)
            else:
                skipped_count += 1

        print(f"📊 疾病名称预处理完成:")
        print(f"   - 原始疾病数量: {len(disease_names)}")
        print(f"   - 清洗后疾病数量: {len(cleaned_names)}")
        print(f"   - 跳过无效疾病: {skipped_count}")

        return cleaned_names

    def build_disease_vocabulary(self, all_diseases):
        """构建疾病词汇表"""
        print("\n📚 构建疾病词汇表...")

        disease_counts = Counter(all_diseases)

        # 过滤低频疾病
        frequent_diseases = {disease for disease, count in disease_counts.items()
                             if count >= self.min_cooccurrence}

        # 创建疾病索引映射
        self.disease_to_idx = {disease: idx for idx, disease in enumerate(frequent_diseases)}
        self.idx_to_disease = {idx: disease for disease, idx in self.disease_to_idx.items()}

        print(f"🏷️  疾病词汇表大小: {len(self.disease_to_idx)}")
        print("🔝 最常见的10种疾病:")
        for disease, count in disease_counts.most_common(10):
            print(f"   {disease}: {count} 次")

        return self.disease_to_idx

    def extract_cooccurrence_relations(self, df):
        """从数据中提取疾病共现关系 - 修复版本"""
        print("\n🔗 提取疾病共现关系...")

        disease_columns = [f'Disease{i}' for i in range(1, 11)]
        cooccurrence_edges = []
        patient_disease_map = {}

        # 收集所有疾病名称
        all_diseases = []
        for col in disease_columns:
            # 处理缺失值并转换为字符串
            diseases = df[col].fillna('').astype(str)
            # 过滤空字符串
            diseases = diseases[diseases != ''].unique()
            all_diseases.extend(diseases)

        print(f"📊 从数据中提取到 {len(all_diseases)} 个原始疾病记录")

        # 预处理疾病名称 - 只调用一次
        cleaned_diseases = self.preprocess_disease_names(all_diseases)

        # 构建疾病词汇表
        self.build_disease_vocabulary(cleaned_diseases)

        # 创建疾病名称到清洗后名称的映射
        disease_mapping = {}
        for orig, cleaned in zip(all_diseases, cleaned_diseases):
            disease_mapping[orig] = cleaned

        # 处理每个患者的疾病
        valid_patients = 0
        for _, row in df.iterrows():
            patient_id = row['Hospitalization_id']
            patient_diseases = set()

            for col in disease_columns:
                disease = row[col]
                # 检查是否为有效疾病
                if pd.notna(disease) and disease != '' and str(disease) != 'nan':
                    orig_disease = str(disease)
                    if orig_disease in disease_mapping:
                        cleaned_disease = disease_mapping[orig_disease]
                        if cleaned_disease in self.disease_to_idx:
                            patient_diseases.add(cleaned_disease)

            # 只有至少有两种疾病的患者才考虑
            if len(patient_diseases) >= 2:
                patient_disease_map[patient_id] = list(patient_diseases)
                valid_patients += 1

                # 为同一患者内的所有疾病对创建边
                disease_list = list(patient_diseases)
                for i in range(len(disease_list)):
                    for j in range(i + 1, len(disease_list)):
                        cooccurrence_edges.append((disease_list[i], disease_list[j]))

        print(f"📈 发现 {len(cooccurrence_edges)} 个疾病共现关系")
        print(f"👥 涉及 {valid_patients} 个有效患者（至少有两种疾病）")

        return cooccurrence_edges, patient_disease_map

    def build_disease_graph(self, cooccurrence_edges):
        """构建疾病图"""
        print("\n🕸️  构建疾病图...")

        self.graph = nx.Graph()

        # 添加疾病节点
        for disease in self.disease_to_idx.keys():
            self.graph.add_node(disease, type='disease')

        # 添加共现边
        edge_weights = Counter(cooccurrence_edges)
        for (disease1, disease2), weight in edge_weights.items():
            self.graph.add_edge(disease1, disease2, weight=weight, type='cooccurrence')

        # 图统计信息
        print(f"📊 疾病图统计:")
        print(f"   疾病节点数: {self.graph.number_of_nodes()}")
        print(f"   疾病共现边数: {self.graph.number_of_edges()}")
        if self.graph.number_of_nodes() > 0:
            avg_degree = np.mean([d for n, d in self.graph.degree()])
            print(f"   平均度: {avg_degree:.2f}")

        return self.graph

    def add_patient_nodes(self, patient_disease_map):
        """添加患者节点和患者-疾病边"""
        print("\n👥 添加患者节点...")

        # 创建患者索引映射
        patient_ids = list(patient_disease_map.keys())
        self.patient_to_idx = {patient: idx for idx, patient in enumerate(patient_ids)}
        self.idx_to_patient = {idx: patient for patient, idx in self.patient_to_idx.items()}

        # 添加患者节点
        for patient_id in patient_ids:
            self.graph.add_node(patient_id, type='patient')

        # 添加患者-疾病边
        has_disease_edges = 0
        for patient_id, diseases in patient_disease_map.items():
            for disease in diseases:
                self.graph.add_edge(patient_id, disease, weight=1, type='has_disease')
                has_disease_edges += 1

        print(f"📊 完整图统计:")
        print(f"   总节点数: {self.graph.number_of_nodes()}")
        print(f"   总边数: {self.graph.number_of_edges()}")
        print(f"   患者-疾病边数: {has_disease_edges}")

        return self.graph

    def _bezier_curve(self, p0, p1, curvature=0.3):
        """生成二次贝塞尔曲线点"""
        # 计算中点
        mid_point = (p0 + p1) / 2

        # 计算垂直方向
        direction = p1 - p0
        perpendicular = np.array([-direction[1], direction[0]])
        perpendicular = perpendicular / np.linalg.norm(perpendicular)

        # 控制点（在中点基础上添加垂直偏移）
        control_point = mid_point + perpendicular * curvature * np.linalg.norm(direction)

        # 生成贝塞尔曲线点
        t = np.linspace(0, 1, 50)
        curve_points = np.outer((1 - t) ** 2, p0) + np.outer(2 * (1 - t) * t, control_point) + np.outer(t ** 2, p1)

        return curve_points

    def _draw_bezier_edge(self, ax, pos, u, v, edge_data, max_weight, color='gray', base_width=0.5):
        """绘制贝塞尔曲线边"""
        p0 = np.array(pos[u])
        p1 = np.array(pos[v])

        # 根据权重计算曲线参数
        weight = edge_data.get('weight', 1)
        normalized_weight = weight / max_weight if max_weight > 0 else 0

        # 曲线弯曲程度（权重越大，曲线越平缓）
        curvature = 0.5 * (1 - normalized_weight * 0.8)

        # 线条宽度和透明度基于权重
        line_width = base_width + normalized_weight * 3
        alpha = 0.2 + normalized_weight * 0.6

        # 生成并绘制贝塞尔曲线
        curve_points = self._bezier_curve(p0, p1, curvature)
        ax.plot(curve_points[:, 0], curve_points[:, 1],
                color=color, linewidth=line_width, alpha=alpha,
                solid_capstyle='round')

    def visualize_graph(self, filename="disease_cooccurrence_graph.png"):
        """优化版图可视化 - 使用贝塞尔曲线"""
        try:
            # 设置matplotlib后端和样式
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            # 创建子图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

            # ===== 左侧：疾病共现网络 =====
            print("📊 绘制疾病共现网络（使用贝塞尔曲线）...")

            # 提取疾病子图（只包含疾病节点和疾病-疾病边）
            disease_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == 'disease']
            disease_edges = [(u, v, self.graph[u][v]) for u, v in self.graph.edges()
                             if self.graph[u][v].get('type') == 'cooccurrence']

            disease_subgraph = self.graph.subgraph(disease_nodes)

            if len(disease_nodes) > 0:
                # 使用spring layout，但调整参数以获得更好布局
                pos = nx.spring_layout(disease_subgraph, k=3 / np.sqrt(len(disease_nodes)),
                                       iterations=200, seed=42)

                # 计算节点度用于大小和颜色
                degrees = dict(disease_subgraph.degree())
                max_degree = max(degrees.values()) if degrees else 1

                # 节点大小基于度（对数缩放避免太大差异）
                node_sizes = [200 + 800 * np.log(degree + 1) for degree in degrees.values()]

                # 节点颜色基于度（使用viridis色彩映射）
                node_colors = [degrees[node] for node in disease_nodes]

                # 首先绘制边（贝塞尔曲线）
                print("🔄 绘制贝塞尔曲线边...")
                edge_weights = [data.get('weight', 1) for _, _, data in disease_edges]
                max_weight = max(edge_weights) if edge_weights else 1

                for u, v, edge_data in disease_edges:
                    self._draw_bezier_edge(ax1, pos, u, v, edge_data, max_weight,
                                           color='steelblue', base_width=0.3)

                # 然后绘制节点（在边的上面）
                nodes = nx.draw_networkx_nodes(disease_subgraph, pos,
                                               nodelist=disease_nodes,
                                               node_size=node_sizes,
                                               node_color=node_colors,
                                               cmap='viridis',
                                               alpha=0.9,
                                               edgecolors='white',
                                               linewidths=1.5,
                                               ax=ax1)

                # 只标记高度中心性的疾病节点
                if len(disease_nodes) > 0:
                    try:
                        # 计算度中心性
                        degree_centrality = nx.degree_centrality(disease_subgraph)
                        # 选择前10个最重要的节点进行标记
                        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]

                        labels = {}
                        for node, centrality in top_nodes:
                            # 缩短长疾病名称
                            if len(node) > 10:
                                label = node[:8] + '..'
                            else:
                                label = node
                            labels[node] = label

                        # 绘制标签，添加背景色提高可读性
                        for node, label in labels.items():
                            x, y = pos[node]
                            ax1.text(x, y, label, fontsize=9, fontweight='bold',
                                     ha='center', va='center',
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                               alpha=0.8, edgecolor='none'))

                    except Exception as e:
                        print(f"⚠️  标签绘制失败: {e}")

                # 添加颜色条
                if nodes:
                    cbar = plt.colorbar(nodes, ax=ax1, shrink=0.8)
                    cbar.set_label('节点度', fontweight='bold')

                ax1.set_title('疾病共现网络\n(贝塞尔曲线边，节点大小和颜色表示疾病关联度)',
                              fontsize=14, fontweight='bold', pad=20)
                ax1.axis('off')

            # ===== 右侧：网络统计信息 =====
            print("📈 绘制网络统计信息...")

            # 隐藏坐标轴
            ax2.axis('off')

            # 添加统计信息文本
            stats_text = []

            # 基本统计
            stats_text.append(f"📊 网络统计信息")
            stats_text.append("=" * 30)
            stats_text.append(f"疾病节点数: {len(disease_nodes)}")
            stats_text.append(f"疾病共现边数: {len(disease_edges)}")

            if disease_nodes:
                # 度分布统计
                degrees = [d for n, d in disease_subgraph.degree()]
                stats_text.append(f"平均度: {np.mean(degrees):.2f}")
                stats_text.append(f"最大度: {max(degrees)}")
                stats_text.append(f"网络密度: {nx.density(disease_subgraph):.4f}")

                # 连通性统计
                connected_components = list(nx.connected_components(disease_subgraph))
                stats_text.append(f"连通分量: {len(connected_components)}")
                if connected_components:
                    largest_component = max(connected_components, key=len)
                    stats_text.append(f"最大分量: {len(largest_component)}节点")

            # 添加患者统计
            patient_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == 'patient']
            has_disease_edges = [(u, v) for u, v, attr in self.graph.edges(data=True) if
                                 attr.get('type') == 'has_disease']

            stats_text.append("")
            stats_text.append(f"👥 患者统计")
            stats_text.append("=" * 30)
            stats_text.append(f"患者节点数: {len(patient_nodes)}")
            stats_text.append(f"患者-疾病边数: {len(has_disease_edges)}")
            stats_text.append(f"总节点数: {self.graph.number_of_nodes()}")
            stats_text.append(f"总边数: {self.graph.number_of_edges()}")

            # 显示统计文本
            stats_str = "\n".join(stats_text)
            ax2.text(0.1, 0.95, stats_str, transform=ax2.transAxes, fontsize=12,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=12))

            # 如果有疾病节点，添加度分布直方图
            if disease_nodes and len(disease_nodes) > 1:
                # 在右侧下方添加度分布直方图
                ax_hist = fig.add_axes([0.55, 0.1, 0.15, 0.3])
                degrees = [d for n, d in disease_subgraph.degree()]
                ax_hist.hist(degrees, bins=min(20, len(set(degrees))),
                             alpha=0.7, color='skyblue', edgecolor='black')
                ax_hist.set_title('度分布', fontsize=10, fontweight='bold')
                ax_hist.set_xlabel('度', fontweight='bold')
                ax_hist.set_ylabel('频数', fontweight='bold')
                ax_hist.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"🎨 贝塞尔曲线图可视化已保存: {filename}")

        except Exception as e:
            print(f"⚠️  图可视化失败: {e}")
            import traceback
            traceback.print_exc()
            # 尝试简单的备用可视化
            try:
                self._create_simple_visualization(filename)
            except Exception as e2:
                print(f"❌ 备用可视化也失败: {e2}")

    def _create_simple_visualization(self, filename):
        """创建简化的备用可视化"""
        import matplotlib.pyplot as plt

        # 只绘制疾病节点
        disease_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == 'disease']
        disease_edges = [(u, v) for u, v, attr in self.graph.edges(data=True) if attr.get('type') == 'cooccurrence']

        if not disease_nodes:
            print("⚠️  没有疾病节点可可视化")
            return

        disease_subgraph = self.graph.subgraph(disease_nodes)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(disease_subgraph, seed=42)

        # 绘制简单的网络
        nx.draw(disease_subgraph, pos,
                node_color='lightblue',
                node_size=100,
                edge_color='gray',
                alpha=0.6,
                with_labels=False)

        plt.title(f"疾病共现网络 ({len(disease_nodes)}种疾病, {len(disease_edges)}条关系)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 简化版图可视化已保存: {filename}")

    def visualize_disease_communities(self, filename="disease_communities.png"):
        """可视化疾病社区结构 - 使用贝塞尔曲线"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from community import community_louvain  # 需要 pip install python-louvain

            # 提取疾病子图
            disease_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == 'disease']
            if not disease_nodes:
                print("⚠️  没有疾病节点进行社区分析")
                return

            disease_subgraph = self.graph.subgraph(disease_nodes)

            # 检测社区
            partition = community_louvain.best_partition(disease_subgraph)

            # 设置图形
            plt.figure(figsize=(16, 12))

            # 计算布局
            pos = nx.spring_layout(disease_subgraph, k=2 / np.sqrt(len(disease_nodes)),
                                   iterations=200, seed=42)

            # 为每个社区分配颜色
            communities = set(partition.values())
            colors = plt.cm.tab20(np.linspace(0, 1, len(communities)))
            community_colors = {comm: colors[i] for i, comm in enumerate(communities)}

            # 首先绘制边（贝塞尔曲线）
            edge_weights = [disease_subgraph[u][v].get('weight', 1) for u, v in disease_subgraph.edges()]
            max_weight = max(edge_weights) if edge_weights else 1

            for u, v in disease_subgraph.edges():
                edge_data = disease_subgraph[u][v]
                self._draw_bezier_edge(plt.gca(), pos, u, v, edge_data, max_weight,
                                       color='lightgray', base_width=0.2)

            # 然后绘制节点（按社区着色）
            for community in communities:
                nodes_in_community = [node for node in disease_nodes if partition[node] == community]
                nx.draw_networkx_nodes(disease_subgraph, pos,
                                       nodelist=nodes_in_community,
                                       node_color=[community_colors[community]],
                                       node_size=300,
                                       alpha=0.9,
                                       edgecolors='white',
                                       linewidths=2,
                                       label=f'社区 {community + 1}')

            # 只标记主要节点
            degrees = dict(disease_subgraph.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:15]
            labels = {}
            for node, _ in top_nodes:
                if len(node) > 10:
                    labels[node] = node[:8] + '..'
                else:
                    labels[node] = node

            # 绘制标签，添加背景色
            for node, label in labels.items():
                x, y = pos[node]
                plt.text(x, y, label, fontsize=8, fontweight='bold',
                         ha='center', va='center',
                         bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                   alpha=0.9, edgecolor='none'))

            plt.title(f"疾病社区结构 (共{len(communities)}个社区)", fontsize=16, fontweight='bold', pad=20)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

            print(f"🎨 疾病社区可视化已保存: {filename}")

            # 打印社区信息
            print(f"📊 发现 {len(communities)} 个疾病社区:")
            for community in communities:
                nodes_in_community = [node for node in disease_nodes if partition[node] == community]
                print(f"  社区 {community + 1}: {len(nodes_in_community)} 种疾病")
                # 显示社区内最常见的疾病
                community_degrees = {node: degrees[node] for node in nodes_in_community}
                top_diseases = sorted(community_degrees.items(), key=lambda x: x[1], reverse=True)[:3]
                for disease, deg in top_diseases:
                    print(f"    - {disease} (度: {deg})")

        except ImportError:
            print("⚠️  未安装 python-louvain，跳过社区分析")
            print("💡 运行: pip install python-louvain")
        except Exception as e:
            print(f"⚠️  社区可视化失败: {e}")

    def create_interactive_visualization(self, filename="interactive_graph.html"):
        """创建交互式可视化（可选功能）"""
        try:
            import plotly.graph_objects as go
            import plotly.offline as pyo

            # 提取疾病子图
            disease_nodes = [n for n, attr in self.graph.nodes(data=True) if attr.get('type') == 'disease']
            disease_subgraph = self.graph.subgraph(disease_nodes)

            if not disease_nodes:
                print("⚠️  没有疾病节点进行交互式可视化")
                return

            pos = nx.spring_layout(disease_subgraph, seed=42)

            # 准备节点数据
            node_x = []
            node_y = []
            node_text = []
            node_size = []
            node_color = []

            degrees = dict(disease_subgraph.degree())
            max_degree = max(degrees.values()) if degrees else 1

            for node in disease_nodes:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(f"{node}<br>度: {degrees[node]}")
                node_size.append(10 + 20 * (degrees[node] / max_degree))
                node_color.append(degrees[node])

            # 准备边数据
            edge_x = []
            edge_y = []

            for edge in disease_subgraph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            # 创建图形
            edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                    line=dict(width=0.5, color='#888'),
                                    hoverinfo='none',
                                    mode='lines')

            node_trace = go.Scatter(x=node_x, y=node_y,
                                    mode='markers',
                                    hoverinfo='text',
                                    text=node_text,
                                    marker=dict(
                                        showscale=True,
                                        colorscale='Viridis',
                                        size=node_size,
                                        color=node_color,
                                        colorbar=dict(
                                            thickness=15,
                                            title='节点度',
                                            xanchor='left',
                                            titleside='right'
                                        ),
                                        line_width=2))

            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                title='疾病共现网络 - 交互式可视化',
                                titlefont_size=16,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20, l=5, r=5, t=40),
                                annotations=[dict(
                                    text="使用Plotly创建的交互式疾病网络",
                                    showarrow=False,
                                    xref="paper", yref="paper",
                                    x=0.005, y=-0.002)],
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                            )

            pyo.plot(fig, filename=filename, auto_open=False)
            print(f"🎨 交互式可视化已保存: {filename}")

        except ImportError:
            print("⚠️  未安装 plotly，跳过交互式可视化")
            print("💡 运行: pip install plotly")
        except Exception as e:
            print(f"⚠️  交互式可视化失败: {e}")