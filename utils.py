# utils.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def save_training_plot(train_losses, test_accuracies, filename='training_plot.png'):
    """保存训练过程图表 - 优化版"""
    plt.figure(figsize=(15, 6))

    # 设置样式
    plt.style.use('default')
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    # 左侧：训练损失
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', linewidth=2.5, alpha=0.8, label='Training Loss')
    plt.title('Training Loss', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Loss', fontweight='bold')
    plt.legend()

    # 右侧：测试准确率
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, 'g-', linewidth=2.5, alpha=0.8, label='Test Accuracy')
    plt.title('Test Accuracy', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Epoch', fontweight='bold')
    plt.ylabel('Accuracy', fontweight='bold')
    plt.ylim(0, 1.0)
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 训练图表已保存: {filename}")


def visualize_predictions(node_features, true_labels, predicted_labels, test_mask,
                          model_name="AdaptiveGCN", filename="predictions.png"):
    """优化版预测结果可视化 - 专业美观"""
    try:
        # 设置中文字体和样式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        plt.style.use('default')

        # 转换数据
        x_np = node_features.detach().numpy() if hasattr(node_features, 'detach') else node_features
        true_labels_np = true_labels.detach().numpy() if hasattr(true_labels, 'detach') else true_labels
        pred_labels_np = predicted_labels.detach().numpy() if hasattr(predicted_labels, 'detach') else predicted_labels
        test_mask_np = test_mask.detach().numpy() if hasattr(test_mask, 'detach') else test_mask

        # 计算准确率
        train_accuracy = (predicted_labels[~test_mask] == true_labels[~test_mask]).float().mean().item()
        test_accuracy = (predicted_labels[test_mask] == true_labels[test_mask]).float().mean().item()
        overall_accuracy = (predicted_labels == true_labels).float().mean().item()

        # 创建多面板图形
        fig = plt.figure(figsize=(20, 12))

        # ===== 面板1: t-SNE可视化 =====
        print("🔄 进行t-SNE降维...")
        ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2, rowspan=2)

        # 使用t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(x_np) - 1))
        x_2d = tsne.fit_transform(x_np)

        # 定义类别颜色和标签
        unique_labels = np.unique(np.concatenate([true_labels_np, pred_labels_np]))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))

        # 分别绘制训练集和测试集
        train_indices = ~test_mask_np
        test_indices = test_mask_np

        # 训练集（实心圆）
        for i, label in enumerate(unique_labels):
            mask = (true_labels_np == label) & train_indices
            if np.any(mask):
                ax1.scatter(x_2d[mask, 0], x_2d[mask, 1],
                            c=[colors[i]], label=f'Train Class {label}',
                            s=60, alpha=0.8, marker='o', edgecolors='white', linewidth=0.5)

        # 测试集（带边框的星形）
        for i, label in enumerate(unique_labels):
            mask = (true_labels_np == label) & test_indices
            if np.any(mask):
                # 正确预测的测试样本
                correct_mask = mask & (pred_labels_np == true_labels_np)
                if np.any(correct_mask):
                    ax1.scatter(x_2d[correct_mask, 0], x_2d[correct_mask, 1],
                                c=[colors[i]], label=f'Test Class {label} (Correct)',
                                s=100, alpha=1.0, marker='*', edgecolors='green', linewidth=2)

                # 错误预测的测试样本
                wrong_mask = mask & (pred_labels_np != true_labels_np)
                if np.any(wrong_mask):
                    ax1.scatter(x_2d[wrong_mask, 0], x_2d[wrong_mask, 1],
                                c=[colors[i]], label=f'Test Class {label} (Wrong)',
                                s=100, alpha=1.0, marker='*', edgecolors='red', linewidth=2)

        ax1.set_title(f'{model_name} - 预测结果可视化 (t-SNE)', fontsize=16, fontweight='bold', pad=20)
        ax1.set_xlabel('t-SNE Component 1', fontweight='bold')
        ax1.set_ylabel('t-SNE Component 2', fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        # ===== 面板2: 混淆矩阵 =====
        ax2 = plt.subplot2grid((2, 3), (0, 2))

        # 计算混淆矩阵
        cm = confusion_matrix(true_labels_np, pred_labels_np)

        # 绘制热力图
        im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax2.set_title('混淆矩阵', fontsize=14, fontweight='bold', pad=10)

        # 添加数值标签
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black",
                         fontweight='bold')

        ax2.set_xlabel('预测标签', fontweight='bold')
        ax2.set_ylabel('真实标签', fontweight='bold')
        plt.colorbar(im, ax=ax2, shrink=0.8)

        # ===== 面板3: 准确率统计 =====
        ax3 = plt.subplot2grid((2, 3), (1, 2))
        ax3.axis('off')

        # 准备统计文本
        stats_text = [
            f"📊 模型性能统计",
            "=" * 30,
            f"整体准确率: {overall_accuracy:.4f}",
            f"训练集准确率: {train_accuracy:.4f}",
            f"测试集准确率: {test_accuracy:.4f}",
            "",
            f"📈 数据分布",
            "=" * 30,
            f"总样本数: {len(true_labels_np)}",
            f"训练样本: {np.sum(~test_mask_np)}",
            f"测试样本: {np.sum(test_mask_np)}",
            f"类别数: {len(unique_labels)}",
            "",
            f"🎯 预测结果",
            "=" * 30,
            f"正确预测: {np.sum(pred_labels_np == true_labels_np)}",
            f"错误预测: {np.sum(pred_labels_np != true_labels_np)}"
        ]

        # 显示统计文本
        stats_str = "\n".join(stats_text)
        ax3.text(0.1, 0.95, stats_str, transform=ax3.transAxes, fontsize=11,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # ===== 面板4: 类别分布柱状图 =====
        # 在右侧下方添加类别分布
        ax4 = fig.add_axes([0.75, 0.1, 0.2, 0.25])

        true_counts = [np.sum(true_labels_np == label) for label in unique_labels]
        pred_counts = [np.sum(pred_labels_np == label) for label in unique_labels]

        x_pos = np.arange(len(unique_labels))
        width = 0.35

        ax4.bar(x_pos - width / 2, true_counts, width, label='真实分布', alpha=0.7, color='skyblue')
        ax4.bar(x_pos + width / 2, pred_counts, width, label='预测分布', alpha=0.7, color='lightcoral')

        ax4.set_xlabel('类别', fontweight='bold')
        ax4.set_ylabel('样本数量', fontweight='bold')
        ax4.set_title('类别分布对比', fontsize=12, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'Class {label}' for label in unique_labels])
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"✅ 优化版预测可视化已保存: {filename}")
        print(f"📊 准确率统计: 整体={overall_accuracy:.4f}, 训练集={train_accuracy:.4f}, 测试集={test_accuracy:.4f}")

    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        # 创建简单的备选可视化
        _create_simple_prediction_visualization(node_features, true_labels, predicted_labels, test_mask, model_name,
                                                filename)


def _create_simple_prediction_visualization(node_features, true_labels, predicted_labels, test_mask, model_name,
                                            filename):
    """创建简化的预测可视化（备用方案）"""
    try:
        plt.figure(figsize=(15, 6))

        # 左侧：真实标签 vs 预测标签
        plt.subplot(1, 2, 1)
        colors = ['green' if pred == true else 'red'
                  for pred, true in zip(predicted_labels, true_labels)]
        plt.scatter(range(len(true_labels)), true_labels.numpy(),
                    c=colors, alpha=0.6, s=50)
        plt.title(f'{model_name} - 预测正确性')
        plt.xlabel('样本索引')
        plt.ylabel('类别')

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='正确预测'),
            Patch(facecolor='red', alpha=0.7, label='错误预测')
        ]
        plt.legend(handles=legend_elements)

        # 右侧：准确率统计
        plt.subplot(1, 2, 2)
        categories = ['整体准确率', '训练集准确率', '测试集准确率']
        train_accuracy = (predicted_labels[~test_mask] == true_labels[~test_mask]).float().mean().item()
        test_accuracy = (predicted_labels[test_mask] == true_labels[test_mask]).float().mean().item()
        overall_accuracy = (predicted_labels == true_labels).float().mean().item()

        accuracies = [overall_accuracy, train_accuracy, test_accuracy]
        bars = plt.bar(categories, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.ylim(0, 1.0)
        plt.title('准确率统计')
        plt.ylabel('准确率')

        # 在柱子上添加数值
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                     f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(filename.replace('.png', '_simple.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 简化版预测可视化已保存: {filename.replace('.png', '_simple.png')}")

    except Exception as e2:
        print(f"❌ 备用可视化也失败: {e2}")


def visualize_node_embeddings(model, x, y_true, model_name, filename='embeddings.png'):
    """可视化学习到的节点嵌入（如果模型支持）"""
    try:
        # 设置样式
        plt.style.use('default')
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

        # 尝试获取模型的嵌入表示
        if hasattr(model, 'E_A') and model.E_A is not None:
            # 自适应GCN：使用学习到的嵌入
            embeddings = torch.mm(x, model.E_A).detach().numpy()
            title_suffix = "学习到的嵌入"
        else:
            # 其他模型：使用原始特征或最后一层前的表示
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'fc1'):
                    # 获取最后一层前的表示
                    embeddings = model.fc1(x).detach().numpy()
                    title_suffix = "隐藏层表示"
                else:
                    embeddings = x.numpy()
                    title_suffix = "输入特征"

        # t-SNE降维
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        embeddings_2d = tsne.fit_transform(embeddings)

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                              c=y_true, cmap='tab10', alpha=0.7, s=50,
                              edgecolors='white', linewidth=0.5)
        plt.title(f'{model_name} - {title_suffix}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('t-SNE Component 1', fontweight='bold')
        plt.ylabel('t-SNE Component 2', fontweight='bold')

        # 添加颜色条
        cbar = plt.colorbar(scatter, shrink=0.8)
        cbar.set_label('节点类别', fontweight='bold')

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ 节点嵌入图已保存: {filename}")

    except Exception as e:
        print(f"⚠️  无法生成节点嵌入图: {e}")


def create_output_directories():
    """创建输出目录"""
    directories = ['plots', 'checkpoints']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 创建目录: {directory}/")


def plot_model_comparison(results_dict, filename='model_comparison.png'):
    """比较不同模型的性能"""
    if not results_dict:
        return

    plt.figure(figsize=(12, 8))

    # 设置样式
    plt.style.use('default')
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    models = list(results_dict.keys())
    accuracies = [results_dict[model]['final_accuracy'] for model in models]

    # 使用渐变色
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = plt.bar(models, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

    plt.title('模型性能比较', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('测试准确率', fontweight='bold')
    plt.ylim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')

    # 在柱子上添加数值
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{acc:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ 模型比较图已保存: {filename}")


def print_training_summary(model_name, final_accuracy, train_time=None):
    """打印训练摘要"""
    print("\n" + "=" * 60)
    print("🏆 训练完成摘要")
    print("=" * 60)
    print(f"📊 模型: {model_name}")
    print(f"✅ 最终测试准确率: {final_accuracy:.4f}")
    if train_time:
        print(f"⏱️  训练时间: {train_time:.2f} 秒")
    print("=" * 60)


def plot_attention_weights(attention_weights, filename='attention_heatmap.png'):
    """可视化注意力权重热力图"""
    try:
        plt.figure(figsize=(12, 10))

        # 绘制热力图
        plt.imshow(attention_weights, cmap='viridis', aspect='auto')
        plt.colorbar(label='注意力权重')
        plt.title('自适应注意力权重热力图', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('目标节点', fontweight='bold')
        plt.ylabel('源节点', fontweight='bold')

        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✅ 注意力权重热力图已保存: {filename}")

    except Exception as e:
        print(f"⚠️  注意力权重可视化失败: {e}")


# 简化的工具函数，移除测试代码
if __name__ == "__main__":
    # 仅用于基本功能验证
    print("🔧 工具函数模块已加载")
    create_output_directories()