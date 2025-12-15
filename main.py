#!/usr/bin/env python3
# main2.py主文件
"""
Adaptive GCN
"""
import argparse
import sys
import os
import torch
import warnings
from gcn_models import AdaptiveGCN, initialize_model, print_model_summary, count_parameters
from utils import (save_training_plot, visualize_predictions,
                   create_output_directories, print_training_summary)
from data_loader import DataLoader

#有一些冲突问题
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

# 抑制警告
warnings.filterwarnings("ignore", category=UserWarning)

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 用了三个自定义模块，gcn_models, utils, data_loader




def train_adaptive_gcn(model, data, optimizer, criterion, epochs=200):
    """训练自适应GCN模型"""
    model.train()
    train_losses = []
    test_accuracies = []

    x, y, edge_index = data
    num_nodes = x.size(0)

    # 划分训练测试集 (80% 训练, 20% 测试)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[:int(0.8 * num_nodes)] = True
    test_mask = ~train_mask

    # 使用跨平台的计时方法
    import time
    start_time = time.time()

    print("🎯 开始训练自适应GCN...")
    for epoch in range(epochs):
        optimizer.zero_grad()

        # 自适应GCN前向传播
        output = model(x)

        loss = criterion(output[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        # 评估
        model.eval()
        with torch.no_grad():
            test_output = model(x)
            pred = test_output.argmax(dim=1)
            test_acc = (pred[test_mask] == y[test_mask]).float().mean()

        model.train()
        train_losses.append(loss.item())
        test_accuracies.append(test_acc.item())

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f'Epoch {epoch:3d} | Loss: {loss.item():.4f} | Test Acc: {test_acc:.4f}')

    training_time = time.time() - start_time

    return train_losses, test_accuracies, training_time


def run_training(data_file=None, epochs=200, lr=0.01, hidden_dim=128):
    """运行训练流程 - 专注于疾病数据"""
    print("=" * 60)
    print("🚀 自适应GCN疾病数据分析开始")
    print("=" * 60)
    print(f"📊 模型: AdaptiveGCN")
    print(f"🔄 训练轮数: {epochs}")
    print(f"📈 学习率: {lr}")
    print(f"🧠 隐藏层维度: {hidden_dim}")

    # 使用默认数据文件路径
    if data_file is None:
        data_file = DataLoader.DEFAULT_DATA_PATH
        print(f"使用默认数据文件: {data_file}")
    else:
        print(f"数据文件: {data_file}")
    print("=" * 60)

    # 检查文件是否存在
    if not os.path.exists(data_file):
        print(f"数据文件不存在: {data_file}")
        print("请检查文件路径或使用 --data-file 参数指定正确的路径")
        return 0.0

    # 设置随机种子
    torch.manual_seed(42)

    # 创建输出目录
    create_output_directories()

    # 加载数据
    print("加载疾病数据...")
    result = DataLoader.load_data(csv_file_path=data_file)
    if result is None:
        print("数据加载失败")
        return 0.0

    x, y, edge_index, patient_disease_map = result
    input_dim = x.shape[1]  # 动态获取嵌入维度
    num_classes = len(torch.unique(y))

    print(f"✅ 数据加载成功:")
    print(f"   - 节点特征维度: {input_dim}")
    print(f"   - 节点数量: {x.shape[0]}")
    print(f"   - 类别数量: {num_classes}")
    print(f"   - 标签分布: {torch.bincount(y).tolist()}")

    # 创建自适应GCN模型
    print(f"🛠️  创建自适应GCN模型...")
    model = AdaptiveGCN(input_dim, hidden_dim, num_classes)
    model = initialize_model(model)

    print_model_summary(model, input_dim)

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    criterion = torch.nn.NLLLoss()

    # 训练模型
    train_losses, test_accuracies, training_time = train_adaptive_gcn(
        model, (x, y, edge_index), optimizer, criterion, epochs=epochs
    )

    # 最终评估
    model.eval()
    with torch.no_grad():
        final_output = model(x)
        num_nodes = x.size(0)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[:int(0.8 * num_nodes)] = True
        test_mask = ~train_mask

        pred = final_output.argmax(dim=1)
        final_acc = (pred[test_mask] == y[test_mask]).float().mean()

        # 计算整体准确率
        overall_acc = (pred == y).float().mean()

    # 保存训练图表
    save_training_plot(
        train_losses, test_accuracies,
        filename=f'plots/adaptive_gcn_training.png'
    )

    # 可视化预测结果
    visualize_predictions(
        x, y, pred, test_mask,
        model_name='AdaptiveGCN',
        filename=f'plots/adaptive_gcn_predictions.png'
    )

    # 打印训练摘要
    print_training_summary('AdaptiveGCN', final_acc.item(), training_time)
    print(f"📊 整体准确率: {overall_acc.item():.4f}")

    # 保存模型
    os.makedirs('checkpoints', exist_ok=True)
    model_path = f'checkpoints/adaptive_gcn_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': 'AdaptiveGCN',
        'final_accuracy': final_acc.item(),
        'overall_accuracy': overall_acc.item(),
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'num_classes': num_classes,
        'training_args': {
            'epochs': epochs,
            'lr': lr,
            'hidden_dim': hidden_dim
        }
    }, model_path)
    print(f"💾 模型已保存到: {model_path}")

    # 保存注意力权重用于分析
    try:
        attention_weights = model.get_attention_weights(x)
        import numpy as np
        np.save('checkpoints/attention_weights.npy', attention_weights)
        print(f"💾 注意力权重已保存: checkpoints/attention_weights.npy")

        # 新增：可视化注意力权重
        from utils import plot_attention_weights
        plot_attention_weights(attention_weights, 'plots/attention_heatmap.png')

        # 打印注意力权重的统计信息
        print(f"📊 注意力权重统计:")
        print(f"   - 最小值: {attention_weights.min():.6f}")
        print(f"   - 最大值: {attention_weights.max():.6f}")
        print(f"   - 平均值: {attention_weights.mean():.6f}")
        print(f"   - 标准差: {attention_weights.std():.6f}")

    except Exception as e:
        print(f"⚠️  保存注意力权重失败: {e}")

    return final_acc.item()


def run_parameter_analysis(data_file=None, hidden_dims=[64, 128, 256], learning_rates=[0.01, 0.001]):
    """运行参数分析"""
    print("=" * 60)
    print("🧪 开始参数分析")
    print("=" * 60)

    # 使用默认数据文件路径
    if data_file is None:
        data_file = DataLoader.DEFAULT_DATA_PATH

    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return {}

    results = {}

    for hidden_dim in hidden_dims:
        for lr in learning_rates:
            print(f"\n🔍 测试参数: hidden_dim={hidden_dim}, lr={lr}")
            accuracy = run_training(
                data_file=data_file,
                epochs=100,  # 减少轮数以加快分析
                lr=lr,
                hidden_dim=hidden_dim
            )
            key = f"hidden_{hidden_dim}_lr_{lr}"
            results[key] = accuracy

    # 显示分析结果
    print("\n" + "=" * 60)
    print("📊 参数分析结果")
    print("=" * 60)
    for config, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"🔧 {config:25} | 准确率: {acc:.4f}")
    print("=" * 60)

    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='自适应GCN疾病数据分析系统')
    parser.add_argument('--data-file', type=str, default=None,
                        help='疾病数据文件路径（CSV格式），如果不指定则使用默认路径')
    parser.add_argument('--epochs', type=int, default=200, help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.01, help='学习率')
    parser.add_argument('--hidden-dim', type=int, default=128, help='隐藏层维度')
    parser.add_argument('--analyze', action='store_true',
                        help='运行参数分析（测试不同隐藏层维度和学习率）')

    args = parser.parse_args()

    if args.analyze:
        # 运行参数分析
        run_parameter_analysis(args.data_file)
    else:
        # 运行单次训练
        run_training(
            data_file=args.data_file,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim
        )


if __name__ == "__main__":
    main()
