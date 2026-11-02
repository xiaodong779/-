import re
import matplotlib.pyplot as plt
import numpy as np

def smooth_curve(points, factor=0.85):
    """
    指数滑动平均 (Exponential Moving Average) 平滑算法。
    factor 越大，曲线越平滑，能完美过滤掉批次间的剧烈震荡。
    """
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def parse_and_plot(log_path):
    print(f"🚀 正在解析你的炼丹炉日志: {log_path} ...")
    
    losses = []
    top1_accs = []
    
    # 强大的正则表达式，精准抓取日志中的数字
    pattern = re.compile(r'Epoch \[\d+\]\[\d+/\d+\](?:.*?)top1_acc: (\d+\.\d+)(?:.*?)loss: (\d+\.\d+)')
    
    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                top1_accs.append(float(match.group(1)))
                losses.append(float(match.group(2)))
                
    if not losses:
        print("❌ 哎呀，没有从日志里找到匹配的数据，请检查日志路径或格式是否正确！")
        return

    print(f"✅ 成功提取了 {len(losses)} 个迭代步 (Iterations) 的数据！开始绘制...")

    # 计算平滑后的数据
    smoothed_losses = smooth_curve(losses, factor=0.90)
    smoothed_accs = smooth_curve(top1_accs, factor=0.90)
    iterations = np.arange(len(losses))

    # 设置高级感的图表样式
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=150)

    # ================= 绘制 Loss 曲线 =================
    ax1.plot(iterations, losses, color='lightcoral', alpha=0.3, label='Raw Loss (批次震荡)')
    ax1.plot(iterations, smoothed_losses, color='red', linewidth=2.5, label='Smoothed Loss (宏观趋势)')
    ax1.set_title('Training Loss Tracking', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # ================= 绘制 Top-1 Accuracy 曲线 =================
    ax2.plot(iterations, top1_accs, color='skyblue', alpha=0.3, label='Raw Top-1 Acc')
    ax2.plot(iterations, smoothed_accs, color='blue', linewidth=2.5, label='Smoothed Top-1 Acc (宏观趋势)')
    ax2.set_title('Top-1 Accuracy Tracking', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Iterations', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.legend(loc='lower right', fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 调整布局并保存
    plt.tight_layout()
    save_path = 'training_dynamics_chart.png'
    plt.savefig(save_path)
    print(f"🎉 大功告成！绝美的折线图已保存至: {save_path}")
    print("快去看看那条红色的 Loss 曲线是不是正在蓄力准备大跳水！")

if __name__ == '__main__':
    # 💥 请在这里替换为你服务器上真实的日志文件路径
    # 例如: 'work_dirs/mynet_r50_diving48_tcm_mgfm/20260310_123456.log'
    LOG_FILE_PATH = '你的日志文件路径.log' 
    parse_and_plot(LOG_FILE_PATH)
    