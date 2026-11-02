import os
from collections import Counter

# --- 配置路径 ---
OLD_TXT = "/root/lanyun-fs/AMS-Net-main/data/finegym/gym99_train_video_v1.txt"
NEW_TXT = "/root/lanyun-fs/AMS-Net-main/data/finegym/gym99_train_cleaned.txt"
VIDEO_DIR = "/root/lanyun-fs/AMS-Net-main/data/finegym/subactions"

def run():
    if not os.path.exists(OLD_TXT):
        print(f"❌ 错误: 找不到原始标签文件 {OLD_TXT}")
        return

    # 1. 读取原始标签并记录原始类别分布
    with open(OLD_TXT, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    
    original_labels = [line.split()[-1] for line in lines]
    original_classes = set(original_labels)
    
    # 2. 开始清洗匹配
    kept_lines = []
    final_labels = []
    
    for line in lines:
        parts = line.split()
        video_entry = parts[0]
        label = parts[-1]
        
        # 提取文件名并检查是否存在
        video_name = os.path.basename(video_entry)
        if os.path.exists(os.path.join(VIDEO_DIR, video_name)):
            kept_lines.append(line)
            final_labels.append(label)

    # 3. 统计结果
    final_classes = set(final_labels)
    class_counts = Counter(final_labels)
    
    # 4. 写入新文件
    with open(NEW_TXT, 'w') as f:
        for kl in kept_lines:
            f.write(kl + "\n")

    # 5. 输出报告
    print("\n" + "="*50)
    print("📊 FineGym99 缩减与类别存活统计")
    print("="*50)
    print(f"✅ 视频匹配完成:")
    print(f"   - 原始总样本数: {len(lines)}")
    print(f"   - 现存总样本数: {len(kept_lines)}")
    print(f"   - 样本流失率:   {((len(lines)-len(kept_lines))/len(lines))*100:.2f}%")
    
    print(f"\n📂 类别存活统计:")
    print(f"   - 原始定义的类别数: {len(original_classes)} (Gym99)")
    print(f"   - 现存有视频的类别: {len(final_classes)}")
    print(f"   - 彻底消失的类别数: {len(original_classes) - len(final_classes)}")
    
    if len(final_classes) < len(original_classes):
        missing_classes = sorted(list(original_classes - final_classes), key=int)
        print(f"   - 消失的类别 ID 示例: {missing_classes[:10]}...")

    print(f"\n🔝 样本量最多的前 5 类:")
    for cls, count in class_counts.most_common(5):
        print(f"   - 类别 {cls}: {count} 个视频")

    print(f"\n📉 样本量最少的后 5 类:")
    sorted_counts = sorted(class_counts.items(), key=lambda x: x[1])
    for cls, count in sorted_counts[:5]:
        print(f"   - 类别 {cls}: {count} 个视频")

    print("\n" + "="*50)
    print(f"💾 清洗后的标签已保存至: {NEW_TXT}")
    print("="*50 + "\n")

if __name__ == "__main__":
    run()
