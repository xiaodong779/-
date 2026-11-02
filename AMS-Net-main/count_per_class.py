import os
from collections import Counter

# --- 配置路径 (请确保这是你清洗后的那个文件) ---
CLEANED_TXT = "/root/lanyun-fs/AMS-Net-main/data/finegym/gym99_train_cleaned.txt"

def show_stats():
    if not os.path.exists(CLEANED_TXT):
        print(f"❌ 错误: 找不到清洗后的标签文件 {CLEANED_TXT}")
        return

    # 读取并统计
    with open(CLEANED_TXT, 'r') as f:
        labels = [line.strip().split()[-1] for line in f if line.strip()]
    
    counts = Counter(labels)
    
    print("\n" + "="*60)
    print(f"📊 FineGym99 每类视频数量详细统计表 (总计: {len(labels)} 个)")
    print("="*60)
    print(f"{'类别ID':<10} | {'视频数量':<10} || {'类别ID':<10} | {'视频数量':<10}")
    print("-" * 60)

    # 按 0-98 顺序排列，为了美观采用两列显示
    for i in range(0, 50):
        c1 = str(i)
        n1 = counts.get(c1, 0)
        
        c2 = str(i + 50)
        n2 = counts.get(c2, 0) if i + 50 < 99 else "-"
        
        print(f"Class {c1:<6} | {n1:<10} || Class {c2:<6} | {n2:<10}")

    print("-" * 60)
    
    # 找出预警类别
    low_data_classes = [cls for cls, count in counts.items() if count < 20]
    if low_data_classes:
        print(f"⚠️ 警告: 以下类别样本数极少 (<20个)，训练时可能难以识别:")
        print(f"   {sorted(low_data_classes, key=int)}")
    else:
        print("✅ 良好: 所有类别样本数均在 20 个以上。")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    show_stats()
