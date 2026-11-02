import os

# 配置路径
TXT_PATH = "/root/lanyun-fs/AMS-Net-main/data/finegym/gym99_train_video_v1.txt"
VIDEO_DIR = "/root/lanyun-fs/AMS-Net-main/data/finegym/subactions"

def run_check():
    print("\n" + "="*50)
    print("FineGym 数据集匹配检查工具")
    print("="*50)

    if not os.path.exists(TXT_PATH):
        print(f"❌ 错误: 找不到标签文件: {TXT_PATH}")
        return
    if not os.path.exists(VIDEO_DIR):
        print(f"❌ 错误: 找不到视频目录: {VIDEO_DIR}")
        return

    with open(TXT_PATH, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    total = len(lines)
    found = 0
    missing = []

    # 检查前几个文件作为示例
    sample_check = True
    
    for line in lines:
        # 兼容处理：取第一列作为路径/文件名
        video_entry = line.split()[0]
        # 提取纯文件名（处理标签中可能带有的 subactions/ 前缀）
        video_name = os.path.basename(video_entry)
        
        # 拼接实际路径
        video_path = os.path.join(VIDEO_DIR, video_name)
        
        if os.path.exists(video_path):
            found += 1
        else:
            missing.append(video_name)

    print(f"📊 检查统计结果:")
    print(f"   - 标签文件总行数: {total}")
    print(f"   - 成功匹配视频数: {found}")
    print(f"   - 缺失视频数量:   {len(missing)}")

    if len(missing) == 0:
        print("\n✅ 完美匹配！你可以直接开始训练。")
    elif found > 0:
        print(f"\n⚠️ 部分匹配。匹配率: {(found/total)*100:.2f}%")
        print("   缺失示例 (前5个):")
        for m in missing[:5]:
            print(f"    - {m}")
        print("\n   [建议]: 如果缺失较少，可继续训练；若缺失较多，请检查下载脚本。")
    else:
        print("\n❌ 完全不匹配！")
        # 诊断原因
        actual_files = os.listdir(VIDEO_DIR)[:3]
        print(f"   [诊断]: 标签期望的文件名类似: {os.path.basename(lines[0].split()[0])}")
        print(f"   [诊断]: 文件夹内实际文件名类似: {actual_files if actual_files else '文件夹为空'}")
        print("   [建议]: 请检查是否是 '.mp4' 后缀名不一致或路径层级问题。")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    run_check()
