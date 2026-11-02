import os

txt_file = '/root/lanyun-fs/AMS-Net-main/data/finegym/gym99_train_video_v1.txt'
video_dir = '/root/lanyun-fs/AMS-Net-main/data/finegym/subactions'

print(f"正在读取标签文件: {txt_file}...")
with open(txt_file, 'r') as f:
    # 忽略空行
    lines = [line.strip() for line in f if line.strip()]

print(f"正在扫描硬盘文件夹: {video_dir}...")
try:
    total_files_on_disk = len(os.listdir(video_dir))
except FileNotFoundError:
    print(f"❌ 错误：找不到文件夹 {video_dir}，请检查路径。")
    exit(1)

valid_count = 0
missing_files = []

for line in lines:
    vid_name = line.split()[0]
    # 处理后缀问题，确保带 .mp4
    if not vid_name.endswith('.mp4'):
        vid_name += '.mp4'
    
    vid_path = os.path.join(video_dir, vid_name)
    
    if os.path.exists(vid_path):
        valid_count += 1
    else:
        missing_files.append(vid_name)

print("\n" + "="*50)
print("🎯 FineGym99 解压与匹配结果核对报告")
print("="*50)
print(f"📁 硬盘文件夹内目前的总视频数 (含非99类) : {total_files_on_disk}")
print(f"📄 Gym99_train.txt 要求的 99 类视频总数  : {len(lines)}")
print("-" * 50)
print(f"✅ 成功匹配（Gym99 视频已就位）           : {valid_count}")
print(f"❌ 缺失匹配（文本里有但硬盘上找不到）     : {len(missing_files)}")
print("="*50)

if missing_files:
    print("\n⚠️ 前 5 个缺失的视频文件示例：")
    for f in missing_files[:5]:
        print(f"   - {f}")
    print("\n👉 提示：由于网络爬取的不稳定性，原始压缩包缺失极少量视频是正常的（通常缺失率在 1% 以内）。")
    print("你可以选择忽视，或者写个脚本把 txt 里找不到的这几行删掉。")
else:
    print("\n🎉 完美匹配！Gym99 训练集所需的所有视频均已安全躺在你的硬盘里，可以开始训练了！")
