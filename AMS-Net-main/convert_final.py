import json
import os

# ================= 配置区域 =================
# 1. 源文件路径 (你刚才提供的路径)
SRC_TRAIN_JSON = '/root/lanyun-fs/data/Diving48_V2_json/Diving48_V2_train.json'
SRC_TEST_JSON  = '/root/lanyun-fs/data/Diving48_V2_json/Diving48_V2_test.json'

# 2. 目标保存目录 (MMAction2 项目的标准标注目录)
DST_DIR = '/root/lanyun-fs/AMS-Net-main/data/diving48/annotations'
# ===========================================

def convert_json_to_txt(json_path, txt_filename):
    # 1. 检查源文件是否存在
    if not os.path.exists(json_path):
        print(f"❌ 错误: 找不到源文件 {json_path}")
        return

    # 2. 准备目标路径
    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)
        print(f"创建目录: {DST_DIR}")
    
    dst_path = os.path.join(DST_DIR, txt_filename)

    # 3. 读取并转换
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败 {json_path}: {e}")
        return

    lines = []
    for item in data:
        vid_name = item['vid_name']
        label = item['label']
        
        # 强制补全 .mp4 后缀 (适配 VideoDataset)
        if not vid_name.endswith('.mp4'):
            vid_name += '.mp4'
            
        lines.append(f"{vid_name} {label}")

    # 4. 写入 txt
    with open(dst_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"✅ 成功转换: {json_path}")
    print(f"   -> 保存为: {dst_path}")
    print(f"   -> 样本数量: {len(lines)}")
    print("-" * 40)

if __name__ == '__main__':
    print("🚀 开始转换 Diving48 V2 标签...")
    print("-" * 40)
    
    # 转换训练集
    convert_json_to_txt(SRC_TRAIN_JSON, 'train_v2.txt')
    
    # 转换测试集 (生成两个文件分别用于 val 和 test)
    convert_json_to_txt(SRC_TEST_JSON, 'val_v2.txt')
    convert_json_to_txt(SRC_TEST_JSON, 'test_v2.txt')
    
    print("\n🎉 全部完成！现在可以直接开始训练了。")