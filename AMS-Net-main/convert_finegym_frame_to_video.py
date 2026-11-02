src_file = "/root/lanyun-fs/AMS-Net-main/data/finegym/gym99_train_frame_v1.txt"
dst_file = "/root/lanyun-fs/AMS-Net-main/data/finegym/gym99_train_video_v1.txt"

with open(src_file, "r", encoding="utf-8") as f_in, open(dst_file, "w", encoding="utf-8") as f_out:
    for line in f_in:
        line = line.strip()

        if not line:
            continue

        items = line.split()

        if len(items) != 3:
            print("跳过格式异常行:", line)
            continue

        clip_name, num_frames, label = items

        video_name = clip_name + ".mp4"

        f_out.write(video_name + " " + label + "\n")

print("转换完成，保存到:", dst_file)
