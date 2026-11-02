src_file = "/root/lanyun-fs/AMS-Net-main/data/finegym/gym99_val_frame.txt"
dst_file = "/root/lanyun-fs/AMS-Net-main/data/finegym/gym99_val_video.txt"

video_prefix = "subactions"

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

        video_path = video_prefix + "/" + clip_name + ".mp4"

        f_out.write(video_path + " " + label + "\n")

print("转换完成，保存到:", dst_file)
