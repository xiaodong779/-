import os
import glob
from collections import Counter

ann_file = "/hy-tmp/AMS-Net-main/data/diving48/annotations/diving48_train_rawframes.txt"
raw_root = "/hy-tmp/Diving48_rawframes"
report_file = "/hy-tmp/check_diving48_train_report.txt"

missing = []
mismatch = []
bad_format = []
bad_label = []
duplicates = []

video_ids = []
labels = Counter()
total = 0

with open(ann_file, "r") as f:
    for line_idx, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 3:
            bad_format.append((line_idx, line))
            continue

        video_id = parts[0]

        try:
            expected_frames = int(parts[1])
            label = int(parts[2])
        except Exception:
            bad_format.append((line_idx, line))
            continue

        total += 1
        video_ids.append(video_id)
        labels[label] += 1

        if label < 0 or label > 47:
            bad_label.append((line_idx, video_id, label))

        frame_dir = os.path.join(raw_root, video_id)

        if not os.path.isdir(frame_dir):
            missing.append((line_idx, video_id, expected_frames, label))
            continue

        real_frames = len(glob.glob(os.path.join(frame_dir, "img_*.jpg")))

        if real_frames != expected_frames:
            mismatch.append((line_idx, video_id, expected_frames, real_frames, label))

cnt = Counter(video_ids)
for vid, c in cnt.items():
    if c > 1:
        duplicates.append((vid, c))

with open(report_file, "w") as w:
    w.write("===== Diving48 Train Rawframes Check =====\n")
    w.write(f"Annotation file: {ann_file}\n")
    w.write(f"Rawframes root:  {raw_root}\n\n")

    w.write(f"Total annotation lines: {total}\n")
    w.write(f"Unique video ids:       {len(cnt)}\n")
    w.write(f"Duplicate video ids:    {len(duplicates)}\n")
    w.write(f"Bad format lines:       {len(bad_format)}\n")
    w.write(f"Bad labels:             {len(bad_label)}\n")
    w.write(f"Missing folders:        {len(missing)}\n")
    w.write(f"Frame mismatches:       {len(mismatch)}\n")
    w.write(f"Num classes:            {len(labels)}\n")

    if labels:
        w.write(f"Label range:            {min(labels.keys())} to {max(labels.keys())}\n")

    w.write("\n===== Missing folders examples =====\n")
    for x in missing[:100]:
        w.write(str(x) + "\n")

    w.write("\n===== Frame mismatch examples =====\n")
    w.write("format: line_idx, video_id, expected_frames, real_frames, label\n")
    for x in mismatch[:100]:
        w.write(str(x) + "\n")

    w.write("\n===== Bad label examples =====\n")
    for x in bad_label[:100]:
        w.write(str(x) + "\n")

    w.write("\n===== Duplicate examples =====\n")
    for x in duplicates[:100]:
        w.write(str(x) + "\n")

    w.write("\n===== Label distribution =====\n")
    for k in sorted(labels.keys()):
        w.write(f"{k}: {labels[k]}\n")

print("===== Diving48 Train Rawframes Check =====")
print("Annotation file:", ann_file)
print("Rawframes root: ", raw_root)
print("Total annotation lines:", total)
print("Unique video ids:", len(cnt))
print("Duplicate video ids:", len(duplicates))
print("Bad format lines:", len(bad_format))
print("Bad labels:", len(bad_label))
print("Missing folders:", len(missing))
print("Frame mismatches:", len(mismatch))
print("Num classes:", len(labels))
print("Label range:", (min(labels.keys()), max(labels.keys())) if labels else None)
print("Report saved to:", report_file)

if len(missing) == 0 and len(mismatch) == 0 and len(bad_format) == 0 and len(bad_label) == 0:
    print("\n✅ 检查通过：标注文件、帧文件夹、帧数、标签基本一一对应。")
else:
    print("\n❌ 检查发现问题，请查看报告：", report_file)
