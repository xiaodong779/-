import os
import argparse
import pickle
import numpy as np


def load_pkl(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'annotations' in data:
        return data['annotations']
    if isinstance(data, list):
        return data
    raise TypeError(f'Unsupported pkl format: {type(data)}')


def parse_rawframe_ann(ann_file):
    out = {}
    with open(ann_file, 'r') as f:
        for line in f:
            items = line.strip().split()
            if len(items) < 3:
                continue
            name = os.path.basename(items[0].rstrip('/'))
            out[name] = int(items[1])
    return out


def draw_gaussian(hm, cx, cy, sigma, value):
    h, w = hm.shape
    radius = int(3 * sigma + 0.5)

    x1 = max(0, cx - radius)
    x2 = min(w, cx + radius + 1)
    y1 = max(0, cy - radius)
    y2 = min(h, cy + radius + 1)

    if x1 >= x2 or y1 >= y2:
        return

    xs = np.arange(x1, x2, dtype=np.float32) - cx
    ys = np.arange(y1, y2, dtype=np.float32) - cy
    yy, xx = np.meshgrid(ys, xs, indexing='ij')

    patch = np.exp(-(xx * xx + yy * yy) / (2 * sigma * sigma))
    patch = patch * float(value)

    hm[y1:y2, x1:x2] = np.maximum(hm[y1:y2, x1:x2], patch)


def render_frame_kp_only(keypoint, score, img_shape, out_size, sigma, score_thr):
    img_h, img_w = img_shape[:2]
    hm = np.zeros((out_size, out_size), dtype=np.float32)

    M, V, _ = keypoint.shape

    for m in range(M):
        for v in range(V):
            s = float(score[m, v])
            if s <= score_thr:
                continue

            x, y = keypoint[m, v]
            if x <= 0 or y <= 0:
                continue

            cx = int(round(float(x) / max(img_w, 1) * out_size))
            cy = int(round(float(y) / max(img_h, 1) * out_size))

            if 0 <= cx < out_size and 0 <= cy < out_size:
                draw_gaussian(hm, cx, cy, sigma=sigma, value=s)

    hm = np.clip(hm, 0.0, 1.0)
    return hm[None, :, :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', required=True)
    parser.add_argument('--ann-files', nargs='+', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--out-size', type=int, default=224)
    parser.add_argument('--sigma', type=float, default=2.4)
    parser.add_argument('--score-thr', type=float, default=1e-4)
    parser.add_argument('--skip-existing', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ann_frames = {}
    for ann_file in args.ann_files:
        ann_frames.update(parse_rawframe_ann(ann_file))

    anns = load_pkl(args.pkl)
    print('pkl samples:', len(anns))
    print('ann samples:', len(ann_frames))
    print('out dir:', args.out_dir)
    print('out size:', args.out_size)
    print('sigma:', args.sigma)
    print('score_thr:', args.score_thr)

    saved = 0
    skipped = 0
    missing = 0

    for idx, anno in enumerate(anns):
        name = os.path.basename(anno['frame_dir'].rstrip('/'))

        if name not in ann_frames:
            missing += 1
            continue

        save_path = os.path.join(args.out_dir, f'{name}.npz')
        if args.skip_existing and os.path.exists(save_path):
            skipped += 1
            continue

        total_frames = ann_frames[name]

        keypoint = anno['keypoint']
        keypoint_score = anno['keypoint_score']
        img_shape = anno.get('img_shape', anno.get('original_shape'))

        if img_shape is None:
            raise KeyError(f'No img_shape/original_shape for {name}')

        T_pose = keypoint.shape[1]
        valid_T = min(total_frames, T_pose)

        heatmap = np.zeros(
            (total_frames, 1, args.out_size, args.out_size),
            dtype=np.uint8
        )

        for t in range(valid_T):
            hm = render_frame_kp_only(
                keypoint=keypoint[:, t],
                score=keypoint_score[:, t],
                img_shape=img_shape,
                out_size=args.out_size,
                sigma=args.sigma,
                score_thr=args.score_thr
            )
            heatmap[t] = (hm * 255.0).astype(np.uint8)

        np.savez_compressed(
            save_path,
            heatmap=heatmap,
            total_frames=np.array([total_frames], dtype=np.int32),
            start_index=np.array([1], dtype=np.int32),
            frame_dir=np.array([name])
        )

        saved += 1

        if saved % 200 == 0:
            print(f'saved {saved}, skipped {skipped}, missing {missing}')

    print('done')
    print('saved:', saved)
    print('skipped:', skipped)
    print('missing:', missing)


if __name__ == '__main__':
    main()
