import cv2
import numpy as np
import argparse
import os

# ========= 以下函数完全不变（与最初版本相同） =========
def motion_compensate(frame1, frame2):
    lk_params = dict(winSize=(15, 15), maxLevel=3,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003))

    width, height = frame2.shape[1], frame2.shape[0]
    scale = 0.5
    frame1_grid = cv2.resize(frame1, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)
    frame2_grid = cv2.resize(frame2, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)

    width_grid, height_grid = frame2_grid.shape[1], frame2_grid.shape[0]
    gridSizeW, gridSizeH = 32 * 2, 24 * 2

    p1 = []
    grid_numW = int(width_grid / gridSizeW - 1)
    grid_numH = int(height_grid / gridSizeH - 1)
    for i in range(grid_numW):
        for j in range(grid_numH):
            point = (np.float32(i * gridSizeW + gridSizeW / 2.0),
                     np.float32(j * gridSizeH + gridSizeH / 2.0))
            p1.append(point)
    p1 = np.array(p1).reshape(-1, 1, 2)

    pts_cur, st, err = cv2.calcOpticalFlowPyrLK(frame1_grid, frame2_grid, p1, None, **lk_params)
    good_new, good_old = pts_cur[st == 1], p1[st == 1]

    motion_distance, translate_x, translate_y, valid_new, valid_old = [], [], [], [], []
    for new, old in zip(good_new, good_old):
        a, b = new.ravel()
        c, d = old.ravel()
        dist = np.sqrt((a - c) ** 2 + (b - d) ** 2)
        if dist > 50:
            continue
        motion_distance.append(dist)
        translate_x.append(a - c)
        translate_y.append(b - d)
        valid_new.append(new)
        valid_old.append(old)

    valid_new, valid_old = np.array(valid_new), np.array(valid_old)
    motion_x = np.mean(translate_x) if translate_x else 0
    motion_y = np.mean(translate_y) if translate_y else 0
    avg_dst = np.mean(motion_distance) if motion_distance else 0

    if len(valid_old) < 15:
        homography_matrix = np.array([[0.999, 0, 0],
                                      [0, 0.999, 0],
                                      [0, 0, 1]])
    else:
        homography_matrix, _ = cv2.findHomography(valid_new, valid_old, cv2.RANSAC, 3.0)
        if homography_matrix is None:
            homography_matrix = np.array([[0.999, 0, 0],
                                          [0, 0.999, 0],
                                          [0, 0, 1]])

    compensated = cv2.warpPerspective(frame1, homography_matrix, (width, height),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

    vertex = np.array([[0, 0], [width, 0], [width, height], [0, height]],
                      dtype=np.float32).reshape(-1, 1, 2)
    try:
        vertex_trans = cv2.perspectiveTransform(vertex, np.linalg.inv(homography_matrix))
        vertex_transformed = np.array(vertex_trans, dtype=np.int32).reshape(1, 4, 2)
        mask = 255 - cv2.fillPoly(np.zeros(frame1.shape[:2], dtype='uint8'),
                                  vertex_transformed, 255)
    except:
        mask = np.zeros(frame1.shape[:2], dtype='uint8')
    return compensated, mask, avg_dst, motion_x, motion_y, homography_matrix


def generate_motion_difference(lastFrame1, lastFrame2, currentFrame):
    lastFrame1 = cv2.GaussianBlur(lastFrame1, (11, 11), 0)
    lastFrame1 = cv2.cvtColor(lastFrame1, cv2.COLOR_BGR2GRAY)
    lastFrame2 = cv2.GaussianBlur(lastFrame2, (11, 11), 0)
    lastFrame2 = cv2.cvtColor(lastFrame2, cv2.COLOR_BGR2GRAY)
    currentFrame = cv2.GaussianBlur(currentFrame, (11, 11), 0)
    currentFrame = cv2.cvtColor(currentFrame, cv2.COLOR_BGR2GRAY)

    img_compensate1, _, _, _, _, _ = motion_compensate(lastFrame1, lastFrame2)
    frameDiff1 = cv2.absdiff(lastFrame2, img_compensate1)

    img_compensate2, _, _, _, _, _ = motion_compensate(currentFrame, lastFrame2)
    frameDiff2 = cv2.absdiff(lastFrame2, img_compensate2)

    frameDiff = (frameDiff1.astype(np.float32) + frameDiff2.astype(np.float32)) / 2
    frameDiff = np.clip(frameDiff, 0, 255).astype(np.uint8)
    return frameDiff
# ========= 以上函数完全不变 =========


def process_video_to_file(input_video_path, output_video_path, brightness_gain=3.0):
    """主处理：带亮度增强"""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f'错误：无法打开视频文件 {input_video_path}')
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height), isColor=False)

    lastFrame1 = lastFrame2 = None
    frame_count = 0
    print('开始处理视频...')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        currentFrame = frame
        frame_count += 1

        if lastFrame1 is None:
            lastFrame1 = currentFrame
            continue
        elif lastFrame2 is None:
            lastFrame2 = currentFrame
            continue

        motion_diff = generate_motion_difference(lastFrame1, lastFrame2, currentFrame)
        motion_diff = cv2.resize(motion_diff, (width, height))

        # ===== 亮度增强（方案 A：线性增益） =====
        motion_diff = np.clip(motion_diff.astype(np.float32) * brightness_gain,
                              0, 255).astype(np.uint8)
        # =======================================

        out.write(motion_diff)
        if frame_count % 100 == 0:
            print(f'已处理 {frame_count} 帧')

        lastFrame1, lastFrame2 = lastFrame2, currentFrame

    cap.release()
    out.release()
    print(f'处理完成！运动差异视频已保存到: {output_video_path}')


def main():
    parser = argparse.ArgumentParser(description='生成更亮的运动差异视频')
    parser.add_argument('-i', '--input',  required=True, help='输入视频路径')
    parser.add_argument('-o', '--output', required=True, help='输出 mp4 路径')
    parser.add_argument('-g', '--gain', type=float, default=3.0,
                        help='亮度增益系数，默认 3.0（越大越亮）')
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f'错误：输入视频文件不存在: {args.input}')
        return

    process_video_to_file(args.input, args.output, args.gain)


if __name__ == '__main__':
    main()
