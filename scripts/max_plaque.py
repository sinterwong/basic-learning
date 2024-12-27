import cv2
import numpy as np
import os
import os.path as osp
import time
import glob


def find_plaque_max_thickness_and_length(mask, distance_threshold=8, angle_weight=0.5, vis_output_dir=""):

    plaque_mask = (mask == 3).astype(np.uint8)
    lumen_mask = np.where(np.isin(mask, [1, 2, 4]), 1, 0).astype(np.uint8)

    # remove noise
    kernel = np.ones((3, 3), np.uint8)
    lumen_mask = cv2.morphologyEx(
        lumen_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    plaque_contours, _ = cv2.findContours(
        plaque_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lumen_contours, _ = cv2.findContours(
        lumen_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(plaque_contours) == 0 or len(lumen_contours) == 0:
        return 0, (), ()

    plaque_contour = max(plaque_contours, key=cv2.contourArea).squeeze()
    lumen_contour = max(lumen_contours, key=cv2.contourArea).squeeze()

    plaque_indices = np.arange(len(plaque_contour))
    prev_indices = (plaque_indices - 5) % len(plaque_contour)
    next_indices = (plaque_indices + 5) % len(plaque_contour)

    # (N, 2)
    prev_points = plaque_contour[prev_indices]
    next_points = plaque_contour[next_indices]
    tangent_vectors = next_points - prev_points
    normal_vectors = np.stack(
        [-tangent_vectors[:, 1], tangent_vectors[:, 0]], axis=1)

    # (N, 1, 2)
    plaque_points = plaque_contour[:, np.newaxis, :]

    # (1, M, 2)
    lumen_points = lumen_contour[np.newaxis, :, :]

    # (N, M, 2)
    diff_vectors = plaque_points - lumen_points

    # (N, M)
    distances = np.linalg.norm(diff_vectors, axis=2)

    # 如果两个contour的最近点的距离相距较远, 则认为不是相邻的
    global_min_distance = np.min(distances)
    if global_min_distance < distance_threshold:
        return 0, (), ()

    # (N, 1, 2)
    normal_vectors = normal_vectors[:, np.newaxis, :]

    normal_norms = np.linalg.norm(normal_vectors, axis=2, keepdims=True)
    diff_norms = np.linalg.norm(diff_vectors, axis=2, keepdims=True)

    normal_vectors_normalized = np.divide(
        normal_vectors, normal_norms, where=normal_norms != 0)
    diff_vectors_normalized = np.divide(
        diff_vectors, diff_norms, where=diff_norms != 0)

    dot_products = np.sum(normal_vectors_normalized *
                          diff_vectors_normalized, axis=2)
    angles = np.arccos(np.clip(dot_products, -1, 1))
    # min_angles = np.min(angles, axis=1)

    angle_threshold = np.pi / 4
    angle_scores = (angles / np.pi) * angle_weight * np.median(distances)
    scores = distances + angle_scores
    score_mask = (distances < distance_threshold) & (angles > angle_threshold)
    scores[score_mask] = np.inf

    best_matches = np.argmin(scores, axis=1)
    # min_scores = np.min(scores, axis=1)

    max_thickness_idx = np.argmax(
        distances[np.arange(len(plaque_contour)), best_matches])
    max_thickness = distances[max_thickness_idx,
                              best_matches[max_thickness_idx]]
    # final_angle = angles[max_thickness_idx, best_matches[max_thickness_idx]]
    max_thickness_points = (
        plaque_contour[max_thickness_idx], lumen_contour[best_matches[max_thickness_idx]])

    # TODO: 最大长度为最大厚度的法方向在plaque_contour内部的线段
    midpoint = (max_thickness_points[0] + max_thickness_points[1]) / 2
    thickness_vector = max_thickness_points[1] - max_thickness_points[0]

    perpendicular_vector = np.array(
        [-thickness_vector[1], thickness_vector[0]])
    perpendicular_vector = perpendicular_vector / \
        np.linalg.norm(perpendicular_vector)

    extension = np.max(np.ptp(plaque_contour, axis=0)) * 2
    point1 = midpoint + perpendicular_vector * extension
    point2 = midpoint - perpendicular_vector * extension

    intersect_points = []
    for i in range(len(plaque_contour)):
        p1 = plaque_contour[i]
        p2 = plaque_contour[(i + 1) % len(plaque_contour)]

        x1, y1 = point1
        x2, y2 = point2
        x3, y3 = p1
        x4, y4 = p2

        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        # 平行或重合，不相交
        if denominator == 0:
            continue

        ua = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        ub = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denominator

        # 相交在p1-p2上
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            intersect_points.append(np.array([x, y]))

    if len(intersect_points) < 2:
        raise Exception("No intersect points found")
    else:
        max_length = 0
        max_length_points = None
        intersect_points = np.array(intersect_points, dtype=np.int32)

        for i in range(len(intersect_points)):
            for j in range(i + 1, len(intersect_points)):
                dist = np.linalg.norm(
                    intersect_points[i] - intersect_points[j])
                if dist > max_length:
                    max_length = dist
                    max_length_points = (
                        intersect_points[i], intersect_points[j])

    if vis_output_dir:
        if (not os.path.exists(vis_output_dir)):
            os.makedirs(vis_output_dir)

        cv2.imwrite(os.path.join(vis_output_dir,
                    "plaque_mask.png"), plaque_mask * 255)
        cv2.imwrite(os.path.join(vis_output_dir,
                    "lumen_mask.png"), lumen_mask * 255)

        img_with_contours = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        for p in plaque_contour:
            cv2.circle(img_with_contours, tuple(p), 2, (0, 0, 255), -1)
        for l in lumen_contour:
            cv2.circle(img_with_contours, tuple(l), 2, (0, 255, 0), -1)
        cv2.imwrite(os.path.join(vis_output_dir,
                    "contours.png"), img_with_contours)

        img_match_points = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img_normal_vectors = img_with_contours.copy()
        for i, (p_point, l_idx) in enumerate(zip(plaque_contour, best_matches)):
            l_point = lumen_contour[l_idx]
            cv2.circle(img_match_points, tuple(p_point), 2, (255, 0, 0), -1)
            cv2.circle(img_match_points, tuple(l_point), 2, (0, 255, 0), -1)
            cv2.arrowedLine(img_match_points, tuple(l_point),
                            tuple(p_point), (0, 0, 255), 1)

            normal_vector = normal_vectors[i, 0]
            normal_vector_scaled = normal_vector / \
                np.linalg.norm(normal_vector) * 20
            end_point = (p_point + normal_vector_scaled).astype(int)
            cv2.arrowedLine(img_normal_vectors, tuple(p_point),
                            tuple(end_point), (255, 255, 0), 1)

        cv2.imwrite(os.path.join(vis_output_dir,
                    "match_points.png"), img_match_points)
        cv2.imwrite(os.path.join(vis_output_dir, "normal_vectors.png"),
                    img_normal_vectors)

        img_final = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_final, [plaque_contour], -1, (0, 0, 255), 2)
        cv2.drawContours(img_final, [lumen_contour], -1, (0, 255, 0), 2)
        cv2.circle(img_final, tuple(
            max_thickness_points[0]), 5, (255, 0, 0), -1)
        cv2.line(img_final, tuple(max_thickness_points[0]),
                 tuple(max_thickness_points[1]), (255, 0, 0), 2)
        cv2.putText(img_final, f"Max Thickness: {max_thickness:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.line(img_final, tuple(max_length_points[0]), tuple(
            max_length_points[1]), (0, 255, 255), 2)
        cv2.putText(img_final, f"Max Length: {max_length:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imwrite(os.path.join(vis_output_dir,
                    "final_result.png"), img_final)
    return max_thickness, max_thickness_points, max_length_points


if __name__ == "__main__":
    mask_root = "/home/sinter/workspace/basic-learning/install/data/carotis_postproc/mask"
    output_root = "/home/sinter/workspace/basic-learning/install/data/carotis_postproc/output"

    mask_paths = glob.glob(os.path.join(mask_root, "*.png"))

    angle_weight = 0.4
    distance_thre = 5

    # for p in mask_paths:
    #     mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    #     # ret = find_plaque_max_thickness(mask)
    #     ret = find_plaque_max_thickness_and_length(mask, distance_thre, angle_weight, os.path.join(
    #         output_root, os.path.basename(p).split(".")[0]))

    #     print(f"Max tickness: {ret[0]}")
    #     print(f"Max tickness point: {ret[1]}")
    #     print(f"Max length: {ret[2]}")

    mask_path = None
    for p in mask_paths:
        if osp.basename(p).split(".")[0] == "H42-G20240522-2-C_471":
            mask_path = p
            break
    if mask_path is not None:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        ret = find_plaque_max_thickness_and_length(
            mask, distance_thre, angle_weight, osp.join(output_root, osp.basename(mask_path).split(".")[0]))
        print(f"Max tickness: {ret[0]}")
        print(f"Max tickness point: {ret[1]}")
    else:
        print("Mask not found")
