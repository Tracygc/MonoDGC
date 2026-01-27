import os
import cv2
import numpy as np
import kitti_object
import kitti_util
from tqdm import tqdm

ROOT_DIR = r'G:\datasets\KITTI\KITTI\data\KITTI'
PRED_DIR = r"C:\Users\ADMIN\Desktop\val"
OUTPUT_DIR = 'output_vis_rotated'
SPLIT_FILE = r"G:\datasets\KITTI\KITTI\data\KITTI\ImageSets\val.txt"


def draw_bev_map_rotated(pc_rect, objects_gt, objects_pred,
                         width=1242, height_limit=80, side_limit=20):
    scale = width / height_limit
    img_height = int((side_limit * 2) * scale)
    bev_map = np.zeros((img_height, width, 3), dtype=np.uint8)
    if pc_rect is not None:
        mask = (pc_rect[:, 2] > 0) & (pc_rect[:, 2] < height_limit) & \
               (pc_rect[:, 0] > -side_limit) & (pc_rect[:, 0] < side_limit)
        points = pc_rect[mask]

        u = points[:, 2] * scale

        v = (points[:, 0] + side_limit) * scale

        u = np.clip(u, 0, width - 1).astype(np.int32)
        v = np.clip(v, 0, img_height - 1).astype(np.int32)

        bev_map[v, u] = (255, 255, 255)

    def draw_boxes(objects, color):
        for obj in objects:
            if obj.type == 'DontCare': continue
            _, corners_3d = kitti_util.compute_box_3d(obj, np.eye(4))

            if corners_3d is None: continue
            bev_corners = corners_3d[:4, [0, 2]]
            poly_pts = []
            for i in range(4):
                cx, cz = bev_corners[i]

                # Apply same rotated mapping
                u_pt = int(cz * scale)  # Z -> U
                v_pt = int((cx + side_limit) * scale)  # X -> V

                poly_pts.append([u_pt, v_pt])

            pts = np.array(poly_pts, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(bev_map, [pts], True, color, 2)

    # Draw GT (Green)
    if objects_gt:
        draw_boxes(objects_gt, (0, 255, 0))

    # Draw Pred (Red)
    if objects_pred:
        draw_boxes(objects_pred, (0, 0, 255))

    return bev_map


def draw_camera_view(img, calib, objects_gt, objects_pred):
    """ Draw 3D boxes projected onto 2D camera image """
    img_draw = img.copy()

    def draw_obj(objects, color):
        for obj in objects:
            if obj.type == 'DontCare': continue
            corners_2d, _ = kitti_util.compute_box_3d(obj, calib.P)
            if corners_2d is not None:
                kitti_util.draw_projected_box3d(img_draw, corners_2d, color=color, thickness=2)

    if objects_gt:
        draw_obj(objects_gt, (0, 255, 0))
    if objects_pred:
        draw_obj(objects_pred, (0, 0, 255))

    return img_draw


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    dataset = kitti_object.kitti_object(ROOT_DIR, split='training')

    if os.path.exists(SPLIT_FILE):
        val_indices = [int(x.strip()) for x in open(SPLIT_FILE).readlines()]
    else:
        files = os.listdir(os.path.join(ROOT_DIR, 'training/label_2'))
        val_indices = [int(f.split('.')[0]) for f in files]

    val_indices.sort()

    print(f"Start processing {len(val_indices)} images (Rotated BEV)...")

    for idx in tqdm(val_indices):
        try:
            img = dataset.get_image(idx)
            lidar = dataset.get_lidar(idx)
            calib = dataset.get_calibration(idx)
            objects_gt = dataset.get_label_objects(idx)

            pred_filename = os.path.join(PRED_DIR, '%06d.txt' % idx)
            objects_pred = []
            if os.path.exists(pred_filename):
                objects_pred = kitti_util.read_label(pred_filename)


            img_vis = draw_camera_view(img, calib, objects_gt, objects_pred)

            pts_rect = calib.project_velo_to_rect(lidar[:, :3])

            bev_vis = draw_bev_map_rotated(pts_rect, objects_gt, objects_pred,
                                           width=img_vis.shape[1],
                                           height_limit=80,
                                           side_limit=20)

            separator = np.ones((5, img_vis.shape[1], 3), dtype=np.uint8) * 255

            final_vis = np.vstack([img_vis, separator, bev_vis])

            cv2.imwrite(os.path.join(OUTPUT_DIR, '%06d_vis.png' % idx), final_vis)

        except Exception as e:
            print(f"Error processing index {idx}: {e}")


if __name__ == '__main__':
    main()