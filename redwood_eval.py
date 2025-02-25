import os
import json
import numpy as np
from PIL import Image
import open3d as o3d
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
from typing import Tuple
from config.config import *
from absl import app
import _pickle as cPickle
FLAGS = flags.FLAGS
import torch
from network.SE3Pose import SE3Pose
from tools.geom_utils import generate_RT, generate_sRT
from date_preprocess.utils import sample_points_from_mesh
from evaluation.eval_utils import *


def draw(img, img_pts, color):
    img_pts = np.int32(img_pts).reshape(-1, 2)
    # draw ground layer in darker color
    color_ground = (int(color[0] * 0.3), int(color[1] * 0.3), int(color[2] * 0.3))
    for i, j in zip([4, 5, 6, 7], [5, 7, 4, 6]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_ground, 1)
    # draw pillars in minor darker color
    color_pillar = (int(color[0] * 0.6), int(color[1] * 0.6), int(color[2] * 0.6))
    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color_pillar, 1)
    # draw top layer in original color
    for i, j in zip([0, 1, 2, 3], [1, 3, 0, 2]):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, 1)

    return img

class Camera:
    """Pinhole camera parameters.

    This class allows conversion between different pixel conventions, i.e., pixel
    center at (0.5, 0.5) (as common in computer graphics), and (0, 0) as common in
    computer vision.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        s: float = 0.0,
        pixel_center: float = 0.0,
    ) -> None:
        """Initialize camera parameters.

        Note that the principal point is only fully defined in combination with
        pixel_center.

        The pixel_center defines the relation between continuous image plane
        coordinates and discrete pixel coordinates.

        A discrete image coordinate (x, y) will correspond to the continuous
        image coordinate (x + pixel_center, y + pixel_center). Normally pixel_center
        will be either 0 or 0.5. During calibration it depends on the convention
        the point features used to compute the calibration matrix.

        Note that if pixel_center == 0, the corresponding continuous coordinate
        interval for a pixel are [x-0.5, x+0.5). I.e., proper rounding has to be done
        to convert from continuous coordinate to the corresponding discrete coordinate.

        For pixel_center == 0.5, the corresponding continuous coordinate interval for a
        pixel are [x, x+1). I.e., floor is sufficient to convert from continuous
        coordinate to the corresponding discrete coordinate.

        Args:
            width: Number of pixels in horizontal direction.
            height: Number of pixels in vertical direction.
            fx: Horizontal focal length.
            fy: Vertical focal length.
            cx: Principal point x-coordinate.
            cy: Principal point y-coordinate.
            s: Skew.
            pixel_center: The center offset for the provided principal point.
        """
        # focal length
        self.fx = fx
        self.fy = fy

        # principal point
        self.cx = cx
        self.cy = cy

        self.pixel_center = pixel_center

        # skew
        self.s = s

        # image dimensions
        self.width = width
        self.height = height

    def get_o3d_pinhole_camera_parameters(self) -> o3d.camera.PinholeCameraParameters():
        """Convert camera to Open3D pinhole camera parameters.

        Open3D camera is at (0,0,0) looking along positive z axis (i.e., positive z
        values are in front of camera). Open3D expects camera with pixel_center = 0
        and does not support skew.

        Returns:
            The pinhole camera parameters.
        """
        fx, fy, cx, cy, _ = self.get_pinhole_camera_parameters(0)
        params = o3d.camera.PinholeCameraParameters()
        params.intrinsic.set_intrinsics(self.width, self.height, fx, fy, cx, cy)
        params.extrinsic = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        return params

    def get_pinhole_camera_parameters(self, pixel_center: float)-> Tuple:
        """Convert camera to general camera parameters.

        Args:
            pixel_center:
                At which ratio of a square the pixel center should be for the resulting
                parameters. Typically 0 or 0.5. See class documentation for more info.

        Returns:
            - fx, fy: The horizontal and vertical focal length
            - cx, cy:
                The position of the principal point in continuous image plane
                coordinates considering the provided pixel center and the pixel center
                specified during the construction.
            - s: The skew.
        """
        cx_corrected = self.cx - self.pixel_center + pixel_center
        cy_corrected = self.cy - self.pixel_center + pixel_center
        return self.fx, self.fy, cx_corrected, cy_corrected, self.s


def draw_depth_geometry(
    posed_mesh: o3d.geometry.TriangleMesh, camera: Camera
) -> np.ndarray:
    """Render a posed mesh given a camera looking along z axis (OpenCV convention)."""
    # see http://www.open3d.org/docs/latest/tutorial/visualization/customized_visualization.html

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=camera.width, height=camera.height, visible=False)

    # Add mesh in correct position
    vis.add_geometry(posed_mesh, True)

    options = vis.get_render_option()
    options.mesh_show_back_face = True

    # Set camera at fixed position (i.e., at 0,0,0, looking along z axis)
    view_control = vis.get_view_control()
    o3d_cam = camera.get_o3d_pinhole_camera_parameters()
    o3d_cam.extrinsic = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )

    view_control.convert_from_pinhole_camera_parameters(o3d_cam)

    # Generate the depth image
    vis.poll_events()
    vis.update_renderer()
    depth = np.asarray(vis.capture_depth_float_buffer())

    return depth



torch.autograd.set_detect_anomaly(True)
device = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES']='0'

def redwood(argv):

    ann_json = os.path.join('/media/ubuntu/data/6DoF/Datasets/REEDWOOD75/redwood75/redwood75/annotations.json')

    with open(ann_json, "r") as f:
        anns_dict = json.load(f)
    raw_samples = []

    category_str_to_id = {
        "bottle": np.array([0.]),
        "bowl": np.array([1.]),
        "mug": np.array([5.]),
    }

    cate_sym = {
        "bottle": np.array([[1., 1., 0., 1.]]),
        "bowl": np.array([[1., 1., 0., 1.]]),
        "mug": np.array([[1., 0., 0., 0.]]),
    }
    cate_mean_shape = {
        "bottle": np.array([[0.0870, 0.2200, 0.0890]]),
        "bowl": np.array([[0.1650, 0.0800, 0.1650]]),
        "mug": np.array([[0.1460, 0.0830, 0.1140]]),
    }


    id = 0

    for seq_id, seq_anns in anns_dict.items():


        mesh_filename = seq_anns["mesh"]
        mesh_path = os.path.join('/media/ubuntu/data/6DoF/Datasets/REEDWOOD75/redwood75/redwood75', mesh_filename)
        cate = seq_anns['category']
        obj_points = sample_points_from_mesh(mesh_path, 1024, fps=True, ratio=3)


        for pose_ann in seq_anns["pose_anns"]:

            cate_id = category_str_to_id[cate]
            sym = cate_sym[cate]
            mean_shape = cate_mean_shape[cate]

            position = pose_ann["position"]
            orientation_q = pose_ann["orientation"]
            rgb_filename = pose_ann["rgb_file"]
            depth_filename = pose_ann["depth_file"]
            size = seq_anns["scale"]
            size = np.asarray(size) * 2

            color_path = os.path.join(
                '/media/ubuntu/data/6DoF/Datasets/REEDWOOD75', seq_id, "rgb", rgb_filename
            )
            depth_path = os.path.join(
                '/media/ubuntu/data/6DoF/Datasets/REEDWOOD75', seq_id, "depth", depth_filename
            )

            depth = np.asarray(Image.open(depth_path)) * 0.001

            posed_mesh = o3d.io.read_triangle_mesh(mesh_path)
            R = Rotation.from_quat(orientation_q).as_matrix()

            posed_mesh.rotate(R)
            posed_mesh.translate(position)

            posed_mesh.compute_vertex_normals()



            cam = Camera(width=640, height=480, fx=525, fy=525, cx=319.5, cy=239.5)
            gt_depth = draw_depth_geometry(posed_mesh, cam)


            mask = gt_depth != 0
            # exclude occluded parts from mask
            mask[(depth != 0) * (depth < gt_depth - 0.01)] = 0


            x = np.linspace(0, 640 - 1, 640, dtype=np.float32)
            y = np.linspace(0, 480 - 1, 480, dtype=np.float32)
            xy = np.asarray(np.meshgrid(x, y))

            cx, cy, fx, fy = 319.5, 239.5, 525, 525
            depth = gt_depth.reshape(-1).astype(np.float)
            valid = depth > 0
            depth = depth[valid]
            x_map = xy[0].reshape(-1)[valid]
            y_map = xy[1].reshape(-1)[valid]
            real_x = (x_map - cx) * depth / fx
            real_y = (y_map - cy) * depth / fy
            pcl = np.stack((real_x, real_y, depth), axis=-1)
            total_pts_num = pcl.shape[0]
            ids = np.random.permutation(total_pts_num)[:1028]
            pcl = pcl[ids]

            rotation_o2n = np.zeros((3, 3))
            rotation_o2n[1, 1] = -1
            rotation_o2n[0, 0] = -1
            rotation_o2n[:, 2] = 1 - np.abs(np.sum(rotation_o2n, 1))  # rows must sum to +-1
            rotation_o2n[:, 2] *= np.linalg.det(rotation_o2n)  # make special orthogonal
            if np.linalg.det(rotation_o2n) != 1.0:  # check if special orthogonal
                raise ValueError("Unsupported combination of remap_{y,x}_axis. det != 1")


            RT = np.eye(4)
            RT[:3, :3] = R @ rotation_o2n
            RT[:3, 3] = position

            data_redwood = {}
            data_redwood['pcl'] = pcl
            data_redwood['model_pcl'] = obj_points
            data_redwood['gt_RT'] = RT
            data_redwood['sym'] = sym
            data_redwood['size'] = size
            data_redwood['mean_shape'] = mean_shape
            data_redwood['cat_id'] = cate_id
            data_redwood['rgb'] = color_path

            save_path = os.path.join('/media/ubuntu/data/6DoF/Datasets/REEDWOOD75', str(id))
            with open(save_path + '_label.pkl', 'wb') as f:
                cPickle.dump(data_redwood, f)
            id += 1

            # x = size[0]/2
            # y = size[1]/2
            # z = size[2]/2
            # box_points = np.array([[-x, y, z],
            #                       [x, y, z],
            #                       [x, y, -z],
            #                       [-x, y, -z],
            #                       [-x, -y, z],
            #                       [x, -y, z],
            #                       [x, -y, -z],
            #                       [-x, -y, -z]])
            # lines_box = np.array([[0, 1], [1, 2], [2, 3], [0, 3],
            #                       [0, 4], [1, 5], [2, 6], [3, 7],
            #                       [4, 5], [5, 6], [6, 7], [4, 7]])

            # mean_shape = torch.as_tensor(mean_shape.astype(np.float32)).contiguous().to(device)
            # sym = torch.as_tensor(sym.astype(np.float32)).contiguous().to(device)
            #
            # output_dict \
            #     = network(PC=torch.as_tensor(pcl.astype(np.float32)).contiguous().to(device).unsqueeze(0),
            #               obj_id=torch.as_tensor(cate_id.astype(np.float32)).contiguous().to(device).unsqueeze(0),
            #               mean_shape= mean_shape,
            #               sym= sym)
            #
            #
            # p_green_R_vec = output_dict['p_green_R'].detach()
            # p_red_R_vec = output_dict['p_red_R'].detach()
            # p_T = output_dict['Pred_T'].detach()
            # p_s = output_dict['Pred_s'].detach()
            # f_green_R = output_dict['f_green_R'].detach()
            # f_red_R = output_dict['f_red_R'].detach()
            #
            # from tools.training_utils import get_gt_v
            # pred_s = torch.abs(p_s + mean_shape.to(device))
            # pred_RT = generate_RT([p_green_R_vec, p_red_R_vec], [f_green_R, f_red_R], p_T, mode='vec', sym=sym.to(device))
            # pred_RT = pred_RT[0].cpu().numpy()

            # points = obj_points
            #
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            # box = o3d.geometry.LineSet()
            # box.lines = o3d.utility.Vector2iVector(lines_box)
            # box.points = o3d.utility.Vector3dVector(box_points)
            # box = box.transform(RT)
            #
            # # axis_pcb = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0,0,0]).transform(pred_RT)
            #
            # axis_gt = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0]).transform(RT)
            #
            # o3d.visualization.draw_geometries([pcd] + [box] + [axis_gt])


        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcl)
        # axis_pcb = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0]).transform(RT)
        # o3d.visualization.draw_geometries([pcd] + [axis_pcb])


def evaluation_mine(arg):

    Train_stage = 'PoseNet_only'
    FLAGS.train = False
    network = SE3Pose(Train_stage)
    network = network.to(device)

    network = network.to(device)

    state_dict = torch.load('./checkpoints/model_149.pth')
    network.load_state_dict(state_dict)
    network = network.eval()
    intrinsics = np.array([[525, 0, 319.5], [0, 525, 239.5], [0, 0, 1]], dtype=np.float)

    for idx in range(0, 75):
        data_path = os.path.join('/media/ubuntu/data/6DoF/Datasets/REEDWOOD75', str(idx))

        with open(data_path + '_label.pkl', 'rb') as f:
            data = cPickle.load(f)


        mean_shape = torch.as_tensor(data['mean_shape'].astype(np.float32)).contiguous().to(device)
        sym = torch.as_tensor(data['sym'].astype(np.float32)).contiguous().to(device)

        output_dict \
            = network(PC=torch.as_tensor(data['pcl'].astype(np.float32)).contiguous().to(device).unsqueeze(0),
                      obj_id=torch.as_tensor(data['cat_id'].astype(np.float32)).contiguous().to(device).unsqueeze(0),
                      mean_shape=mean_shape,
                      sym=sym)


        p_green_R_vec = output_dict['p_green_R'].detach()
        p_red_R_vec = output_dict['p_red_R'].detach()
        p_T = output_dict['Pred_T'].detach()
        p_s = output_dict['Pred_s'].detach()
        f_green_R = output_dict['f_green_R'].detach()
        f_red_R = output_dict['f_red_R'].detach()

        pred_s = torch.abs(p_s + mean_shape.to(device))
        pred_RT = generate_RT([p_green_R_vec, p_red_R_vec], [f_green_R, f_red_R], p_T, mode='vec', sym=sym.to(device))
        pred_RT = pred_RT[0].cpu().numpy()

        R1 = pred_RT[:3, :3] / np.cbrt(np.linalg.det(pred_RT[:3, :3]))
        R2 = data['gt_RT'][:3, :3] / np.cbrt(np.linalg.det(data['gt_RT'][:3, :3]))
        T1 = pred_RT[:3, 3]
        T2 = data['gt_RT'][:3, 3]

        if data['cat_id'] == 5:
            R = R1 @ R2.transpose()
            theta = np.arccos((np.trace(R) - 1) / 2)
            gt_RT = data['gt_RT']

        else:
            y = np.array([0, 1, 0])
            y1 = R1 @ y
            y2 = R2 @ y
            theta = np.arccos(
                y1.dot(y2) / (np.linalg.norm(y1) * np.linalg.norm(y2)))

            pred_RT = align_rotation(pred_RT)
            gt_RT =  align_rotation(data['gt_RT'])


        theta *= 180 / np.pi
        shift = np.linalg.norm(T1 - T2) * 100
        print(theta)

        img = cv2.imread(data['rgb'])

        gt_box = get_3d_bbox(data['size'], 0)
        transformed_bbox_3d = transform_coordinates_3d(gt_box, gt_RT)
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
        img = draw(img, projected_bbox, (255, 255, 255))



        bbox_3d = get_3d_bbox(pred_s[0, :], 0)
        transformed_bbox_3d = transform_coordinates_3d(bbox_3d, pred_RT)
        projected_bbox = calculate_2d_projections(transformed_bbox_3d, intrinsics)
        img = draw(img, projected_bbox, (0, 255, 0))


        image_path = '/media/ubuntu/data/6DoF/Datasets/REEDWOOD75/Ours/' + str(idx) +'.png'
        cv2.imwrite(image_path, img)




        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(data['pcl'])
        # axis_pcb = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0]).transform(data['gt_RT'])
        # #o3d.visualization.draw_geometries([pcd] + [axis_pcb])
        # image_path = '/media/ubuntu/data/6DoF/Datasets/REEDWOOD75/gts/' + str(idx) +'_gts.png'
        # vis = o3d.visualization.Visualizer()
        # vis.create_window()
        # vis.add_geometry(pcd)
        # vis.update_geometry(pcd)
        # vis.add_geometry(axis_pcb)
        # vis.update_geometry(axis_pcb)
        # vis.poll_events()
        # vis.capture_screen_image(image_path)
        # vis.destroy_window()

        # with open(os.path.join('/media/ubuntu/data/6DoF/Datasets/REEDWOOD75', 'myresult.txt'), 'a') as f:
        #     f.write("%s %s rot_error: %s trans_error: %s\n" % (str(idx), str(data['cat_id'][0]), theta, shift))



if __name__ == "__main__":

    # app.run(redwood)
    app.run(evaluation_mine)
