import cv2
import numpy as np
import cv2.gapi
import torch
import PIL.Image
import hloc.extract_features, hloc.match_features ## these are files
import viz2d
import datetime
import os 
import h5py

import open3d as o3d
from types import SimpleNamespace
from rendering import Renderer
from pathlib import Path


## to download the models correctly
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


## fixed conf 
retrieval_conf = hloc.extract_features.confs["netvlad"] ## global features
# feature_conf = hloc.extract_features.confs["superpoint_inloc"] ## local features
feature_conf = {
        "output": "feats-superpoint-n1024-r1024",
        "model": {
            "name": "superpoint",
            "nms_radius": 3,
            "max_keypoints": 1024,
        },
        "preprocessing": {
            "grayscale": True,
            "resize_max": 800, # todo: investigate bug with resizing images
        },
    }
matcher_conf = hloc.match_features.confs["superpoint+lightglue"] ## local feature matching 

default_conf = {
    "globs": ["*.jpg", "*.png", "*.jpeg", "*.JPG", "*.PNG"],
    "grayscale": False,
    "resize_max": None,
    "resize_force": False,
    "interpolation": "cv2_area",  # pil_linear is more accurate but slower
}


loaded_model_cache = {}
def load_model(conf: dict, device: torch.device, mode):
    global loaded_model_cache

    key = str((conf,device))
    if key in loaded_model_cache:
        return loaded_model_cache[key]

    Model = hloc.utils.base_model.dynamic_load(mode, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)
    loaded_model_cache[key] = model
    return model

def read_image(path, grayscale=False):
    if grayscale:
        mode = cv2.IMREAD_GRAYSCALE
    else:
        mode = cv2.IMREAD_COLOR
    image = cv2.imread(path, mode)
    if image is None:
        raise ValueError(f"Cannot read image {path}.")
    if not grayscale and len(image.shape) == 3:
        image = image[:, :, ::-1]  # BGR to RGB
    return image

def resize_image(image, size, interp):
    if interp.startswith("cv2_"):
        interp = getattr(cv2, "INTER_" + interp[len("cv2_") :].upper())
        h, w = image.shape[:2]
        if interp == cv2.INTER_AREA and (w < size[0] or h < size[1]):
            interp = cv2.INTER_LINEAR
        resized = cv2.resize(image, size, interpolation=interp)
    elif interp.startswith("pil_"):
        interp = getattr(PIL.Image, interp[len("pil_") :].upper())
        resized = PIL.Image.fromarray(image.astype(np.uint8))
        resized = resized.resize(size, resample=interp)
        resized = np.asarray(resized, dtype=image.dtype)
    else:
        raise ValueError(f"Unknown interpolation {interp}.")
    return resized

def normalise_image(conf, image_path):
    conf = SimpleNamespace(**{**default_conf, **conf})

    image = read_image(image_path, conf.grayscale)
    image = image.astype(np.float32)
    size = image.shape[:2][::-1]
    # print("size0=", size)

    if conf.resize_max and (
        conf.resize_force or max(size) > conf.resize_max
    ):
        scale = conf.resize_max / max(size)
        size_new = tuple(int(round(x * scale)) for x in size)
        image = resize_image(image, size_new, conf.interpolation)

    if conf.grayscale:
        image = image[None]
    else:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    image = image / 255.0

    data = {
        "image": image,
        "original_size": np.array(size),
    }
    return data


## works for both global and local features. 
## use differnt models. 
def extract_features(conf, device, image_path, extractor):
    data = normalise_image(conf["preprocessing"], image_path)

    image_tensor = torch.tensor(data["image"], device=device, requires_grad=False).unsqueeze(0)

    # extractor = load_model(conf, device, hloc.extractors)
    pred = extractor({"image": image_tensor})
    pred = {k: v[0] for k, v in pred.items()}
    pred["image_size"] = original_size = data["original_size"]

    if "keypoints" in pred:
        size = np.array(data["image"].shape[-2:][::-1])
        scales = (original_size / size).astype(np.float32)

        pred["keypoints"] = (pred["keypoints"] + 0.5) * scales[None] - 0.5
        if "scales" in pred:
            pred["scales"] *= scales.mean()

    return pred

def visualise_extracted_features(image_path, features):
    image = cv2.imread(image_path)
    keypoints1 = [cv2.KeyPoint(x=int(k[0]), y=int(k[1]), size=1) for k in features["keypoints"]]
    image1_with_kp = cv2.drawKeypoints(image, keypoints1, None, color=(0, 255, 0), flags=0)
    cv2.imshow('Image 1 with keypoints', image1_with_kp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def pack_match_inputs(img0, img1, features1, features2):
    print(f"img0.shape={img0.shape}")
    print(f"img1.shape={img1.shape}")
    img0 = torch.from_numpy(img0).to(device)
    img0 = img0[None, ...]
    img1 = torch.from_numpy(img1).to(device)
    img1 = img1[None, ...]
    features1["keypoints"] = features1["keypoints"][None, ...]
    features1["descriptors"] = features1["descriptors"][None, ...]
    features2["keypoints"] = features2["keypoints"][None, ...]
    features2["descriptors"] = features2["descriptors"][None, ...]

    inputs = {'image0':img0, 'keypoints0':features1["keypoints"], 'descriptors0':features1["descriptors"], 'image1':img1, 'keypoints1':features2["keypoints"], 'descriptors1':features2["descriptors"]}
    return inputs

def match_features(conf, image_path1, image_path2, features1, features2):
    matcher = load_model(conf, device, hloc.matchers)
    img0 = cv2.imread(image_path1)
    img1 = cv2.imread(image_path2)
    inputs = pack_match_inputs(img0, img1, features1, features2)

    # pairs = matcher({"image0": features1, "image1": features2})
    pairs = matcher(inputs)

    feats0, feats1, pairs = [
        rbd(x) for x in [features1, features2, pairs]
    ]

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], pairs["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

    return kpts0, kpts1, m_kpts0, m_kpts1, pairs


def visualise_matches(image0_path, image1_path, kpts0, kpts1, m_kpts0, m_kpts1, matches01):
    image0 = cv2.imread(image0_path)
    image1 = cv2.imread(image1_path)
    axes = viz2d.plot_images([image0, image1])
    viz2d.plot_matches(m_kpts0, m_kpts1, color="lime", lw=0.2)
    viz2d.add_text(0, f'Stop after {matches01["stop"]} layers', fs=20)
    viz2d.save_plot("matches.png")

    kpc0, kpc1 = viz2d.cm_prune(matches01["prune0"]), viz2d.cm_prune(matches01["prune1"])
    viz2d.plot_images([image0, image1])
    viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
    viz2d.save_plot("prunes.png")


class Camera:
    def __init__(self, fx, fy, cx, cy):
        self.c = np.array([cx, cy])
        self.f = np.array([fx, fy])

    def get_intrinsic_matrix(self):
        fx, fy = self.f
        cx, cy = self.c
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def read_intrinsic_matrix(lst):
    fx, fy, cx, cy = lst
    return Camera(fx, fy, cx, cy)

def read_trajectory_to_matrix(lst):
    w, x, y, z, tx, ty, tz = lst

    R = np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])
    t = np.array([tx, ty, tz])
    return R, t

def read_global_aligment(lst):
    w, x, y, z, tx, ty, tz = lst
    return np.array([w, x, y, z]), np.array([tx, ty, tz])
    

def to_homogeneous(p: np.ndarray) -> np.ndarray:
    return np.pad(p, ((0, 0),)*(p.ndim-1) + ((0, 1),), constant_values=1)

def lift_points2D(c, f, pose, p2d, renderer):
    # camera = self.session.sensors[key[1]]
    # T_cam2w = self.session.get_pose(*key)
    R, t = pose 
    origins = np.tile(t.astype(np.float32)[None], (len(p2d), 1)) ## camera origin in world frame, t is row vector, stacked vertically for each feature point. 
    p2d_norm = image2world(p2d.astype(np.float32), c, f) ## img points z = 1x ## still in camera space. 
    # R = T_cam2w.R ## R transforms from camera to world.
    directions = to_homogeneous(p2d_norm.astype(np.float32)) @ R.astype(np.float32).T ## p2d: (N, 3), p2d * RT = (R * p2dT)T = (N, 3) ## packed by rows

    origins = np.ascontiguousarray(origins, dtype=np.float32)
    directions = np.ascontiguousarray(directions, dtype=np.float32)
    rays = (origins, directions)

    xyz, valid = renderer.compute_intersections(rays)
    return np.array(valid, bool), xyz

## c is offset, f is focal length should be (2,)
## camera is origin 
## ix, iy at z = f
## return cwx, cwy at z = 1
def image2world(pts: np.ndarray, c, f) -> np.ndarray:
    return (pts - c) / f


def read_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    return o3d.io.read_triangle_mesh(str(path))

def apply_transformation_to_mesh(q, t, mesh):
    R = o3d.geometry.get_rotation_matrix_from_quaternion(q)
    mesh.rotate(R, center=np.array([0, 0, 0]).reshape(3, 1))
    mesh.translate(t)


## get the 2d-3d correspondences of all images. 
def save_2d_3d(root_path: str):
    C = read_intrinsic_matrix([960, 960, 639.8245614035088, 959.8245614035088]) ## all 4 are the same. 

    with open(os.path.join(root_path, "trajectories.txt"), "r") as trajectory_file:
        trajectory_str = trajectory_file.readlines()[1:]
        trajectories = list(map(lambda x: [ float(i) for i in x.split(", ")[2:]],trajectory_str))
    
    # with open("../data/navvis_2022-02-06_12.55.11/global_alignment.txt", "r") as global_alignment_file:

    with open(os.path.join(root_path, "images.txt"), "r") as images_file:
        images = images_file.readlines()[1:]
        image_names = list(map(lambda x: x.strip().split(", ")[2], images))

    assert len(trajectories) == len(image_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    assert os.path.exists(os.path.join(root_path, "proc/meshes/mesh_simplified.ply"))
    mesh = read_mesh(os.path.join(root_path, "proc/meshes/mesh_simplified.ply"))

    renderer = Renderer(mesh)
    extractor = load_model(feature_conf, device, hloc.extractors)

    for i in range(len(trajectories)):
        # if i < 207*4+1:
        #     continue
        ## extract features
        print(f"{i}/{len(trajectories)}")
        image_path = os.path.join(root_path, f"raw_data/{image_names[i]}")
        name = image_names[i].split("/")[1]

 
        trajectory = trajectories[i]
        features = extract_features(feature_conf, device, image_path, extractor) ## keypoint, descriptor, scores, image_size

        # print(f"trajectory={trajectory}")

        ## 2d-3d correspondences
        p2d = features["keypoints"]
        # print(f"p2d={p2d}")
        ret, p3d = lift_points2D(C.c, C.f,read_trajectory_to_matrix(trajectory), p2d.numpy(), renderer)

        # print(f"p2d.shape={p3d.shape}")
        # print(f"ret.shape={ret.shape}")

        ## update features to store 
        # features["ret"] = ret
        features["p3d"] = p3d
        features["keypoints"] = p2d[ret]
        features["descriptors"] = features["descriptors"].T[ret].T.detach().cpu().numpy()
        features["scores"] = features["scores"][ret].detach().cpu().numpy()

        # for k, v in features.items():
        #     print(f"Saving {k} of type {v}")

        ## save features
        with h5py.File(os.path.join(root_path, "features.h5"), "a", libver="latest") as fd:
            try:
                if name in fd:
                    del fd[name]
                grp = fd.create_group(name)
                for k, v in features.items():
                    grp.create_dataset(k, data=v)
            except OSError as error:
                if "No space left on device" in error.args[0]:
                    print(
                        "Out of disk space: storing features on disk can take "
                        "significant space, did you enable the as_half flag?"
                    )
                    del grp, fd[name]
                raise error

if __name__ == "__main__":
    save_2d_3d("/home/ubuntu/localization_server/data/navvis_2022-02-06_12.55.11")