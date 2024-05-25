from pathlib import Path
from pprint import pformat
import hloc.extract_features, hloc.match_features, hloc.pairs_from_retrieval, hloc.extractors, hloc.matchers
from hloc.utils import base_model
from torch import nn
from torchvision import transforms
from typing import Tuple, List, Dict, Any
from multiprocessing import freeze_support
import open3d as o3d
import cv2
import h5py
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import logging
import time
import torch
from dataclasses import dataclass
import itertools
import glob
import os
import PIL
import gc
import json

@torch.jit.script
@dataclass
class RetrievalFeatures:
    features: torch.Tensor
    db_names: List[str]

@torch.jit.script
@dataclass
class LocalFeatures:
    keypoints: List[torch.Tensor]
    scores: List[torch.Tensor]
    descriptors: List[torch.Tensor]
    db_names: List[str]
    uncertainty: float = 0

@torch.jit.script
@dataclass
class ImageDataset:
    images: List[torch.Tensor]
    image_shapes: List[torch.Tensor] # torch.Tensor instead of Tuple[int,int,int] for torch.jit workaround
    db_names: List[str]

@torch.jit.script
@dataclass
class Camera:
    width: int 
    height: int
    K: torch.Tensor

@torch.jit.script
@dataclass 
class ImagePose:
    camera: int
    t: torch.Tensor
    R: torch.Tensor # Quaternion


class PretrainedModel(nn.Module):
    preprocess_conf : Dict[str,int]

    def __init__(self, module, conf, device):
        super().__init__()
        Model = base_model.dynamic_load(module, conf["model"]["name"])
        self.model = Model(conf["model"]).eval()
        #.to(device)
        #self.device = device

        if "preprocessing" in conf:
            self.preprocess_conf = {k: int(v) for k,v in conf["preprocessing"].items()}
        else:
            self.preprocess_conf = {}

    def preprocess(self, image):
        image = image.type(torch.float32)
        size = image.shape[:2][::-1]
        conf = self.preprocess_conf

        if conf.get("grayscale",0):
            image = image[None] if len(image.shape)==2 else image[:,:,0][None]
        else:
            image = image.transpose(1,2).transpose(0,1) # (2, 0, 1))  # HxWxC to CxHxW

        """ 
        todo: Resizing does not currently compile with torch.jit
        if conf.get("resize_max",0) and (conf.get("resize_force",0) or max(size) > conf["resize_max"]):
            scale = conf["resize_max"] / max(size)
            size_new = (int(size[0]*scale), int(size[1]*scale))
            image = transforms.functional.resize(image, size_new)
            print("Resize to (", image.shape[2], image.shape[1], ")")
        """

        image = image / 255.0
        return image

    def forward(self, input: Dict[str, torch.Tensor]):
        return self.model(input)

def to_batch(device: torch.device,x) -> torch.Tensor:
    ftype = torch.float32
    return x.unsqueeze(0).type(ftype).to(device)

def pairs_from_retrieval(device: torch.device, query_desc: RetrievalFeatures, db_desc : RetrievalFeatures, num_matched:int=5, min_score:int = -1):
    scores = torch.einsum("id,jd->ij", query_desc.features.to(device), db_desc.features.to(device))

    # Avoid self-matching
    invalid = torch.zeros((query_desc.features.shape[0],db_desc.features.shape[0],), dtype=torch.bool, device=device)
        #([[query_desc.db_names == other for other in db_desc.db_names]]).to(device)
    if min_score != -1:
        invalid |= scores < min_score
    scores.masked_fill_(invalid, float("-inf"))

    topk = torch.topk(scores, num_matched, dim=1)
    indices = topk.indices.cpu()
    valid = topk.values.isfinite().cpu()

    pairs : List[Tuple[str,str,float]] = []

    for i in range(valid.shape[0]):
        for j in range(valid.shape[1]):
            if valid[i,j]:
                name_query = query_desc.db_names[i]
                name_db = db_desc.db_names[indices[i,j].item()]
                score = scores[i, j].item()

                pairs.append((name_query, name_db, score))
    return pairs

def estimate_pose(trajectories: Dict[str,ImagePose], matches: List[Tuple[str,str,float]]):
    scores = torch.zeros((len(matches),))
    t = torch.zeros((len(matches), 3))
    R = torch.zeros((len(matches), 4))
    
    for i, (query, ref, score) in enumerate(matches):
        pose = trajectories[ref[ref.find("/") + 1:ref.find(".")].replace("__", "_")]
        print("Image ", ref, pose)
        t[i] = pose.t
        R[i] = pose.R
        scores[i] = score
    scores = torch.softmax(scores, dim=0)
    return R, t, scores


from hloc.utils.io import list_h5_names

def db_iterator(paths):
    if isinstance(paths, (Path, str)):
        paths = [paths]

    name2db = {n: i for i, p in enumerate(paths) for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())
    db_names = hloc.pairs_from_retrieval.parse_names(None, None, db_names_h5)

    names_by_db = itertools.groupby(db_names, key=lambda name: name2db[name])
    for idx, group in names_by_db:
        with h5py.File(str(paths[idx]), "r", libver="latest") as fd:
            for name in group:
                if name.startswith("floor_plan"):
                    continue
                yield name[name.rfind("/")+1:], fd[name]


def load_retrieval_database(paths) -> RetrievalFeatures:
    db_names = []
    desc = []
    for name, data in db_iterator(paths):
        db_names.append(name)
        desc.append(data['global_descriptor'].__array__())

    return RetrievalFeatures(
        features=torch.from_numpy(np.stack(desc)).type(torch.float),
        db_names=db_names,
    )


def load_keypoint_database(paths) -> LocalFeatures:
    if isinstance(paths, (Path, str)):
        paths = [paths]

    db_names = []
    desc = []
    keypoints = []
    scores = []
    for name, data in db_iterator(paths):
        db_names.append(name)
        desc.append(torch.tensor(data['descriptors'].__array__()))
        keypoints.append(torch.tensor(data['keypoints'].__array__()).squeeze(0))
        scores.append(torch.tensor(data['scores'].__array__()))

    return LocalFeatures(
        keypoints=keypoints,
        scores=scores,
        descriptors=desc,
        db_names=db_names
    )

def load_image_database(dir, images_in_mem=False) -> ImageDataset:
    image_shapes = []
    images = []
    db_names = []
    print(str(dir/"*.jpg"))
    for filename in glob.glob(str(dir/"*.jpg")):
        if images_in_mem:
            img = cv2.imread(filename)
            image_shapes.append(torch.tensor(img.shape, dtype=int))
            images.append(torch.tensor(img))
        else:
            img = PIL.Image.open(filename)
            width, height = img.size
            image_shapes.append(torch.tensor([height,width,len(img.getbands())], dtype=int))

        canon_filename = filename[filename.rfind("/")+1:]
        db_names.append(canon_filename)

    return ImageDataset(images=images, image_shapes=image_shapes, db_names=db_names)

def plot_pose_on_map(building_path, fig_path, final_pose):
    if len(final_pose.shape) == 1:
        final_pose = final_pose.unsqueeze(0)
    pos = final_pose[:, 4:7]

    floor_plan = cv2.imread(str(building_path / "reference" / "segmentation" / "floor_map.png"))

    with open(str(building_path / "reference" / "transforms" / "transforms.json")) as f:
        transforms = json.load(f)

    aabb_min = torch.tensor(transforms["min"])
    aabb_max = torch.tensor(transforms["max"])


    print("Plotting pos", pos, aabb_min, aabb_max)
    rel_pos = (pos - aabb_min.unsqueeze(0)) / (aabb_max.unsqueeze(0) - aabb_min.unsqueeze(0))
    #rel_pos = [rel_pos[1],rel_pos[0]]

    #rel_pos = torch.tensor([rel_pos[1],rel_pos[0]])
    img_pos = rel_pos[:, :2] * torch.tensor([floor_plan.shape[1],floor_plan.shape[0]]).unsqueeze(0)

    #plt.clf()
    plt.xlim(0, floor_plan.shape[1])
    plt.ylim(0, floor_plan.shape[0])
    plt.imshow(floor_plan)
    plt.scatter(img_pos[:,0].detach().numpy(), img_pos[:,1].detach().numpy())
    plt.savefig(fig_path)


def parse_trajectories(ref_trajectories):
    with open(ref_trajectories, 'r') as f:
        # todo: read camera
        camera = Camera( 
            width=1024,
            height=1024,
            K=torch.tensor([
                [1,0,0],
                [0,1,0],
                [0,0,1]
            ])
        )

        trajectories = {}
        for i, lines in enumerate(f.read().split("\n")[1:-1]):
            tokens = lines.replace(" ", "").split(",")
            cam = tokens[1]
            image = str(i // 4).zfill(5) + "-" + tokens[1]

            pose = ImagePose(
                camera=0, # todo: select based on cam
                t=torch.tensor([float(x) for x in tokens[6:9]]),
                R=torch.tensor([float(x) for x in tokens[2:6]])
            )
            
            trajectories[image] = pose
        
        return trajectories
    raise "Could not load reference trajectories"


logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@torch.jit.script
@dataclass
class LocalizationConf:
    device : torch.device
    num_retrieval : int = 3

class LocalizationModule(nn.Module):
    #retrieval_model : PretrainedModel
    #keypoint_model : PretrainedModel
    #matcher_model : PretrainedModel
    db_retrieval_desc : RetrievalFeatures
    db_feature_desc : LocalFeatures
    db_image : ImageDataset
    trajectories : Dict[str, ImagePose]

    def __init__(self, retrieval_model, keypoint_model, matcher_model,
                 db_retrieval_desc: RetrievalFeatures, db_feature_desc: LocalFeatures, db_image: ImageDataset, trajectories: Dict[str, ImagePose], device: torch.device, conf : LocalizationConf):
        super().__init__()

        self.retrieval_model = retrieval_model
        self.keypoint_model = keypoint_model
        self.matcher_model = matcher_model
        self.db_retrieval_desc = db_retrieval_desc
        self.db_feature_desc = db_feature_desc
        self.db_image = db_image
        self.trajectories = trajectories
        self.dummy_param = nn.Parameter(torch.zeros((), dtype=torch.float))
        self.conf = conf

    def get_device(self):
        return self.dummy_param.device

    def extract_retrieval_features(self, image, db_name: str ="tmp") -> RetrievalFeatures:
        model = self.retrieval_model
        image_tensor = model.preprocess(image).unsqueeze(0).to(self.get_device())
        features = model({"image": image_tensor})["global_descriptor"]

        return RetrievalFeatures(
            features=features,
            db_names=[db_name],
        )

    def extract_local_features(self, image, db_name: str ="tmp") -> LocalFeatures:
        model = self.keypoint_model
        image_tensor = model.preprocess(image).unsqueeze(0).to(self.get_device())
        pred = model({"image": image_tensor})
        pred = {k: v[0].type(torch.float) for k, v in pred.items()}
        # uncertainty = hloc.extract_features.calculate_uncertainity(model, data, pred)

        return LocalFeatures(
            keypoints=[pred["keypoints"]],
            scores=[pred["scores"]],
            descriptors=[pred["descriptors"]],
            db_names=[db_name],
        )

    def match_features(self, features1: LocalFeatures, features2: LocalFeatures, query_image: ImageDataset,
                       db_image: ImageDataset, retrieval_matches: List[Tuple[str, str, float]]) -> List[Dict[str, List[torch.Tensor]]]:
        model = self.matcher_model
        device = self.get_device()
        names2idx1: Dict[str, int] = {name: i for i, name in enumerate(features1.db_names)}
        names2idx2: Dict[str, int] = {name: i for i, name in enumerate(features2.db_names)}
        ftype = torch.float32

        result : List[Dict[str, List[torch.Tensor]]] = []
        for name_i, name_j, score in retrieval_matches:
            i = names2idx1[name_i]
            j = names2idx2[name_j]

            if len(db_image.images) == 0:  # todo: models require just correct shape, not the image
                print("index i=", i, "j=", j, ", image_shapes size", len(db_image.image_shapes))

                s0 = query_image.image_shapes[i]
                s1 = db_image.image_shapes[j]

                image0 = model.preprocess(torch.zeros((int(s0[0]), int(s0[1]), int(s0[2])), dtype=ftype))
                image1 = model.preprocess(torch.zeros((int(s1[0]), int(s1[1]), int(s1[2])), dtype=ftype))
            else:
                image0 = model.preprocess(query_image.images[i]).type(ftype)
                image1 = model.preprocess(db_image.images[j]).type(ftype)

            match_result = model({
                "scores0": to_batch(device, features1.scores[i]),
                "scores1": to_batch(device, features2.scores[j]),
                "descriptors0": to_batch(device, features1.descriptors[i]),
                "descriptors1": to_batch(device, features2.descriptors[j]),
                "keypoints0": to_batch(device, features1.keypoints[i]),
                "keypoints1": to_batch(device, features2.keypoints[j]),
                "image0": to_batch(device, image0),
                "image1": to_batch(device, image1),
            })
            result.append(match_result)
            print("==== Matched =====")

        return result

    def extract(self, image) -> Tuple[RetrievalFeatures, LocalFeatures]:
        return (
            extract_retrieval_features(self.retrieval_model, image), 
            extract_local_features(self.keypoint_model, image)
        )

    def forward(self, query_image: torch.Tensor):
        #t3 = time.time()
        #logger.info(f"Extract feature descriptors : {(t3 - t2) * 1000} ms")

        print("EXTRACTING RETRIEVAL")
        retrieval_desc = self.extract_retrieval_features(query_image)
        print("EXTRACTED RETRIEVAL")

        print("EXTRACTING LOCAL FEATURES")
        feature_desc = self.extract_local_features(query_image)
        print("EXTRACTED LOCAL FEATURES")

        print("FINDING IMAGE PAIRS")
        matches = pairs_from_retrieval(self.get_device(), retrieval_desc, self.db_retrieval_desc, num_matched=self.conf.num_retrieval)
        print("FOUND IMAGE PAIRS")

        print("Matches", matches)
        #t4 = time.time()
        #logger.info(f"Match images in database : {(t4 - t3) * 1000} ms")

        query_image_dataset = ImageDataset(
            images=[query_image],
            image_shapes=[torch.tensor(query_image.shape)],
            db_names=retrieval_desc.db_names
        )

        kp_matches = self.match_features(feature_desc, self.db_feature_desc, query_image_dataset, self.db_image, matches)
        R, t, scores = estimate_pose(self.trajectories, matches)
        #t5 = time.time()
        #logger.info(f"Localize : {(t5 - t1) * 1000} ms")
        return torch.cat([R, t, scores.unsqueeze(1)], dim=1)


def downsample_directory(src_dir, dst_dir, target_size):
    for file in glob.glob(src_dir+"/*.jpg"):
        img = cv2.imread(file)
        dst_file = dst_dir + file[file.find(src_dir+"/"):]
        img_downsampled = cv2.resize(img, target_size)
        print(file, "->", dst_file)
        cv2.imwrite(dst_file)


def create_localizer(base_path, conf, retrieval_conf, feature_conf, matcher_conf, build=False):
    ref_model_path = base_path / "raw_data/pointcloud_small.ply"
    ref_images = base_path / "raw_data/images_undistr_center"
    ref_trajectories = base_path / "trajectories.txt"

    outputs = base_path

    output_ref = outputs / "reference"
    output_ref_retrieval = output_ref / "retrieval"
    output_ref_features = output_ref / "features"

    feature_ref_path = output_ref_features / (feature_conf["output"]+".h5")

    if not os.path.exists(feature_ref_path):
        #query_image = cv2.imread("../data/tmp/query.jpg")
        #query_image = torch.from_numpy(query_image)
        #plt.imshow(PretrainedModel(hloc.extractors, feature_conf, 'cpu').preprocess(query_image)[0, :, :], cmap='gray')
        #plt.show()

        if build:
            logging.warning("===== Extracting retrieval features ====")
            freeze_support()
            feature_path = hloc.extract_features.main(conf=feature_conf, image_dir=ref_images, export_dir=output_ref_features)
        else:
            raise Exception("Could not find stored retrieval features")

    retrieval_ref_path = output_ref_retrieval / (retrieval_conf["output"]+".h5")
    if not os.path.exists(retrieval_ref_path):
        if build:
            logging.warning("===== Extracting descriptor features ====")
            retrieval_ref_path = hloc.extract_features.main(conf=retrieval_conf, image_dir=ref_images,
                                                       export_dir=output_ref_retrieval)
        else:
            raise Exception("Could not find stored local features")

    device = conf.device

    trajectories = parse_trajectories(ref_trajectories)
    print(trajectories.keys())

    debug_map = False
    if debug_map:
        poses = []
        for i in range(500):
            idx = str(i).zfill(5)+"-cam0_center"
            final_pose = torch.tensor(
                torch.cat([trajectories[idx].R, trajectories[idx].t])
            )
            poses.append(final_pose)
        poses = torch.stack(poses)
        #poses = torch.tensor([[0,0,0,0,16.08563, -0.06662704, 0.9588583]])

        plot_pose_on_map(base_path, "../data/tmp/fig.png", poses)
        plt.show()

    matcher_model = PretrainedModel(hloc.matchers, matcher_conf, device)
    #torch.jit.script(matcher_model)
    retrieval_model = PretrainedModel(hloc.extractors, retrieval_conf, device)
    keypoint_model = PretrainedModel(hloc.extractors, feature_conf, device)

    t1 = time.time()
    db_retrieval_desc = load_retrieval_database(retrieval_ref_path)
    t2 = time.time()
    logger.info(f"Loaded retrieval database : {(t2 - t1) * 1000} ms")
    db_feature_desc = load_keypoint_database(feature_ref_path)
    t3 = time.time()
    logger.info(f"Loaded keypoint database : {(t3 - t2)*1000}")

    matcher_requires_images = not (matcher_conf["model"]["name"] in ["lightglue", "superglue", "nearest"])
    db_image = load_image_database(ref_images, images_in_mem= matcher_requires_images)
    t4 = time.time()
    logger.info(f"Loaded image database : {(t4 - t3)*1000}")


    return LocalizationModule(
        retrieval_model,
        keypoint_model,
        matcher_model,
        db_retrieval_desc, db_feature_desc, db_image, trajectories, device, conf)

def compile_model():
    debug_fig_plot_path = "../data/outputs/CAB_navviz/logging/fig.png"
    from torch.utils.mobile_optimizer import optimize_for_mobile

    building_path = Path("../data/HG_navviz")
    localizer_conf = LocalizationConf(
        num_retrieval=3,
        device="mps"
    )
    retrieval_conf = {
        "output": "global-feats-netvlad_800",
        "model": {"name": "netvlad"},
        "preprocessing": {"resize_max": 800},
    }

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

    matcher_conf = {
        "output": "matches-superpoint-lightglue",
        "model": {
            "name": "lightglue",
            "features": "superpoint",
        },
    }

    localizer = create_localizer(building_path, localizer_conf, retrieval_conf, feature_conf, matcher_conf, build=True)
    localize = localizer

    # Load the saved TorchScript model
    query_image = cv2.imread("../data/tmp/fountain.jpg") #00162-cam3__center.jpg")

    size = query_image.shape
    scale = min(feature_conf["preprocessing"]["resize_max"], retrieval_conf["preprocessing"]["resize_max"]) / max(size)
    size_new = (int(size[0] * scale), int(size[1] * scale))
    query_image = cv2.resize(query_image, size_new)
    query_image = torch.from_numpy(query_image)

    gc.collect()

    t1 = time.time()
    print("Start localization")
    final_pose = localizer(query_image)
    t2 = time.time()
    print("Localized ", final_pose, " in ", (t2-t1)*1000, "ms")

    plot_pose_on_map(building_path, debug_fig_plot_path, final_pose)
    plt.show()

    scripted_model = torch.jit.script(localizer)
    gc.collect()
    print("==== SCRIPTED MODEL =====")
    #scripted_output = final_pose # scripted_model(query_image)

    #scripted_model = optimize_for_mobile(scripted_model, backend='cpu')
    print(scripted_model.graph)

    #torch.onnx.export(scripted_model, query_image, "model.onnx", verbose=True, input_names=["image"], output_names=["localization"]))
    scripted_model.save("model.pt")

    serialized_model = torch.jit.load("model.pt").cpu()
    serialized_output = serialized_model(query_image.cpu())

    print("Pose : ", final_pose, serialized_output)
    plt.show()

if __name__ == "__main__":
    freeze_support()
    compile_model()
