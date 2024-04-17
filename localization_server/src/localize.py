from pathlib import Path
from pprint import pformat
import hloc.extract_features, hloc.match_features, hloc.pairs_from_retrieval

from hloc.utils import base_model
import open3d as o3d
import cv2
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

import logging
import time

retrieval_conf = hloc.extract_features.confs["netvlad"]
feature_conf = hloc.extract_features.confs["superpoint_inloc"]
matcher_conf = hloc.match_features.confs["superglue"]

ref_model_path = Path("../data/CAB_navviz/raw_data/pointcloud_small.ply")
ref_images = Path("../data/CAB_navviz/raw_data")
ref_trajectories = Path("../data/CAB_navviz/trajectories.txt")

outputs = Path("../data/outputs/CAB_navviz")

output_ref = outputs / "reference"
output_ref_retrieval = output_ref / "retrieval"
output_ref_features = output_ref / "features"

output_query = outputs / "query"
output_query_retrieval = output_query / "retrieval"
output_query_features = output_query / "features"
loc_pairs = output_query / "loc_pairs.txt"
segmentation_path = outputs / "segmentation/plan_vote.png"

debug_fig_plot_path = outputs / "logging/fig.png"

# feature_path = extract_features.main(conf=feature_conf, image_dir=ref_images, export_dir=output_ref_features)
feature_path = outputs / "feats-superpoint-n4096-r1600.h5"

# retrieval_ref_path = extract_features.main(conf=retrieval_conf, image_dir=ref_images, export_dir=output_ref_retrieval)
retrieval_ref_path = output_ref_retrieval / "global-feats-netvlad.h5"

floor_plan = cv2.imread(str(segmentation_path))

device = base_model.select_device()

with open(ref_trajectories, 'r') as f:
    trajectories = {}
    for i, lines in enumerate(f.read().split("\n")[1:-1]):
        tokens = lines.replace(" ", "").split(",")
        cam = tokens[1]
        image = str(i // 4).zfill(5) + "-" + tokens[1]
        trajectories[image] = tokens[2:]

print(len(trajectories) // 4)

from hloc import extractors
import functools
import torch

loaded_model_cache = {}
def load_model(conf: dict, device: torch.device):
    global loaded_model_cache

    key = str((conf,device))
    if key in loaded_model_cache:
        return loaded_model_cache[key]

    Model = base_model.dynamic_load(extractors, conf["model"]["name"])
    model = Model(conf["model"]).eval().to(device)
    loaded_model_cache[key] = model
    return model

# Preload
def extract_features(conf, device, image, db_name="tmp"):
    data = hloc.extract_features.normalize_image(conf["preprocessing"], image)

    image_tensor = torch.tensor(data["image"], device=device, requires_grad=False).unsqueeze(0)

    model = load_model(conf, device)
    pred = model({"image": image_tensor})
    pred = {k: v[0] for k, v in pred.items()}
    uncertainty = hloc.extract_features.calculate_uncertainity(model, data, pred)

    pred = {k: v.unsqueeze(0) for k, v in pred.items()}
    pred["uncertainty"] = [uncertainty]
    pred["db_names"] = [db_name]

    return pred

from hloc.utils.io import list_h5_names

def load_database(device, paths, key="global_descriptor"):
    if isinstance(paths, (Path, str)):
        paths = [paths]

    name2db = {n: i for i, p in enumerate(paths) for n in list_h5_names(p)}
    db_names_h5 = list(name2db.keys())
    db_names = hloc.pairs_from_retrieval.parse_names(None, None, db_names_h5)
    return {
        "db_names": db_names,
        key: hloc.pairs_from_retrieval.get_descriptors(db_names, paths, name2db, key).to(device)
    }

def pairs_from_retrieval(device, db_desc, query_desc, num_matched, key="global_descriptor"):
    sim = torch.einsum("id,jd->ij", query_desc[key].to(device), db_desc[key].to(device))

    # Avoid self-matching
    self = np.array(query_desc["db_names"])[:, None] == np.array(db_desc["db_names"])[None]
    pairs = hloc.pairs_from_retrieval.pairs_from_score_matrix(sim, self, num_matched, min_score=0)
    pairs = [(query_desc["db_names"][i],db_desc["db_names"][j]) for i, j in pairs]
    return pairs


def estimate_pose(device, db_retrieval_desc, matches):
    poses = np.zeros((len(matches), 7))
    for i, (query, ref) in enumerate(matches):
        pose = trajectories[ref[ref.find("/") + 1:ref.find(".")].replace("__", "_")]
        poses[i] = pose[:7]

    final_pose = np.mean(poses, axis=0)
    return final_pose


def plot_pose_on_map(final_pose):
    pos = final_pose[4:7]

    aabb_min = np.array([-80.19033051, -160.14266968, -14.49797497])
    aabb_max = np.array([173.27059428, 107.40621185, 46.93898773])

    rel_pos = (pos - aabb_min) / (aabb_max - aabb_min)

    img_pos = rel_pos[:2] * [floor_plan.shape[1],floor_plan.shape[0]]
    #plt.clf()
    plt.imshow(floor_plan)
    plt.scatter([img_pos[0]], [img_pos[1]])
    plt.savefig(debug_fig_plot_path)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

db_feature_desc = None

t1 = time.time()
db_retrieval_desc = load_database(device, retrieval_ref_path)
t2 = time.time()
logger.info(f"Loaded image database : {(t2-t1)*1000} ms")


def localize(query_image):
    t1 = time.time()
    retrieval_desc = extract_features(retrieval_conf, device, query_image)
    t2 = time.time()
    logger.info(f"Extract retrieval descriptors : {(t2-t1)*1000} ms")
    feature_desc = extract_features(feature_conf, device, query_image)
    t3 = time.time()
    logger.info(f"Extract feature descriptors : {(t3-t2)*1000} ms")
    matches = pairs_from_retrieval(device, db_retrieval_desc, retrieval_desc, num_matched=5)
    t4 = time.time()
    logger.info(f"Match images in database : {(t4-t3)*1000} ms")
    final_pose = estimate_pose(device, db_feature_desc, matches)
    t5 = time.time()
    logger.info(f"Localize : {(t5-t1)*1000} ms")
    return final_pose

if __name__ == "__main__":
    img = cv2.imread("../data/tmp/query.jpg")
    final_pose = localize(img)
    localize(img)
    plot_pose_on_map(final_pose)
    plt.show()
else:
    matplotlib.use('Agg')
