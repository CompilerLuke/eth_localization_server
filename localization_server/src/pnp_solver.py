import cv2
import numpy as np
import cv2.gapi
import torch
import h5py
import torch
from types import SimpleNamespace

class Camera:
    def __init__(self, param):
        fx, fy, cx, cy = param
        self.c = np.array([cx, cy])
        self.f = np.array([fx, fy])

    def get_intrinsic_matrix(self):
        fx, fy = self.f
        cx, cy = self.c
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


class H5py_file_reader:
    def __init__(self, path):
        self.f = h5py.File(path, 'r')

    def get_group(self, name):
        group = self.f[name]
        dic = {}
        for k, v in group.items():
            data = np.array(v.__array__())
            dic[k] = torch.from_numpy(data)

        return dic 

def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


class PnpSolver:
    def __init__(self):
        self.camera = Camera([960, 960, 639.8245614035088, 959.8245614035088])

    ## qurey_features: output of feature extraction of query image, from extract_features(...)
    ## group: h5py group of ref image
    ## pred: output of feature matching, pred = matcher(inputs)
    ## ref_pose: pose of ref image from trajectory.txt 
    def solve(self, qurey_features, group, pred):
        # qurey_features = rbd(qurey_features)
        pred = rbd(pred)
        ## matches.keys=dict_keys(['matches0', 'matches1', 'matching_scores0', 'matching_scores1', 'stop', 'matches', 'scores', 'prune0', 'prune1'])
        matches = SimpleNamespace(**pred)

        ## get 2d points from query image
        p2d = qurey_features['keypoints'][matches.matches[:, 0]]

        ## get 3d points from ref image
        p3d = group['p3d'][matches.matches[:, 1]]

        assert len(p2d) == len(p3d)

        ## solve pnp
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(p3d.numpy(), p2d.numpy(), self.camera.get_intrinsic_matrix(), None)

        ## recover rotation matrix and translation. 
        ## [P]c = rmat [P]w + tvec
        np_rodrigues = np.asarray(rvec[:,:],np.float64) 
        rmat, _ = cv2.Rodrigues(np_rodrigues)
        camera_position = -np.matrix(rmat).T @ np.matrix(tvec)
        rotation_matrix = rmat.T


        ## calcualte score 
        ratio = len(inliers) / len(p2d)

        return rotation_matrix, camera_position, ratio,



