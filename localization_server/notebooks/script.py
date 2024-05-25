import json
import numpy as np

building_path = "../data/HG_navviz"
transforms_path = building_path + "/reference/transforms/transforms.json"
path_segmentation = building_path + "/reference/segmentation"
walkable_area_path = path_segmentation + "/walkable_areas_raw.json"
aligned_walkable_area_path = path_segmentation + "/walkable_areas.json"

with open(walkable_area_path) as f:
    walkable_areas = json.load(f)

with open(transforms_path) as f:
    transforms = json.load(f)

trans = np.array(transforms["relative_to_absolute"]) @ np.array(transforms["floor_to_relative"])
walkable_areas_aligned = []
for contour in walkable_areas["walkable_areas"]:
    contour = np.array(contour)
    contour = contour / 100.0

    contour_homo = np.concatenate([contour, np.ones((contour.shape[0],1))], axis=1)
    aligned_contour = (trans @ contour_homo.transpose()).transpose()[:,:2]

    walkable_areas_aligned.append(aligned_contour.tolist())

with open(aligned_walkable_area_path,"w") as f:
    walkable_areas["walkable_areas"] = walkable_areas_aligned
    json.dump(walkable_areas, f)


