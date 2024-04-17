from distutils.log import debug
from fileinput import filename
from flask import *
from localize import localize
from pathlib import Path
from navigation import building, nearest_node, navigate
import cv2

app = Flask(__name__)


@app.route('/')
def main():
    return ""

@app.route('/localize', methods=['POST'])
def localize_req():
    if request.method == 'POST':
        f = request.files['upload_image']
        filename = "../data/tmp/query.jpg"
        f.save(filename)

        img = cv2.imread(filename)

        pose = localize(img)
        pos = pose[4:]

        node = nearest_node(building, pos)

        print("Found pose ", pose)
        return {
            "pos": list(pos),
            "node": node.to_json()
        }

@app.route('/shortest_path', methods=['GET'])
def path():
    src = request.args.get("src")
    dst = request.args.get("dst")
    return [p.to_json() for p in navigate(building, src, dst)]


@app.route('/map', methods=['GET'])
def map():
    floor_label = request.args.get("floor", 0)
    floor = building.floors[floor_label]

    json = floor.to_json()
    json["rooms"] = [building.rooms[id].to_json() for id in json["rooms"]]
    return json

if __name__ == '__main__':
    app.run(debug=True, port=3001, host='0.0.0.0')