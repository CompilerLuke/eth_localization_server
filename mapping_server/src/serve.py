from flask import *
import automatic_annotation

app = Flask(__name__)

@app.route('/')
def main():
    return ""

floor_plan_img_path = "../../floor_plan_segmentation/data/cab_floor_0.png"
floor_plan_segmentation = "../../floor_plan_segmentation/data/cab_floor_segmentation.json"

@app.route('/mapping/annotation/floorplan_img', methods=['GET'])
def floorplan():
    return send_file(floor_plan_img_path, mimetype='image/jpeg')

@app.route('/mapping/annotation/floorplan_annotation', methods=['GET'])
def floorplan_annotations():
    return send_file(floor_plan_segmentation, mimetype='application/json')


@app.route('/mapping/annotation/automatic_annotate_floor_plan', methods=['POST'])
def automatic_annotation():
    pass
    #automatic_annotation.annotate()

if __name__ == '__main__':
    app.run(debug=True, port=6000, host='0.0.0.0')