from flask import *
from auth import config_oauth, google_token
from cloud_bucket import config_map_storage
from building_model import *
import automatic_annotation
import os
import pyodbc

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

@app.route("/mapping/new_building", methods=['POST'])
def new_building():
    token = google_token()
    if token is None:
        return Response("Please sign in", status=401)
    
    name = request.form["name"]
    username = token["email"]

    cursor = app.conn.cursor()
    id = db_create_building_entry(cursor, BuildingModel(id=0,name=name,rooms=[],floors=[],graph=None,creator=username))
    cursor.commit()
    print("New id", id)
    return {
        "id": id,
    }


#@app.route('/mapping/upload/file')

def config_db(app):
    host = os.environ["SQL_SERVER_HOST"]
    port = os.environ["SQL_SERVER_PORT"]  
    pwd = os.environ["SQL_SERVER_PASSWORD"]

    app.conn = pyodbc.connect("Driver={ODBC Driver 18 for SQL Server};"
                        +f"Server={host};"
                        +f"Port={port};"
                        +f"Database=master;"
                        +f"UID=sa;"
                        +f"PWD={pwd};"
                        +f"TrustServerCertificate=yes")

if __name__ == '__main__':
    app.config.from_envvar('WEBSITE_CONF')
    config_db(app)
    config_oauth(app)
    config_map_storage(app)
    app.run(debug=True, port=6000, host='0.0.0.0')