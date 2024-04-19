from distutils.log import debug
from fileinput import filename
from flask import *
from localize import localize
from pathlib import Path
from navigation import building, nearest_node, navigate
from building_model import *
import cv2
import os

conn = None
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

        print("Found pose ", pose)
        return {
            "pos": list(pos),
        }

@app.route('/navigate', methods=['GET'])
def path():
    srcPosition = np.array([float(x) for x in request.args.get("srcPosition").split(",")])
    dstLocation = request.args.get("dstLocation")

    query = """
    SELECT contour.ToString() 
    FROM BuildingModel.Location
    WHERE id=?
    """
    # todo: suboptimal, could query node -> location directly
    # but currently ids don't match between building_model and database
    cursor = conn.cursor()
    cursor.execute(query, (dstLocation,))
    contour = parse_geometry(cursor.fetchone()[0])
    dstPosition = np.mean(contour, axis=0)

    z = 0 # todo: find actual z
    dstPosition = [dstPosition[0],dstPosition[1], z]

    print(dstPosition)

    src = nearest_node(building, srcPosition)
    dst = nearest_node(building, dstPosition)

    print(src, dst)

    return [p.to_json() for p in navigate(building, src, dst)]


@app.route('/map', methods=['GET'])
def map():
    building_id = request.args.get("building")
    matches = request.args.get("matches", "")
    floor_label = request.args.get("floor", 0)

    locations = db_query_locations(conn.cursor(), LocationFilter(building_id=building_id, matches=matches))

    floor = building.floors[floor_label]

    # todo: get outline
    json = floor.to_json()
    json["locations"] = [location.to_json() for location in locations]
    return json

if __name__ == '__main__':
    host = os.environ["SQL_SERVER_HOST"]
    port = os.environ["SQL_SERVER_PORT"]  
    pwd = os.environ["SQL_SERVER_PASSWORD"]

    conn = pyodbc.connect("Driver={ODBC Driver 18 for SQL Server};"
                          +f"Server={host};"
                          +f"Port={port};"
                          +f"Database=master;"
                          +f"UID=sa;"
                          +f"PWD={pwd};"
                          +f"TrustServerCertificate=yes")
    
    app.conn = conn

    app.run(debug=True, port=3001, host='0.0.0.0')