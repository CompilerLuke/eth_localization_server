from google.cloud import storage
import os
import io
from auth import google_token
from flask import Blueprint, request, current_app, send_file, Response
import tempfile

project_id = os.environ["GCP_PROJECT_ID"]
map_data_bucket_name = os.environ["GCP_MAP_DATA_BUCKET"]

"""
    //  1. Before running this sample,
    //  set up ADC as described in https://cloud.google.com/docs/authentication/external/set-up-adc
    //  2. Replace the project variable.
    //  3. Make sure that the user account or service account that you are using
    //  has the required permissions. For this sample, you must have "storage.buckets.list".
"""

storage_client = storage.Client(project=project_id)
map_data_bucket = storage_client.get_bucket(map_data_bucket_name)

bp = Blueprint("cloud_bucket", __name__)

def upload_building_file(
    username : str,
    fileName : str,
    fileType : str,
    building : int,
    metadata : str,
    file : io.FileIO,
    replace : bool = False
):
    cursor = current_app.conn.cursor()
    
    blobName = str(building) + "%" + username + "%" + fileName

    if replace:
        query = """
        SELECT id FROM BuildingScan.FileUpload WHERE username=? filename=?
        """
        cursor.execute(query, (username,fileName))
    else:
        query = """
        INSERT INTO BuildingScan.FileUpload (username,building,filename,blobFilename,fileType,metadata)
        OUTPUT inserted.id
        VALUES (?,?,?,?,?,?)
        """
        print("TRY TO INSERT Upload with building ", building)
        cursor.execute(query, (username, building, fileName,"", fileType, metadata))
    id = cursor.fetchone()[0]

    try:
        blob = map_data_bucket.blob(blobName)
        file.seek(0)
        blob.upload_from_file(file)    
    
        query = """
        UPDATE BuildingScan.FileUpload
        SET blobFilename=?, metadata=?, fileType=?
        WHERE id=?
        """
        cursor.execute(query, (blobName, metadata, fileType, id))

        print("Uploaded file ", blobName)
        cursor.commit()
        return "Okay"
    except Exception as e:
        print("ERROR", e)
        return Response("Failed to upload", status=500)
    


@bp.route("/mapping/file_server/upload", methods=['POST','PUT'])
def upload_building_file_route():
    if not request.method in ['POST','PUT']:
        print("Incorrect request method ", request.method)
        return Response(status=400)
    
    token = google_token()
    if token is None:
        return Response("Please sign in", status=401)
    
    if len(request.files) != 1:
        return Response("Expecting one file", status=400)

    fileName= request.form['fileName']
    fileType= request.form['fileType']
    building= request.form['building']
    metadata= request.form['metadata']
    replace = request.form["replace"] if 'replace' in request.args else False

    file = list(request.files.values())[0]
    username = token["email"]

    return upload_building_file(
        fileName=fileName,
        fileType=fileType,
        building=building,
        metadata=metadata,
        username=username,
        file=file
    )

def query_building_file_metadata(
    username : str,
    fileName : str
):
    query = """
    SELECT id,blobFilename,fileType,metadata FROM BuildingScan.FileUpload 
    WHERE username=? AND fileName=?
    """
    
    cursor = current_app.conn.cursor()
    cursor.execute(query, (username, fileName))
    id,blobFilename,fileType,metadata = cursor.fetchone()
    return {
        "id": id,
        "blobFilename": blobFilename,
        "fileType": fileType,
        "metadata": metadata
    }

def download_building_file(
    file : io.FileIO,
    username : str,
    fileName : str
):
    metadata = query_building_file_metadata(username=username, fileName=fileName)
    print("Download ", metadata["blobFilename"])
    blob = map_data_bucket.blob(metadata["blobFilename"])
    blob.download_to_file(file)
    return metadata

@bp.route("/mapping/file_server/metadata")
def download_building_file_metadata():
    fileName = request.args['fileName']
    username = google_token().email
    metadata = query_building_file_metadata(fileName=fileName, username=username)
    return metadata, 'application/json'

@bp.route("/mapping/file_server/download")
def download_building_file_route():   
    fileName = request.args['fileName']
    username = google_token().email
    file = io.FileIO()
    download_building_file(file=file, username=username, fileName=fileName)
    send_file(file)



def config_map_storage(app):
    app.register_blueprint(bp)

if __name__ == "__main__":
    pass