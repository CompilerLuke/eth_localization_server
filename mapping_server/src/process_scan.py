import cloud_bucket
import tempfile
import zipfile
import serve
import flask
import hloc.extract_features, hloc.match_features, hloc.pairs_from_retrieval
from time import sleep
import os

retrieval_conf = hloc.extract_features.confs["netvlad"]
feature_conf = hloc.extract_features.confs["superpoint_inloc"]

def process_scan(username, building, filename):
    tmp_dir = tempfile.TemporaryDirectory()
    
    tf_zip = open("../data/zip_download.zip","rb+") #tempfile.NamedTemporaryFile(suffix=".zip")

    cloud_bucket.download_building_file(tf_zip, username=username, fileName=filename)
    print("")
    #tf_zip.close()

    tf_dir = tempfile.TemporaryDirectory()

    with zipfile.ZipFile(tf_zip, 'r') as zip_ref:
        print("Extracting")
        zip_ref.extractall(tf_dir.name)

    
    ref_images = tf_dir.name+"/"+os.listdir(tf_dir.name)[0]+"/images"

    output_ref_features = tempfile.TemporaryDirectory()
    output_ref_retrieval = tempfile.TemporaryDirectory()

    feature_path = hloc.extract_features.main(conf=feature_conf, image_dir=ref_images, export_dir=output_ref_features.name)
    retrieval_path = hloc.extract_features.main(conf=retrieval_conf, image_dir=ref_images, export_dir=output_ref_retrieval.name)

    cloud_bucket.upload_building_file(
        username=username,
        fileName=fileName+"$retrieval",
        fileType="h5",
        building=building,
        metadata="",
        file=retrieval_path
    )

    cloud_bucket.upload_building_file(
        username=username,
        fileName=fileName+"$features",
        fileType="h5",
        building=building,
        metadata="",
        file=feature_path
    )


if __name__ == "__main__":
    serve.config_db(serve.app)
    with serve.app.app_context():
        process_scan("dummy@gmail.com", 3, "8")
    pass