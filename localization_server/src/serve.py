from distutils.log import debug
from fileinput import filename
from flask import *
from localize import localize
from pathlib import Path

app = Flask(__name__)


@app.route('/')
def main():
    return ""

@app.route('/localize', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['upload_image']
        filename = "../data/tmp/query.jpg"
        f.save(filename)

        pose = localize(Path("../data/tmp"))

        print("Found pose ", pose)

        return ""


if __name__ == '__main__':
    app.run(debug=True, port=3001, host='0.0.0.0')