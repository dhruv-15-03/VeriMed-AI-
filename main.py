from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from Diabetes_script import predict_from_report

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(filepath)
        try:
            result = predict_from_report(filepath)
            os.remove(filepath)
            return jsonify({'result': result})
        except Exception as e:
            return jsonify({'error1 ': filepath}), 500

    return jsonify({'error': 'File upload failed'}), 500

if __name__ == '__main__':
    app.run(debug=True,port=5001)
