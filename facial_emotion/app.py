from flask import Flask, request, jsonify, render_template
import os
from load_model  import predict_image
import uuid

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static'

@app.route('/', methods=['GET'])
def hello():
    return '<p>Hello, World!</p>'

@app.route('/predict', methods=['GET'])
def get_predict():
    return render_template('getimage.html')


@app.route('/estimate', methods=['POST'])
def predict():
    file = request.files['image']
    extension = file.filename.split('.')[-1]
    filename = str(uuid.uuid4()) + '.' + extension
    file.filename = filename
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(save_path)

    predicted_class = predict_image(save_path)
    save_path = '../' + save_path

    return render_template('display_result.html', prediction=predicted_class, probability=save_path)


if __name__ == '__main__':
    app.run(debug=True)