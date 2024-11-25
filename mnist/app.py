from flask import Flask, request, jsonify, render_template
import os
from load_image import predict_image

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return '<p>Hello, World!</p>'

@app.route('/predict', methods=['GET'])
def get_predict():
    return render_template('getimage.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    filename = file.filename

    file.save(filename)

    predicted_class = predict_image(filename)
    os.remove(filename)

    return jsonify({'number predicted': predicted_class})


if __name__ == '__main__':
    app.run(debug=True)