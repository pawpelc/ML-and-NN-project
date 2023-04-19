from flask import Flask, jsonify, request
import pandas as pd
import pickle
from functions import data_preparation, heuristic
import numpy as np

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Read data from CSV file and prepare
    data = pd.read_csv('data.csv')
    X, y = data_preparation(data)

    # Get selected model from request
    selected_model = request.json['model']

    # Load selected model
    if selected_model == 'hypermodel':
        # Load model
        from tensorflow import keras
        model = keras.models.load_model('models/hypermodel.h5')
        # Preprocess the data
        from sklearn.utils import shuffle
        from sklearn.preprocessing import MinMaxScaler
        samples, labels = X.to_numpy(), y.to_numpy() - 1
        samples, labels = shuffle(samples, labels, random_state=0)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_samples = scaler.fit_transform(samples)
        # Predict
        model_pred = model.predict(x=scaled_samples, verbose=0)
        predictions = np.argmax(model_pred, axis=-1)

    elif selected_model == 'logistic_reg':
        with open("logistic_reg.pkl", 'rb') as file:
            model = pickle.load(file)
        predictions = model.predict(X)

    elif selected_model == 'xgb':
        with open("xgb.pkl", 'rb') as file:
            model = pickle.load(file)
        predictions = model.predict(X)

    elif selected_model == 'heuristic':
        predictions = heuristic(X, y)[0]

    else:
        return jsonify({'error': 'Invalid model selection'})

    # Return the predictions as a JSON response
    return jsonify(predictions.tolist())


if __name__ == '__main__':
    app.run(debug=True)