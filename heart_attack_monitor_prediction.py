from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# Endpoint to receive data from the Raspberry Pi Pico W
@app.route('/predict-data', methods=['POST'])
def receive_data():
    model = tf.keras.models.load_model('heart_attack_svm.h5')
    data = request.get_json()
    print("Received data:", data)

    x_val = [data[' PULSE'], data[' SpO2']]
    y_test_pred = model.predict(x_val)
    y_test_pred = (y_test_pred > 0.5).astype(int)
    # Process the received data
    # Perform ML predictions or any other desired operation

    # Prepare response data
    response_data = {'Healthy heart': y_test_pred}

    return jsonify(response_data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

