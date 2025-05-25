import tensorflow as tf

# Load Keras model
model = tf.keras.models.load_model("ShapeDetection.keras")
model.output_names=['output']
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the .tflite model
with open("ShapeDetection.tflite", "wb") as f:
    f.write(tflite_model)

import tf2onnx

# Load Keras model
model = tf.keras.models.load_model("ShapeDetection.keras")
model.output_names=['output']
# Convert to ONNX
spec = (tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path="ShapeDetection.onnx")