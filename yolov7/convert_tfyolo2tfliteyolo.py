import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('./content/')
tflite_model = converter.convert()

with open('./content/yolov7_model.tflite', 'wb') as f:
  f.write(tflite_model)