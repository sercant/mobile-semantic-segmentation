import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from data import standardize

prefix = 'hair_recognition'


def main(pb_file, img_file):
    """
    Predict and visualize by TensorFlow.
    :param pb_file:
    :param img_file:
    :return:
    """
    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name=prefix)

    for op in graph.get_operations():
        print(op.name)

    x = graph.get_tensor_by_name('%s/input_1:0' % prefix)
    y = graph.get_tensor_by_name('%s/output_0:0' % prefix)

    loaded_image = cv2.cvtColor(cv2.imread(img_file,-1), cv2.COLOR_BGR2RGB)
    resized_image =cv2.resize(loaded_image, (128, 128))
    input_image = np.expand_dims(np.float32(resized_image[:128, :128]),axis=0)/255.0

    # images = np.load(img_file).astype(float)
    # img_h = images.shape[1]
    # img_w = images.shape[2]

    with tf.Session(graph=graph) as sess:
        # for img in images:
        # batched = img.reshape(-1, img_h, img_w, 3)
        normalized = standardize(input_image)
        
        converter = tf.contrib.lite.TocoConverter.from_session(sess, [x], [y])
        tflite_model = converter.convert()
        open("artifacts/converted_model.tflite", "wb").write(tflite_model)

        # Load TFLite model and allocate tensors.
        interpreter = tf.contrib.lite.Interpreter(model_content=tflite_model)
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        # input_shape = input_details[0]['shape']
        input_data = np.array(normalized, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # print(output_data)
        

        # pred = sess.run(y, feed_dict={
        #     x: normalized
        # })
        plt.imshow(output_data.reshape(128, 128))
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pb_file',
        type=str,
        default='artifacts/model.pb',
    )
    parser.add_argument(
        '--img_file',
        type=str,
        default='data/glasshair.jpg',
        help='image file as numpy format'
    )
    args, _ = parser.parse_known_args()
    main(**vars(args))
