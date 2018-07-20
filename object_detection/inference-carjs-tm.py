import argparse
import os
import io

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

from utils import visualization_utils as vis_util
from utils import label_map_util

import logging

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

NUM_CLASSES = 1


def parse_args(check=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    #parser.add_argument('--dataset_dir', type=str, required=True)
    FLAGS, unparsed = parser.parse_known_args()
    return FLAGS, unparsed


if __name__ == '__main__':
    FLAGS, unparsed = parse_args()


    PATH_TO_CKPT = '/output/exported_graphs/frozen_inference_graph.pb'
    PATH_TO_LABELS = '/data/jia0/faster-rcnn-inception-resnet-v2-coco-2018-01-28/label_map.pbtxt'

    tfrecord_files = "/data/jia0/faster-rcnn-inception-resnet-v2-coco-2018-01-28/inference.tfrecord-*"
    #tfrecord_files = "/data/jia0/car-detection-fasterrcnn-inception-resnet/train1w.tfrecord-*"

    export_file = os.path.join(FLAGS.output_dir, 'inception-resnet_99_675_10W.csv')
    #export_file = os.path.join(FLAGS.output_dir, 'example_1w.csv')

    threhold = 0.775



    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    #label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    #categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    #category_index = label_map_util.create_category_index(categories)

    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)


    jpgs = []
    boxstrs = []
    width = 1069
    height = 500

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            files = tf.train.match_filenames_once(tfrecord_files)
            filename_queue = tf.train.string_input_producer(files, shuffle=True, num_epochs=1)
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'imgname': tf.FixedLenFeature([], tf.string),
                    'imgraw': tf.FixedLenFeature([], tf.string),
                })

            # tf.train.match_filenames_once函数需要初始化
            tf.local_variables_initializer().run()
            print(sess.run(files))
            # 声明tf.train.Coordinator类来协同不同线程，并启动线程
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # 多次执行获取数据的操作
            for index in range(5000):

                imgname, imgraw = sess.run([features['imgname'], features['imgraw']])

                encoded_jpg_io = io.BytesIO(imgraw)
                image = Image.open(encoded_jpg_io)
                image_np = load_image_into_numpy_array(image)
                image_np_expanded = np.expand_dims(image_np, axis=0)

                (boxes, scores, classes) = sess.run(
                    [detection_boxes, detection_scores, detection_classes],
                    feed_dict={image_tensor: image_np_expanded})

                boxes_squeeze = np.squeeze(boxes)
                classes_squeeze = np.squeeze(classes)
                scores_squeeze = np.squeeze(scores)

                boxstr = ''
                for i in range(len(classes_squeeze)):
                    # if (classes_squeeze[i] == 3 or classes_squeeze[i] == 8) and scores_squeeze[i] > 0.85:
                    # if classes_squeeze[i] == 3 or classes_squeeze[i] == 6 or classes_squeeze[i] == 8 :
                    if scores_squeeze[i] > threhold:
                        ymin = int(round(boxes_squeeze[i][0] * height))
                        xmin = int(round(boxes_squeeze[i][1] * width))
                        h = int(round((boxes_squeeze[i][2] - boxes_squeeze[i][0]) * height))
                        w = int(round((boxes_squeeze[i][3] - boxes_squeeze[i][1]) * width))
                        boxstr = boxstr + str(xmin) + "_" + str(ymin) + "_" + str(w) + "_" + str(h) + ';'

                # logging.warning('%s', boxstr)
                if index % 100 == 0:
                    logging.info('%s', index)
                    logging.info('%s', imgname)
                    #print(index)
                    #print(imgname)

                jpgs.append(str(imgname, encoding = "utf-8"))
                boxstrs.append(boxstr[:-1])

            coord.request_stop()
            coord.join(threads)

    jpgser = pd.Series(data=jpgs, name='name')
    boxstrser = pd.Series(data=boxstrs, name='coordinate')
    df = pd.concat([jpgser, boxstrser], axis=1)
    df.to_csv(export_file,index=False)

