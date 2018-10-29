# coding: utf-8

import time
import pickle

"""シリアル通信のためのモジュール"""
import serial
from serial.tools import list_ports
import io
import cv2
"""*************************"""

## ここから tensorflow object detection

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict  
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

# sys.path.append("..")
sys.path.append("./object_detection")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

# ## Object detection imports
# Here are the imports from the object detection module.

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


t_ = time.time()

# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# What model to download.の件は不要なので削除．

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join("object_detection",'data', 'mscoco_label_map.pbtxt')


# ## Download Modelの件は不要なので削除．

# モデルのダウンロード・読み込みをpickle読み込みに置換．
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def =  pickle.load(open("./object_detection/od_graph_def.pkl", "rb"))
    tf.import_graph_def(od_graph_def, name='')

print("モデル読み終わり {:.2f} sec".format(time.time() - t_))


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# ## Helper code
def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# # Detection
# For the sake of simplicity we will use only 2 images: # Detectionのくだりは削除．

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# def run_inference_for_single_image(image, graph): のくだりは削除．

print("with手前 {:.2f} sec".format(time.time() - t_))

# 画像表示ようフラグ．True なら画像を生成して表示する．
# SHOW_PIC = False
SHOW_PIC = True

ports = serial.tools.list_ports.comports()
ser = serial.Serial()
ser.baudrate = 9600
# ser.port = ports[1].device  # Winでの利用時
ser.port = ports[4].device   # Macでの利用時
ser.open()

 # ここからobject detection

# opencvのwindow生成
if SHOW_PIC:
    cv2.namedWindow("HSL_mummy", cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow("raw_pic", cv2.WINDOW_AUTOSIZE)

while True:
    barray  = ser.read(90000) #画像が欠けない中で最も小さい値を目指した．
    b_list = barray.split(b'\xff\xd8')
    cut_bytes = b_list[1].partition(b'\xff\xd9')
    fig_bytes = bytes().join([b'\xff\xd8',cut_bytes[0],b'\xff\xd9'])

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # for image_path in TEST_IMAGE_PATHS:
            image = Image.open(io.BytesIO(fig_bytes))
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            
            
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # plt.imshow(image_np_expanded)
            # plt.show()
            # モデルに入力するためのimage_tensorを構成．
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            
            # Actual detection.
            # ここで初めて画像をモデルに読み込む．
            (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

            # 物体検知のスコアが0.5以上のもののみ，そのクラスを表示する．
            # category_indexはクラスidとクラス名の対応を格納したdict．
            
            # dangerous_list = 危険物リスト．category_indexの一部を抜粋する形で定義する．
            dangerous_list = {49: {'id': 49, 'name': 'knife'}, 
                            87: {'id': 87, 'name': 'scissors'},
                            89: {'id': 89, 'name': 'hair drier'}
                            }
            

            # 物体検出を正解とみなすスコア閾値
            score_thresh = 0.3

            print("画像中の物体")
            isDanger = False
            for cl in classes[scores>score_thresh]:
                print(category_index[cl]["name"])
                if cl in dangerous_list:
                    isDanger = True
            
            if isDanger:
                print("WARNING!!")

            print("画像表示　直前 {:.2f} sec".format(time.time() - t_))
            frame = cv2.imdecode(np.fromstring(fig_bytes,dtype="uint8"), -1)
            if SHOW_PIC:
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8
                    ,max_boxes_to_draw=9
                    ,min_score_thresh=score_thresh
                )
                # plt.imshow(image_np)
                # plt.show()
                # plt.figure(figsize=IMAGE_SIZE, dpi=300) # dpiいじったら文字が読めるようになる
                # plt.imshow(image_np)
                # plt.axis("off")
                print("画像表示　後 {:.2f} sec".format(time.time() - t_))
                # plt.show()

                # 描画
                cv2.imshow("HSL_mummy",frame)
                cv2.imshow("raw_pic",cv2.imdecode(np.fromstring(fig_bytes,dtype="uint8"), -1))
                # cv2.imwrite("/Users/uchida/Documents/mummy_素材/figs/raw_{}.png".format(time.time()), cv2.imdecode(np.fromstring(fig_bytes,dtype="uint8"), -1))
                # cv2.imwrite("/Users/uchida/Documents/mummy_素材/figs/clf_{}.png".format(time.time()), frame)
                cv2.waitKey(33)
## ここまでtensorflowの設定
ser.close()