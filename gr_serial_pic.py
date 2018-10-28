# GR-LYCHEEからシリアル通信で受信したjpgバイト列を
# リアルタイムで描画し続ける．

"""シリアル通信のためのモジュール"""
import serial
from serial.tools import list_ports
import sys
import io
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2
"""*************************"""

ports = serial.tools.list_ports.comports()
ser = serial.Serial()
ser.baudrate = 9600
# ser.port = ports[1].device  # Winでの利用時
ser.port = ports[4].device   # Macでの利用時
ser.open()

# port番号が未特定の場合は，以下で全てのポートをprintして確認．
# Macでの利用時は ports[4] がGRのCDCカメラでした．
# Winなら ports[1] になりやすい．
# 
# for p in ports:
#     print(p)

cv2.namedWindow("Capture", cv2.WINDOW_AUTOSIZE)
while True:
    barray  = ser.read(90000) #画像が欠けない中で最も小さい値を目指した．
    b_list = barray.split(b'\xff\xd8')
    # print(len(b_list)) 
    cut_bytes = b_list[1].partition(b'\xff\xd9')
    fig_bytes = bytes().join([b'\xff\xd8',cut_bytes[0],b'\xff\xd9'])
    
    image = Image.open(fig_bytes)
    print(type(image))

    # print(fig_bytes)
    print("")
    # cv2.imshow("Capture",cv2.imdecode(np.fromstring(fig_bytes,dtype="uint8"), -1))
    # cv2.waitKey(33)
ser.close()