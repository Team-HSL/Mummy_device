import serial
from serial.tools import list_ports
# import sys
import io
import numpy as np
import cv2

ports = list_ports.comports()

ser = serial.Serial()
ser.baudrate = 9600
print(ports)
# ser.port = ports[1].device
ser.port = ports[0].device
ser.open()

cv2.namedWindow("HSL GR-LYCHEE", cv2.WINDOW_AUTOSIZE)
while True:
    barray  = ser.read(150000)
    b_list = barray.split(b'\xff\xd8')
    cut_bynary = b_list[1].partition(b'\xff\xd9')
    cut_bynary_modified = bytes().join([b'\xff\xd8',cut_bynary[0],b'\xff\xd9'])

    cv2.imshow("HSL GR-LYCHEE",cv2.imdecode(np.fromstring(cut_bynary_modified,dtype="uint8"), -1))

    cv2.waitKey(33)
ser.close()