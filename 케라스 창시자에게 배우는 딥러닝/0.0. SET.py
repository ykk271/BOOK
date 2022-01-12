# 그래픽카드 확인
from tensorflow.python.client import device_lib
device_lib.list_local_devices()


# 경고메세지 해결
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
hallo = tf.constant('why?' )
