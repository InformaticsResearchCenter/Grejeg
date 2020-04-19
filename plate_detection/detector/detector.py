# Import library
import os
import cv2
import numpy as np
import tensorflow as tf
import sys


# untuk path folder sistem object detection
sys.path.append("..")

#class detector
class detector:
#inisialisasi var
    def __init__(self):
        #nyari lokasi folder detector
        folder_detector = 'detector'

        # Mengarahkan path ke frozen.pb yang berisikan model yang 
        # kan digunakan untuk sistem objek deteksi
        PATH_TO_CKPT = os.path.sep.join([folder_detector, 'frozen_inference_graph.pb'])

        # mengarahkan path ke labelmap.pbtxt untuk mengetahui jumlah class
        PATH_TO_LABELS = os.path.sep.join([folder_detector, 'labelmap.pbtxt'])

        # Jumlah kelas yang bisa diidentifikasi oleh objek detektor adalah 1 yaitu plate
        NUM_CLASSES = 1

        # Muat model Tensorflow ke dalam memori untuk persiapkan sessios tersorflow.
        # session ini yang dipanggil sama detector
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.Session(graph=self.detection_graph)

        # inisialisasi variabel untuk penyimpanan sementara saat testing
        # Tetapkan tensor input dan output (mis. Data) untuk classifier deteksi objek Input tensor adalah gambar
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

        # Output tensor adalah kotak deteksi, skor, dan kelas
        # Setiap kotak mewakili bagian dari gambar di mana objek tertentu terdeteksi
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')

        # Setiap skor mewakili tingkat kepercayaan untuk masing-masing objek.
        # Skor ditampilkan pada gambar hasil, bersama dengan label kelas.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')

        # Jumlah objek yang terdeteksi
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def detect_plate(self, frame):

        image_expanded = np.expand_dims(frame, axis=0)

        # Disini deteksi sebenarnya dilakukan dengan menjalankan model dengan gambar sebagai input
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_expanded})
        return (boxes, scores, classes, num, None)





