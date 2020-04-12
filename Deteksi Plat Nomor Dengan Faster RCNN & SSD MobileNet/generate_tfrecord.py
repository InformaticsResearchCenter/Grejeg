
# Import Library
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# untuk mengatur lokasi path
import os

# dalam script ini digunakan untuk konversi image kedalam format byte
import io

# untuk mengolah csv
import pandas as pd

# untuk generate tfrecord
import tensorflow as tf

# untuk mengolah/read data image
from PIL import Image

# dalam script ini digunakan untuk transformasi setiap value yg akan dijadikan tfrecord
from object_detection.utils import dataset_util

# untuk keperluan mengolah array, list, dict
from collections import namedtuple, OrderedDict


# parsing parameter dari command python (train/test, data csv, path output untuk tfrecord)
flags = tf.app.flags
flags.DEFINE_string('type', '', 'Type of CSV input (train/test)')
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS
# ---------------------------------------------------------------------


# Translasi nama kelas dalam bentuk text ke bentuk indeks integer
def class_text_to_int(row_label):
    if row_label == 'plate':
        return 1
    else:
        None
# ---------------------------------------------------------------------


# untuk memisahkan data csv yg terbaca kedalam beberapa kolom (delimiter)
def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]
# ---------------------------------------------------------------------


# Fungsi untuk menghasilkan tf record
def create_tf_example(group, path):
    # membaca file image
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    # -----------------------------------------------------------------    

    # mendapatkan ukuran image
    width, height = image.size

    # mendapatkan nama file dari image
    filename = group.filename.encode('utf8')
    image_format = b'jpg'

    # inisialisasi list
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []
    # -----------------------------------------------------------------    


    # untuk setiap box pada image lakukan berikut,..
    for index, row in group.object.iterrows():
        # konversi setiap koordinat dari box dari pixel ke domain 0 - 1
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        # -----------------------------------------------------------------

        # Menyimpan nama kelas kedalam list dalam bentuk text
        classes_text.append(row['class'].encode('utf8'))

        # Menyimpan nama kelas kedalam list dalam bentuk index integer
        classes.append(class_text_to_int(row['class']))
    

    # -----------------------------------------------------------------        

    # konversi ke tfrecord menggunakan fungsi tensorflow
    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    # -----------------------------------------------------------------


    return tf_example
# ---------------------------------------------------------------------



def main(_):
    # menyiapkan variable untuk menyimpan tf record kedalam bentuk file
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path) 

    # inisialisasi path, apakah untuk path data train atau data test
    path = os.path.join(os.getcwd(), 'images/{}'.format(FLAGS.type))

    # membaca file csv
    examples = pd.read_csv(FLAGS.csv_input)
    
    # memisahkan setiap kolom yg terbaca dari data csv
    grouped = split(examples, 'filename')

    # mengolah tfrecord lalu menyimpannya kedalam bentuk file .tfrecord
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString())
    writer.close()
    # -----------------------------------------------------------------

    # menampilkan lokasi file .tfrecord
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))
    # -----------------------------------------------------------------


if __name__ == '__main__':
    tf.app.run()