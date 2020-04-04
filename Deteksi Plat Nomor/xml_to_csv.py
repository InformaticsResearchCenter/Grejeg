
# untuk mengatur lokasi path
import os

# dalam script ini untuk membaca file xml dalam bentuk glob
import glob

# untuk mengolah data csv
import pandas as pd

# untuk mengolah data xml
import xml.etree.ElementTree as ET



def xml_to_csv(path):
    
    # inisialisasi list xml
    xml_list = []
    
    # untuk setiap file xml dalam folder annotations lakukan langkah berikut
    for xml_file in glob.glob(path + '/*.xml'):
        
        # baca file xml dari bagian terluar
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # ------------------------------------------------------------------

        print(path)
        print(root.find('filename').text)        


        # untuk setiap objek yg diinformasikan pada file xml lakukan langkah berikut
        for member in root.findall('object'):
            # dapatkan informasi nama kelas, ukuran image, dan lokasi box
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            # --------------------------------------------------------------

            # masukan informasi objek kedalam list xml
            xml_list.append(value)
            # --------------------------------------------------------------            

        # ------------------------------------------------------------------

    # ----------------------------------------------------------------------

    # mendefinisikan column untuk file csv yg akan di generate
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    
    # konversi list xml kedalam format table menggunakan pandas
    xml_df = pd.DataFrame(xml_list, columns=column_name)


    return xml_df


def main():
    for directory in ['train', 'test']:

        # mendapatkan list lokasi image (train/test)
        image_path = os.path.join(os.getcwd(), 'annotations/{}'.format(directory))
        
        # konversi informasi objeck dari file xml & image, 
        # untuk dicatat kedalam bentuk tabel yg nantinya akan disimpan dalam bentuk csv
        xml_df = xml_to_csv(image_path)

        # lakukan konversi kedalam csv lalu simpan
        xml_df.to_csv('data/{}_labels.csv'.format(directory), index=None)

        # notif proses konversi xml to csv berhasil
        print('Successfully converted xml to csv.')

#untuk menjalankan program
main()