
import os
import socket


class ACDCSettings(object):

    def __init__(self):

        if socket.gethostname() == "qiaubuntu" or socket.gethostname() == "toologic-ubuntu2":
            self.data_path = os.path.expanduser("~/repository/data/ACDC/all_cardiac_phases/")
        else:
            self.data_path = os.path.expanduser("~/data/ACDC/all_cardiac_phases/")
        self.output_dir_preprocessing = "ACDC/preprocessed"
        self.number_of_patients = 100
        self.max_number_of_patients = 20
        self.voxel_spacing = (1.4, 1.4)
        self.int_percentiles = (5, 99)
        # padding to left and right of the image in order to reach the final image size for classification
        self.patch_size = 256
        # class labels
        self.num_of_tissue_classes = int(4)
        self.class_lbl_background = int(0)
        self.class_lbl_RV = int(1)
        self.class_lbl_myo = int(2)
        self.class_lbl_LV = int(3)


acdc_settings = ACDCSettings()

