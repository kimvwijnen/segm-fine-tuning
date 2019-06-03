
import os
import socket


class NEOBrainsSettings(object):

    def __init__(self):

        if socket.gethostname() == "qiaubuntu" or socket.gethostname() == "toologic-ubuntu2":
            self.data_path = os.path.expanduser("~/repository/data/hackathon/neobrains/")
        else:
            self.data_path = os.path.expanduser("~/data/hackathon/neobrains/")
        self.output_dir_preprocessing = "preprocessed"
        self.number_of_patients = 10
        self.max_number_of_patients = 2
        self.voxel_spacing = (0.34, 0.34)
        self.int_percentiles = (1, 99)
        # padding to left and right of the image in order to reach the final image size for classification
        self.patch_size = 256
        # class labels
        self.num_of_tissue_classes = int(9)


neobrain_settings = NEOBrainsSettings()


