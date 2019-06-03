import os
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from glob import glob
from configs.Config_unet import get_config
from datasets.neobrains.settings import neobrain_settings
from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from datasets.neobrains.io_tools.utilities import rescale_intensities


def save_preprocessed_data(combined_img_seg, output_dir, file_name):
    """

    :param data_item:
    :return: None

    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=False)

    np.save(os.path.join(output_dir, file_name), combined_img_seg.astype(np.float32))
    print(combined_img_seg.shape)
    print(file_name)


def reshape(images, labels, crop_size=256):
    # image_seg is numpy array shape [2, #slices, y, x]: for dim0, index 0: image and index 1: seg labels
    #               this needs to be separated for the crop function.
    #               But crop function expects images and segs (see below) to have shape [batch, #slices, y, x]
    #               I am not using batches here so that's why i insert dummy first dim.
    # new_shape is tuple (new_y, new_x)
    # this is rough but we use prior knowlegde and shorten the images/labels to better fit our needs
    # we know that all images have shape [50, 384, 384]
    # target tissue structures are situated more to the top, hence we shorten image to 320 and get rid of same out
    # margins that contain only background (25 to each side)
    images = images[:, :320, 25:images.shape[2] - 25]
    labels = labels[:, :320, 25:labels.shape[2] - 25]
    images = images[np.newaxis]
    segs = labels[np.newaxis]

    data_cropped, segs_cropped = crop(images, seg=segs, crop_size=crop_size, margins=(0, 0, 0), crop_type="center",
                                      pad_mode='constant', pad_kwargs={'constant_values': 0},
                                      pad_mode_seg='constant', pad_kwargs_seg={'constant_values': 0})
    # squeeze out again dummy dims
    return np.stack((np.squeeze(data_cropped), np.squeeze(segs_cropped)))


def preprocess(src_data_path):

    input_dir = os.path.join(src_data_path, "images/")

    search_path = input_dir + "*.nii"
    file_list = glob(search_path)
    if len(file_list) <= 0:
        raise IOError("ERROR - No files found for search mask {}".format(search_path))
    file_list.sort()
    for f in tqdm(file_list, desc="Preprocess all neobrain images"):
        neo_img = sitk.ReadImage(f)
        neo_img = sitk.GetArrayFromImage(neo_img).astype(np.float32)
        neo_img = rescale_intensities(neo_img, percentile=neobrain_settings.int_percentiles)
        neo_labels = sitk.GetArrayFromImage(sitk.ReadImage(f.replace("images", "labels"))).astype(np.int)
        print("Original shape ", neo_img.shape, np.min(neo_img), np.max(neo_img))
        combined_image_labels = reshape(neo_img, neo_labels, crop_size=neobrain_settings.patch_size)
        output_dir = os.path.join(src_data_path, neobrain_settings.output_dir_preprocessing)
        fname = os.path.join(output_dir, os.path.basename(f).replace(".nii", ".npy"))
        save_preprocessed_data(combined_image_labels, output_dir, fname)


if __name__ == '__main__':

    preprocess()