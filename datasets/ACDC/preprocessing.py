#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from tqdm import tqdm
from datasets.ACDC.io_tools.dataset import ACDCImageEDES, resample2d
from configs.Config_unet import get_config
from datasets.ACDC.settings import acdc_settings
from batchgenerators.augmentations.crop_and_pad_augmentations import crop


def do_stack_cardiac_phases(data_item):
    result_ed = np.stack((data_item["image_ed"], data_item["label_ed"]))
    result_es = np.stack((data_item["image_es"], data_item["label_es"]))
    return result_ed, result_es


def save_preprocessed_data(combined_img_seg, output_dir, file_name):
    """

    :param data_item:
    :return: None

    """
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=False)

    np.save(os.path.join(output_dir, file_name + '.npy'), combined_img_seg)
    print(combined_img_seg.shape)
    print(file_name)


def reshape(image_seg, crop_size):
    # image_seg is numpy array shape [2, #slices, y, x]: for dim0, index 0: image and index 1: seg labels
    #               this needs to be separated for the crop function.
    #               But crop function expects images and segs (see below) to have shape [batch, #slices, y, x]
    #               I am not using batches here so that's why i insert dummy first dim.
    # new_shape is tuple (new_y, new_x)
    images = image_seg[0][np.newaxis]
    segs = image_seg[1][np.newaxis]

    data_cropped, segs_cropped = crop(images, seg=segs, crop_size=crop_size, margins=(0, 0, 0), crop_type="center",
                                      pad_mode='constant', pad_kwargs={'constant_values': 0},
                                      pad_mode_seg='constant', pad_kwargs_seg={'constant_values': 0})
    # squeeze out again dummy dims
    return np.stack((np.squeeze(data_cropped), np.squeeze(segs_cropped)))


def acdc_preprocess(src_data_path, do_resample=True, do_reshape=True, limited_load=False, verbose=False):
    global_settings = get_config(dataset="ACDC")
    patient_ids = np.arange(1, acdc_settings.number_of_patients + 1)
    if verbose:
        print(">>>>>>>>>>>> INFO - ACDC Dataset - Loading from {}".format(src_data_path))
    data = {}
    if limited_load:
        # patient_ids = patient_ids[:acdc_settings.max_number_of_patients]
        patient_ids = [38, 85]

    mygenerator = tqdm(patient_ids, desc='Loading {} dataset'.format("Pre-processing"))

    for patid in mygenerator:
        acdc_edes_image = ACDCImageEDES(patid, root_dir=acdc_settings.data_path)
        data_item = acdc_edes_image.get_item()

        if do_resample:
            # print("Before resampling ", data_item['image_ed'].shape)
            data_item = resample2d(data_item,  debug_info=str(patid))
        # returns 2 numpy arrays. one for ED and one for ES. Both have shape [2, #slices, y, x]
        # where dim0 index=0 -> image and index=1 -> seg labels (multi labels {0..3})
        img_seg_ed, img_seg_es = do_stack_cardiac_phases(data_item)
        fname_ed = "{}_frame{:02d}".format(data_item['patient_id'], data_item["frame_id_ed"])
        fname_es = "{}_frame{:02d}".format(data_item['patient_id'], data_item["frame_id_es"])
        if do_reshape:
            # print("Before reshape ", data_item['image_ed'].shape)
            img_seg_ed = reshape(img_seg_ed, crop_size=acdc_settings.patch_size)
            img_seg_es = reshape(img_seg_es, crop_size=acdc_settings.patch_size)
        # REMEMBER: numpy arrays have shape [z, y, x] now
        output_dir = os.path.join(global_settings.data_root_dir, acdc_settings.output_dir_preprocessing)
        save_preprocessed_data(img_seg_ed, output_dir, fname_ed)
        save_preprocessed_data(img_seg_es, output_dir, fname_es)
        del data_item

    # print(total)
    # for i in range(classes):
    #     print(class_stats[i], class_stats[i]/total)
