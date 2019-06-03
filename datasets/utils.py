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

import numpy as np
from batchgenerators.augmentations.crop_and_pad_augmentations import crop


def reshape_old(orig_img, append_value=-1024, new_shape=(512, 512, 512)):
    reshaped_image = np.zeros(new_shape)
    reshaped_image[...] = append_value
    x_offset = 0
    y_offset = 0  # (new_shape[1] - orig_img.shape[1]) // 2
    z_offset = 0  # (new_shape[2] - orig_img.shape[2]) // 2

    reshaped_image[x_offset:orig_img.shape[0]+x_offset, y_offset:orig_img.shape[1]+y_offset, z_offset:orig_img.shape[2]+z_offset] = orig_img
    # insert temp_img.min() as background value

    return reshaped_image


def reshape(image_seg, crop_size):
    # image_seg is numpy array shape [2, #slices, y, x]: for dim0, index 0: image and index 1: seg labels
    #               this needs to be separated for the crop function.
    #               But crop function expects images and segs (see below) to have shape [batch, #slices, y, x]
    #               I am not using batches here so that's why i insert dummy first dim.
    # crop_size: integer (assuming new size is 2**n and width=height
    images = image_seg[0][np.newaxis]
    segs = image_seg[1][np.newaxis]

    data_cropped, segs_cropped = crop(images, seg=segs, crop_size=crop_size, margins=(0, 0, 0), crop_type="center",
                                      pad_mode='constant', pad_kwargs={'constant_values': 0},
                                      pad_mode_seg='constant', pad_kwargs_seg={'constant_values': 0})
    # squeeze out again dummy dims
    return np.stack((np.squeeze(data_cropped), np.squeeze(segs_cropped)))