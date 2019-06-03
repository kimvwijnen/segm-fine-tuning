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

import pickle
from utilities.file_and_folder_operations import subfiles

import os
import numpy as np
import itertools


def split_dataset(fold_id, allpatnumbers, num_folds=3) -> tuple((list, list)):

    foldmask = np.tile(np.arange(num_folds)[::-1].repeat(1), 3)

    training_nums, validation_nums = allpatnumbers[foldmask != fold_id], allpatnumbers[foldmask == fold_id]
    return training_nums, validation_nums


def create_splits(output_dir, image_dir, num_folds=3):
    npy_files = subfiles(image_dir, suffix=".npy", join=False)
    allpatient_ids = np.array([int(p.split("_")[1].strip(".npy")) for p in npy_files])
    print(allpatient_ids)

    splits = []
    # Neo brains has 9 images/labels. We use 6 for training and 3 for testing

    for fold_id in range(0, num_folds):
        train_pat_ids, val_pat_ids = split_dataset(fold_id, allpatient_ids, num_folds=num_folds)
        trainset = []
        valset = []

        for fname in npy_files:
            # filenames have format "patient<xxx>_frame<xx>.npz. We split patient id and check whether it is
            # in train or val set
            patient_id = int(fname.split("_")[1].strip(".npy"))

            if patient_id in train_pat_ids:
                trainset.append(fname[:-4])
            else:
                valset.append(fname[:-4])
            split_dict = dict()

        print("INFO - train {} - test {}".format(len(trainset), len(valset)))
        split_dict['train'] = trainset
        split_dict['val'] = valset
        split_dict['test'] = valset

        splits.append(split_dict)

    with open(os.path.join(output_dir, 'splits.pkl'), 'wb') as f:
       pickle.dump(splits, f)
