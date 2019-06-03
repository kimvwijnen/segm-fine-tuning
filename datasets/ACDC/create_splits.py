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
from datasets.ACDC.io_tools.utilities import split_acdc_dataset


def create_splits(output_dir, image_dir):
    npy_files = subfiles(image_dir, suffix=".npy", join=False)

    splits = []
    # ACDC (total=100) uses 4 folds train=75 and val/test=25

    for fold_id in range(0, 4):
        train_pat_ids, val_pat_ids = split_acdc_dataset(fold_id, num_of_pats=100, max_per_set=None)
        trainset = []
        valset = []
        for fname in npy_files:
            # filenames have format "patient<xxx>_frame<xx>.npz. We split patient id and check whether it is
            # in train or val set
            patient_id = int(fname.split("_")[0].strip("patient"))
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
