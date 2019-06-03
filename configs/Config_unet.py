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

from trixi.util import Config


def get_config():
    # Set your own path, if needed.
    data_root_dir = os.path.abspath('data')  # The path where the downloaded dataset is stored.

    c = Config(
        update_from_argv=True,

        # Train parameters
        num_classes=2,
        in_channels=1,
        batch_size=16,
        patch_size=256,
        n_epochs=100,
        learning_rate=0.00005,
        fold=0,  # The 'splits.pkl' may contain multiple folds. Here we choose which one we want to use.

        device="cuda",  # 'cuda' is the default CUDA device, you can use also 'cpu'. For more information, see https://pytorch.org/docs/stable/notes/cuda.html

        # Logging parameters
        name='FinetuneSpleenforSpleen',
        author='kvw',  # Author of this project
        plot_freq=10,  # How often should stuff be shown in visdom
        append_rnd_string=False,
        start_visdom=False,

        do_instancenorm=True,  # Defines whether or not the UNet does a instance normalization in the contracting path
        do_load_checkpoint=True,
        checkpoint_filename='checkpoint_spleen_latest.pth.tar',
        checkpoint_dir='./models/',
        fine_tune='classy',

        # Adapt to your own path, if needed.
        download_data=False,
        google_drive_id='1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C',
        dataset_name='Task09_Spleen',
        base_dir=os.path.abspath('output_experiment'),  # Where to log the output of the experiment.

        data_root_dir=data_root_dir,  # The path where the downloaded dataset is stored.
        data_dir=os.path.join(data_root_dir, 'Task09_Spleen/preprocessed'),  # This is where your training and validation data is stored
        data_test_dir=os.path.join(data_root_dir, 'Task09_Spleen/preprocessed'),  # This is where your test data is stored

        split_dir=os.path.join(data_root_dir, 'Task09_Spleen'),  # This is where the 'splits.pkl' file is located, that holds your splits.

        # Testing
        visualize_segm = True

    )

    print(c)
    return c
