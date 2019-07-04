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

    dataset = 'atrial'
    # TODO dataset = 'atrial' (for training the atrial on 5 images)

    fine_tune = 'None'
    # TODO 'None' (no freezing, training whole network from scratch)

    dataset = 'brats'
    # TODO brats with 3 fine tuning options:

    # TODO fine_tune = 'None' (no freezing, training whole network)
    # TODO fine_tune = 'expanding_all' (for freezing contracting/left path, training expanding/right path)
    # TODO fine_tune = 'expanding_plus1' (for freezing the first bit of contracting path, training bottom part and expanding path)

    if dataset == 'atrial':
        checkpoint_dir = '' # leave empty to train from scratch for atrial segmentation on 5 samples
        checkpoint_filename = ''
        exp_name = 'train_from_scratch_heart'
    elif dataset == 'brats':
        checkpoint_dir = './output_experiments/[dir_atrial_exp]/checkpoint/' # TODO add name of directory with model
        checkpoint_filename = 'checkpoint_last.pth.tar'
        exp_name = 'brats_for_atrialsegm_finetune_' + 'all_layers' if fine_tune=='None' else fine_tune
    else:
        raise ValueError('No config settings for this dataset')

    c = get_config_heart(fine_tune_type=fine_tune,
                          exp_name=exp_name,
                          checkpoint_filename=checkpoint_filename,
                          checkpoint_dir=checkpoint_dir,
                          nr_train_samples=5)
    # training on 5 images (if want to use original split use train_samples=0 instead of train_samples=5

    print(c)
    return c


def get_config_heart(fine_tune_type='None', exp_name='', checkpoint_filename='', checkpoint_dir='', nr_train_samples=0):
    # Set your own path, if needed.
    data_root_dir = os.path.abspath('data')  # The path where the downloaded dataset is stored.

    c = Config(
        update_from_argv=True,

        # Train parameters
        num_classes=2,
        in_channels=1,
        batch_size=8,
        patch_size=256,
        n_epochs=60,
        learning_rate=0.0002,
        fold=0,  # The 'splits.pkl' may contain multiple folds. Here we choose which one we want to use.

        device="cuda",
        # 'cuda' is the default CUDA device, you can use also 'cpu'. For more information, see https://pytorch.org/docs/stable/notes/cuda.html

        # Logging parameters
        name=exp_name,
        author='maxi',  # Author of this project
        plot_freq=10,  # How often should stuff be shown in visdom
        append_rnd_string=False,
        start_visdom=False,

        do_instancenorm=True,  # Defines whether or not the UNet does a instance normalization in the contracting path
        do_load_checkpoint=True,
        checkpoint_filename=checkpoint_filename,
        checkpoint_dir=checkpoint_dir,
        fine_tune=fine_tune_type,

        # Adapt to your own path, if needed.
        download_data=False,
        google_drive_id='1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C',
        dataset_name='Task02_Heart',
        base_dir=os.path.abspath('output_experiment'),  # Where to log the output of the experiment.

        data_root_dir=data_root_dir,  # The path where the downloaded dataset is stored.
        data_dir=os.path.join(data_root_dir, 'Task02_Heart/preprocessed'),
        # This is where your training and validation data is stored
        data_test_dir=os.path.join(data_root_dir, 'Task02_Heart/preprocessed'),
        # This is where your test data is stored

        split_dir=os.path.join(data_root_dir, 'Task02_Heart'),
        # This is where the 'splits.pkl' file is located, that holds your splits.
        train_samples=nr_train_samples,
        # This is the amount of samples used in the train set. Use 0 for original split (1/2 train, 1/4 val, 1/4 test)

        # Testing
        visualize_segm=True

    )

    print(c)
    return c
