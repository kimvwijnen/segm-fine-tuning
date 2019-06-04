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


def get_config(dataset="neobrains"):

    if dataset == "ACDC":
        c = get_config_acdc()
    elif dataset == "neobrains":
        c = get_config_neobrains()
    elif dataset == "heart":
        c = get_config_deca_heart()
    elif dataset == "spleen":
        c = get_config_deca_spleen()
    else:
        c = get_config_hippocampus()
    print(c)
    return c


def get_config_deca_spleen():
    raise NotImplementedError


def get_config_deca_heart():
    # Set your own path, if needed.
    # The path where the downloaded dataset is stored.
    data_root_dir = os.path.abspath(os.path.expanduser('~/data/hackathon'))

    c = Config(
        update_from_argv=True,

        # Train parameters
        num_classes=9,
        in_channels=1,
        batch_size=16,
        patch_size=256,
        n_epochs=100,
        learning_rate=0.0002,
        fold=0,  # The 'splits.pkl' may contain multiple folds. Here we choose which one we want to use.

        device="cuda",  # 'cuda' is the default CUDA device, you can use also 'cpu'. For more information, see https://pytorch.org/docs/stable/notes/cuda.html

        # Logging parameters
        name='Basic_Unet',
        author='kleina',  # Author of this project
        plot_freq=10,  # How often should stuff be shown in visdom
        append_rnd_string=False,
        # Visdom config
        visdom_server="http://seize",
        visdom_port=8030,
        start_visdom=False,

        do_instancenorm=True,  # Defines whether or not the UNet does a instance normalization in the contracting path
        do_load_checkpoint=True,
        checkpoint_dir='/home/jorg/models/hackathon/acdc_fold1/',
        checkpoint_filename="checkpoint_last.pth.tar",
        fine_tune='classy',
        block_names=['expanding'],
        block_numbers=[4],  # 1,2,3,4

        # Adapt to your own path, if needed.
        google_drive_id='',
        dataset_name='heart',
        base_dir=os.path.abspath('output_experiment'),  # Where to log the output of the experiment.

        data_root_dir=os.path.join(data_root_dir), # The path where the downloaded dataset is stored.
        data_dir=os.path.join(data_root_dir, 'neobrains/preprocessed'),  # This is where your training and validation data is stored
        data_test_dir=os.path.join(data_root_dir, 'neobrains/preprocessed'),  # This is where your test data is stored

        split_dir=os.path.join(data_root_dir, 'neobrains'),  # This is where the 'splits.pkl' file is located, that holds your splits.
    )

    return c


def get_config_neobrains():
    # Set your own path, if needed.
    # The path where the downloaded dataset is stored.
    data_root_dir = os.path.abspath(os.path.expanduser('~/data/hackathon'))

    c = Config(
        update_from_argv=True,

        # Train parameters
        num_classes=9,
        in_channels=1,
        batch_size=16,
        patch_size=256,
        n_epochs=100,
        learning_rate=0.0002,
        fold=0,  # The 'splits.pkl' may contain multiple folds. Here we choose which one we want to use.

        device="cuda",  # 'cuda' is the default CUDA device, you can use also 'cpu'. For more information, see https://pytorch.org/docs/stable/notes/cuda.html

        # Logging parameters
        name='Basic_Unet',
        author='kleina',  # Author of this project
        plot_freq=10,  # How often should stuff be shown in visdom
        append_rnd_string=False,
        # Visdom config
        visdom_server="http://seize",
        visdom_port=8330,
        start_visdom=False,

        do_instancenorm=True,  # Defines whether or not the UNet does a instance normalization in the contracting path
        do_load_checkpoint=True,
        checkpoint_dir='/home/jorg/models/hackathon/acdc_fold1/',
        checkpoint_filename="checkpoint_last.pth.tar",
        fine_tune='classy',
        block_names=['expanding'],
        block_numbers=[4],  # 1,2,3,4

        # Adapt to your own path, if needed.
        google_drive_id='',
        dataset_name='neobrains',
        base_dir=os.path.abspath('output_experiment'),  # Where to log the output of the experiment.

        data_root_dir=os.path.join(data_root_dir), # The path where the downloaded dataset is stored.
        data_dir=os.path.join(data_root_dir, 'neobrains/preprocessed'),  # This is where your training and validation data is stored
        data_test_dir=os.path.join(data_root_dir, 'neobrains/preprocessed'),  # This is where your test data is stored

        split_dir=os.path.join(data_root_dir, 'neobrains'),  # This is where the 'splits.pkl' file is located, that holds your splits.
    )

    return c


def get_config_acdc():
    # Set your own path, if needed.
    # The path where the downloaded dataset is stored.
    data_root_dir = os.path.abspath(os.path.expanduser('~/data/hackathon'))

    c = Config(
        update_from_argv=True,
        # Visdom config
        visdom_server="http://seize",
        visdom_port=8030,
        # Train parameters
        num_classes=4,   # acdc=4
        in_channels=1,
        batch_size=16,   # for acdc 16
        patch_size=256,
        n_epochs=10,
        learning_rate=0.0002,
        fold=1,  # The 'splits.pkl' may contain multiple folds. Here we choose which one we want to use.

        device="cuda",  # 'cuda' is the default CUDA device, you can use also 'cpu'. For more information, see https://pytorch.org/docs/stable/notes/cuda.html

        # Logging parameters
        name='Basic_Unet',
        author='kleina',  # Author of this project
        plot_freq=10,  # How often should stuff be shown in visdom
        append_rnd_string=False,
        start_visdom=False,

        do_instancenorm=True,  # Defines whether or not the UNet does a instance normalization in the contracting path
        do_load_checkpoint=True,
        checkpoint_dir='/home/jorg/models/hackathon/spleen_fold0/',
        checkpoint_filename="checkpoint_spleen_latest.pth.tar",
        fine_tune='classy',
        block_names=['expanding'],
        block_numbers=[4],  # 1,2,3,4

        # Adapt to your own path, if needed.
        google_drive_id='',
        dataset_name='ACDC',
        base_dir=os.path.abspath('output_experiment'),  # Where to log the output of the experiment.

        data_root_dir=data_root_dir,  # The path where the downloaded dataset is stored.
        data_dir=os.path.join(data_root_dir, 'ACDC/preprocessed'),  # This is where your training and validation data is stored
        data_test_dir=os.path.join(data_root_dir, 'ACDC/preprocessed'),  # This is where your test data is stored

        split_dir=os.path.join(data_root_dir, 'ACDC'),  # This is where the 'splits.pkl' file is located, that holds your splits.
    )

    return c


def get_config_hippocampus():
    # Set your own path, if needed.
    data_root_dir = os.path.abspath(os.path.expanduser('~/data/hackathon'))  # The path where the downloaded dataset is stored.

    c = Config(
        update_from_argv=True,
        # Visdom config
        visdom_server="http://seize",
        visdom_port=8030,
        # Train parameters
        num_classes=3,
        in_channels=1,
        batch_size=8,
        patch_size=64,
        n_epochs=10,
        learning_rate=0.0002,
        fold=0,  # The 'splits.pkl' may contain multiple folds. Here we choose which one we want to use.

        device="cuda",  # 'cuda' is the default CUDA device, you can use also 'cpu'. For more information, see https://pytorch.org/docs/stable/notes/cuda.html

        # Logging parameters
        name='Basic_Unet',
        author='kleina',  # Author of this project
        plot_freq=10,  # How often should stuff be shown in visdom
        append_rnd_string=False,
        start_visdom=False,

        do_instancenorm=True,  # Defines whether or not the UNet does a instance normalization in the contracting path
        do_load_checkpoint=True,
        checkpoint_dir='/home/jorg/models/hackathon/acdc_fold1/',
        checkpoint_filename="checkpoint_last.pth.tar",

        # Adapt to your own path, if needed.
        google_drive_id='1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C',
        dataset_name='Task04_Hippocampus',
        base_dir=os.path.abspath('output_experiment'),  # Where to log the output of the experiment.

        data_root_dir=data_root_dir,  # The path where the downloaded dataset is stored.
        data_dir=os.path.join(data_root_dir, 'Task04_Hippocampus/preprocessed'),  # This is where your training and validation data is stored
        data_test_dir=os.path.join(data_root_dir, 'Task04_Hippocampus/preprocessed'),  # This is where your test data is stored

        split_dir=os.path.join(data_root_dir, 'Task04_Hippocampus'),  # This is where the 'splits.pkl' file is located, that holds your splits.
    )
    return c