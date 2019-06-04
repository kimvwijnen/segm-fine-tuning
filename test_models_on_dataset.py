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

import matplotlib
matplotlib.use('Agg')

import os
from os.path import exists
from glob import glob
from datasets.example_dataset.create_splits import create_splits
from datasets.example_dataset.download_dataset import download_dataset
from datasets.example_dataset.preprocessing import preprocess_data
from experiments.UNetFineTune import UNetFineTune

from torchvision.utils import save_image

import torch
from configs.Config_unet import get_config
from experiments.UNetExperiment import UNetExperiment

import pandas as pd

# class_color = [
#     (0, 0, 0),
#     (128, 0, 0),
#     (0, 0, 128),
# ]
#
#
# class LabelTensorToColor(object):
#    def __call__(self, label):
#        label = label.squeeze()
#        colored_label = torch.zeros(3, label.size(0), label.size(1)).byte()
#        for i, color in enumerate(class_color):
#            mask = label.eq(i)
#            for j in range(3):
#                colored_label[j].masked_fill_(mask, color[j])
#
#        return colored_label
#
#
# def segm_visualization(mr_data, mr_target, pred_argmax, color_class_converter):
#     # Rescale data
#     data = (mr_data + mr_data.min()) / mr_data.max() * 256
#     data = data.type(torch.uint8)
#
#     # Make classes color
#     mr_target = mr_target.cpu()
#     target_list = []
#     for i in range(mr_data.size()[0]):
#         target_list.append(color_class_converter(mr_target[i]))
#     target = torch.stack(target_list)
#
#     # Same color as target
#     pred_argmax = pred_argmax.cpu()
#     pred_list = []
#     for i in range(mr_data.size()[0]):
#         pred_list.append(color_class_converter(pred_argmax[i]))
#     pred = torch.stack(pred_list)
#
#     save_image(torch.cat([data.repeat(1, 3, 1, 1).cpu(), target, pred]), 'data_target_prediction.png', nrow=8)
#

def run_on_dataset():

    #models = os.listdir('./models/')
    models = glob('./output_experiment/*/checkpoint/checkpoint_last.pth.tar')

    metric_list = ["Dice",
                   "Hausdorff Distance",
                   "Hausdorff Distance 95",
                   "Avg. Symmetric Surface Distance",
                   "Avg. Surface Distance"]

    summary = pd.DataFrame(columns=['modelname'] + metric_list)
    summary_mean = pd.DataFrame(columns=['modelname'] + metric_list)

    for model_name in models:
        # Load data
        exp = UNetExperiment(config=c, name=c.name, n_epochs=c.n_epochs,
                             seed=42, append_rnd_to_name=c.append_rnd_string, globs=globals())
        exp.setup()
        exp.test_data_loader.do_reshuffle = False

        # Load checkpoint
        checkpoint = torch.load(model_name)
        exp.model.load_state_dict(checkpoint['model'])

        # exp.model.eval() # done in UNetExperiment

        exp.run_test(setup=False)

        #TODO get metrics
        # select interesting ones, add to pandas dataframe

        json_scores = exp.scores

        result_dict = json_scores["results"] #{"all": results, "mean": results_mean}

        im_scores = result_dict['all']

        model_res = {'modelname': model_name}

        for nr, im in enumerate(im_scores):
            for cl in range(c.num_classes):
                for m in metric_list:
                    model_res['class_' + str(cl) + '_imagenr_' + str(nr) + '_' + m] = im[str(cl)][m]

        summary = summary.append(model_res, ignore_index=True)

        im_mean_scores = result_dict['mean']
        model_res_mean = {'modelname': model_name}

        for cl in range(c.num_classes):
            for m in metric_list:
                model_res_mean['class_' + str(cl) + '_' + m] = im_mean_scores[str(cl)][m]

        summary_mean = summary_mean.append(model_res_mean, ignore_index=True)

        summary.to_csv('./output_experiment/summary_ims' + c.dataset_name + '.csv')
        summary_mean.to_csv('./output_experiment/summary_mean_' + c.dataset_name + '.csv')


if __name__ == "__main__":

    c = get_config()

    print('Testing on dataset ' + c.dataset_name)

    run_on_dataset()



