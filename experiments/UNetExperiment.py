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
import pickle
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from datasets.two_dim.NumpyDataLoader import NumpyDataSet
from trixi.experiment.pytorchexperiment import PytorchExperiment

from networks.RecursiveUNet import UNet
from loss_functions.dice_loss import SoftDiceLoss

from torchvision.utils import save_image

from trixi.util import name_and_iter_to_filename
from networks.utilities import unfreeze_block_parameters


class LabelTensorToColor(object):
   def __call__(self, label, nr_classes):
       class_color = [
           (0, 0, 0),
           (128, 0, 0),
           (0, 0, 128),
           (0, 128, 128),
           (128, 128, 0),
           (128, 0, 128),
           (255, 255, 51),
           (102, 255, 255),
           (102, 255, 102),
       ]

       label = label.squeeze()
       colored_label = torch.zeros(3, label.size(0), label.size(1)).byte()
       for i in range(nr_classes):
           mask = label.eq(i)
           for j in range(3):
               colored_label[j].masked_fill_(mask, class_color[i][j])

       return colored_label


def segm_visualization(mr_data, mr_target, pred_argmax, color_class_converter, config):
    # Rescale data
    data = (mr_data + mr_data.min()) / mr_data.max() * 256
    data = data.type(torch.uint8)

    # Make classes color
    mr_target = mr_target.cpu()
    target_list = []
    for i in range(mr_data.size()[0]):
        target_list.append(color_class_converter(mr_target[i], config.num_classes))
    target = torch.stack(target_list)

    # Same color as target
    pred_argmax = pred_argmax.cpu()
    pred_list = []
    for i in range(mr_data.size()[0]):
        pred_list.append(color_class_converter(pred_argmax[i], config.num_classes))
    pred = torch.stack(pred_list)

    save_image(torch.cat([data.repeat(1, 3, 1, 1).cpu(), target, pred]), 'output_experiment/' + config.name + '_data_target_prediction.png', nrow=8)


class UNetExperiment(PytorchExperiment):
    """
    The UnetExperiment is inherited from the PytorchExperiment. It implements the basic life cycle for a segmentation task with UNet(https://arxiv.org/abs/1505.04597).
    It is optimized to work with the provided NumpyDataLoader.

    The basic life cycle of a UnetExperiment is the same s PytorchExperiment:

        setup()
        (--> Automatically restore values if a previous checkpoint is given)
        prepare()

        for epoch in n_epochs:
            train()
            validate()
            (--> save current checkpoint)

        end()
    """

    def setup(self):
        pkl_dir = self.config.split_dir
        with open(os.path.join(pkl_dir, "splits.pkl"), 'rb') as f:
            splits = pickle.load(f)

        tr_keys = splits[self.config.fold]['train']
        val_keys = splits[self.config.fold]['val']
        test_keys = splits[self.config.fold]['test']

        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")

        self.train_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size,
                                              keys=tr_keys)
        self.val_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size,
                                            keys=val_keys, mode="val", do_reshuffle=False)
        self.test_data_loader = NumpyDataSet(self.config.data_test_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size,
                                             keys=test_keys, mode="test", do_reshuffle=False)
        self.model = UNet(num_classes=self.config.num_classes, in_channels=self.config.in_channels)

        self.model.to(self.device)

        # We use a combination of DICE-loss and CE-Loss in this example.
        # This proved good in the medical segmentation decathlon.
        self.dice_loss = SoftDiceLoss(batch_dice=True)  # Softmax for DICE Loss!
        self.ce_loss = torch.nn.CrossEntropyLoss()  # No softmax for CE Loss -> is implemented in torch!

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        # If directory for checkpoint is provided, we load it.
        if self.config.do_load_checkpoint:
            if self.config.checkpoint_dir == '':
                print('checkpoint_dir is empty, please provide directory to load checkpoint.')
            else:
                self.load_checkpoint(name=self.config.checkpoint_filename, save_types=("model"), path=self.config.checkpoint_dir)

            if self.config.fine_tune=='classy':
                unfreeze_block_parameters(model=self.model, block_names=self.config.block_names,
                                          block_numbers=self.config.block_numbers)

        self.save_checkpoint(name="checkpoint_start")
        self.elog.print('Experiment set up.')


    # overloaded method from the base class PytorchExperiment
    def load_checkpoint(self,
                        name="checkpoint",
                        save_types=("model", "optimizer", "simple", "th_vars", "results"),
                        n_iter=None,
                        iter_format="{:05d}",
                        prefix=False,
                        path=None):
        """
        Loads a checkpoint and restores the experiment.
        Make sure you have your torch stuff already on the right devices beforehand,
        otherwise this could lead to errors e.g. when making a optimizer step
        (and for some reason the Adam states are not already on the GPU:
        https://discuss.pytorch.org/t/loading-a-saved-model-for-continue-training/17244/3 )
        Args:
            name (str): The name of the checkpoint file
            save_types (list or tuple): What kind of member variables should be loaded? Choices are:
                "model" <-- Pytorch models,
                "optimizer" <-- Optimizers,
                "simple" <-- Simple python variables (basic types and lists/tuples),
                "th_vars" <-- torch tensors,
                "results" <-- The result dict
            n_iter (int): Number of iterations. Together with the name, defined by the iter_format,
                a file name will be created and searched for.
            iter_format (str): Defines how the name and the n_iter will be combined.
            prefix (bool): If True, the formatted n_iter will be prepended, otherwise appended.
            path (str): If no path is given then it will take the current experiment dir and formatted
                name, otherwise it will simply use the path and the formatted name to define the
                checkpoint file.
        """
        if self.elog is None:
            return

        model_dict = {}
        optimizer_dict = {}
        simple_dict = {}
        th_vars_dict = {}
        results_dict = {}

        if "model" in save_types:
            model_dict = self.get_pytorch_modules()
        if "optimizer" in save_types:
            optimizer_dict = self.get_pytorch_optimizers()
        if "simple" in save_types:
            simple_dict = self.get_simple_variables()
        if "th_vars" in save_types:
            th_vars_dict = self.get_pytorch_variables()
        if "results" in save_types:
            results_dict = {"results": self.results}

        checkpoint_dict = {**model_dict, **optimizer_dict, **simple_dict, **th_vars_dict, **results_dict}

        if n_iter is not None:
            name = name_and_iter_to_filename(name,
                                             n_iter,
                                             ".pth.tar",
                                             iter_format=iter_format,
                                             prefix=prefix)

        # Jorg Begin
        if self.config.dont_load_lastlayer:
            exclude_layer_dict = {'model': ['model.model.5.weight', 'model.model.5.bias']}
        else:
            exclude_layer_dict = {}
        # Jorg End

        if path is None:
            restore_dict = self.elog.load_checkpoint(name=name, **checkpoint_dict)
        else:
            checkpoint_path = os.path.join(path, name)
            if checkpoint_path.endswith("/"):
                checkpoint_path = checkpoint_path[:-1]
            restore_dict = self.elog.load_checkpoint_static(checkpoint_file=checkpoint_path,
                                                            exclude_layer_dict=exclude_layer_dict, **checkpoint_dict)

        self.update_attributes(restore_dict)


    def train(self, epoch):
        self.elog.print('=====TRAIN=====')
        self.model.train()

        data = None
        batch_counter = 0
        for data_batch in self.train_data_loader:

            self.optimizer.zero_grad()

            # Shape of data_batch = [1, b, c, w, h]
            # Desired shape = [b, c, w, h]
            # Move data and target to the GPU
            data = data_batch['data'][0].float().to(self.device)
            target = data_batch['seg'][0].long().to(self.device)

            pred = self.model(data)
            pred_softmax = F.softmax(pred, dim=1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.

            loss = self.dice_loss(pred_softmax, target.squeeze()) + self.ce_loss(pred, target.squeeze())


            loss.backward()
            self.optimizer.step()

            # Some logging and plotting
            if (batch_counter % self.config.plot_freq) == 0:
                self.elog.print('Epoch: {0} Loss: {1:.4f}'.format(self._epoch_idx, loss))

                self.add_result(value=loss.item(), name='Train_Loss', tag='Loss', counter=epoch + (batch_counter / self.train_data_loader.data_loader.num_batches))

                self.clog.show_image_grid(data.float().cpu(), name="data", normalize=True, scale_each=True, n_iter=epoch)
                self.clog.show_image_grid(target.float().cpu(), name="mask", title="Mask", n_iter=epoch)
                self.clog.show_image_grid(torch.argmax(pred.cpu(), dim=1, keepdim=True), name="unt_argmax", title="Unet", n_iter=epoch)
                self.clog.show_image_grid(pred.cpu()[:, 1:2, ], name="unt", normalize=True, scale_each=True, n_iter=epoch)

            batch_counter += 1

        assert data is not None, 'data is None. Please check if your dataloader works properly'

    def validate(self, epoch):
        self.elog.print('VALIDATE')
        self.model.eval()

        data = None
        loss_list = []

        with torch.no_grad():
            for data_batch in self.val_data_loader:
                data = data_batch['data'][0].float().to(self.device)
                target = data_batch['seg'][0].long().to(self.device)

                pred = self.model(data)
                pred_softmax = F.softmax(pred, dim=1)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.

                loss = self.dice_loss(pred_softmax, target.squeeze()) + self.ce_loss(pred, target.squeeze())
                loss_list.append(loss.item())

        assert data is not None, 'data is None. Please check if your dataloader works properly'
        self.scheduler.step(np.mean(loss_list))

        self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, np.mean(loss_list)))

        self.add_result(value=np.mean(loss_list), name='Val_Loss', tag='Loss', counter=epoch+1)

        self.clog.show_image_grid(data.float().cpu(), name="data_val", normalize=True, scale_each=True, n_iter=epoch)
        self.clog.show_image_grid(target.float().cpu(), name="mask_val", title="Mask", n_iter=epoch)
        self.clog.show_image_grid(torch.argmax(pred.data.cpu(), dim=1, keepdim=True), name="unt_argmax_val", title="Unet", n_iter=epoch)
        self.clog.show_image_grid(pred.data.cpu()[:, 1:2, ], name="unt_val", normalize=True, scale_each=True, n_iter=epoch)

    def test(self):
        from evaluation.evaluator import aggregate_scores, Evaluator
        from collections import defaultdict

        self.elog.print('=====TEST=====')
        self.model.eval()

        pred_dict = defaultdict(list)
        gt_dict = defaultdict(list)

        batch_counter = 0

        if self.config.visualize_segm:
            color_class_converter = LabelTensorToColor()

        with torch.no_grad():
            for data_batch in self.test_data_loader:
                print('testing...', batch_counter)
                batch_counter += 1

                # Get data_batches
                mr_data = data_batch['data'][0].float().to(self.device)
                mr_target = data_batch['seg'][0].float().to(self.device)

                pred = self.model(mr_data)
                pred_argmax = torch.argmax(pred.data.cpu(), dim=1, keepdim=True)

                fnames = data_batch['fnames']
                for i, fname in enumerate(fnames):
                    pred_dict[fname[0]].append(pred_argmax[i].detach().cpu().numpy())
                    gt_dict[fname[0]].append(mr_target[i].detach().cpu().numpy())

                if batch_counter == 35 and self.config.visualize_segm:
                    segm_visualization(mr_data, mr_target, pred_argmax, color_class_converter, self.config)

        test_ref_list = []
        for key in pred_dict.keys():
            test_ref_list.append((np.stack(pred_dict[key]), np.stack(gt_dict[key])))

        scores = aggregate_scores(test_ref_list, evaluator=Evaluator, json_author=self.config.author, json_task=self.config.name, json_name=self.config.name,
                                  json_output_file=self.elog.work_dir + "/{}_".format(self.config.author) + self.config.name + '.json')

        self.scores = scores

        print("Scores:\n", scores)
