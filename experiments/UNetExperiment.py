import os
import pickle
from collections import defaultdict

import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from datasets.NumpyDataLoader import NumpyDataSet
from trixi.experiment.pytorchexperiment import PytorchExperiment
from networks.UNET import UNet
from loss_functions.dice_loss import SoftDiceLoss


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

        self.train_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size,
                                              keys=tr_keys)
        self.val_data_loader = NumpyDataSet(self.config.data_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size,
                                            keys=val_keys, mode="val", do_reshuffle=False)
        self.test_data_loader = NumpyDataSet(self.config.data_test_dir, target_size=self.config.patch_size, batch_size=self.config.batch_size,
                                             keys=test_keys, mode="test", do_reshuffle=False)
        self.model = UNet(num_classes=3, in_channels= 1, do_batchnorm=self.config.do_batchnorm)
        if self.config.use_cuda:
            self.model.cuda()

        # We use a combination of DICE-loss and CE-Loss in this example.
        # This proved good in the medical segmentation decathlon.
        self.dice_loss = SoftDiceLoss(batch_dice=True)  # Softmax für DICE Loss!
        self.ce_loss = torch.nn.CrossEntropyLoss()  # Kein Softmax für CE Loss -> ist in torch schon mit drin!

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        # If directory for checkpoint is provided, we load it.
        if self.config.do_load_checkpoint:
            if self.config.checkpoint_dir == '':
                print('checkpoint_dir is empty, please provide directory to load checkpoint.')
            else:
                self.load_checkpoint(name=self.config.checkpoint_dir, save_types=("model"))

        self.save_checkpoint(name="checkpoint_start")
        self.elog.print('Experiment set up.')

    def train(self, epoch):
        self.elog.print('=====TRAIN=====')
        self.model.train()

        batch_counter = 0
        for data_batch in self.train_data_loader:

            self.optimizer.zero_grad()

            # TODO
            # Shape of data_batch = [1, b, c, no_of_additional_slices, w, h]
            # Desired shape = [b, c, w, h]
            # Move data and target to the GPU
            data = data_batch['data'][0].float().cuda()
            target = data_batch['seg'][0].long().cuda()

            pred = self.model(data)
            pred_softmax = F.softmax(pred)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.

            loss = self.dice_loss(pred_softmax, target.squeeze()) + self.ce_loss(pred, target.squeeze())
            # loss = self.ce_loss(pred, target.squeeze())
            loss.backward()
            self.optimizer.step()

            # Some logging and plotting
            if (batch_counter % self.config.plot_freq) == 0:
                self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, loss))

                self.add_result(value=loss.item(), name='Train_Loss',   label='Loss', counter=epoch + (batch_counter / self.train_data_loader.data_loader.num_batches))

                self.clog.show_image_grid(data.float(), name="data", normalize=True, scale_each=True, n_iter=epoch)
                self.clog.show_image_grid(target.float(), name="mask", title="Mask", n_iter=epoch)
                self.clog.show_image_grid(torch.argmax(pred.data.cpu(), dim=1, keepdim=True), name="unt_argmax", title="Unet", n_iter=epoch)
                self.clog.show_image_grid(pred.data.cpu()[:, 1:2, ], name="unt", normalize=True, scale_each=True, n_iter=epoch)

            batch_counter += 1

    def validate(self, epoch):
        self.elog.print('VALIDATE')
        self.model.eval()

        loss_list = []

        with torch.no_grad():
            for data_batch in self.val_data_loader:
                data = data_batch['data'][0].float().cuda()
                target = data_batch['seg'][0].long().cuda()

                pred = self.model(data)
                pred_softmax = F.softmax(pred)  # We calculate a softmax, because our SoftDiceLoss expects that as an input. The CE-Loss does the softmax internally.

                loss = self.dice_loss(pred_softmax, target.squeeze()) + self.ce_loss(pred, target.squeeze())
                loss_list.append(loss.item())

        self.scheduler.step(np.mean(loss_list))

        self.elog.print('Epoch: %d Loss: %.4f' % (self._epoch_idx, np.mean(loss_list)))

        self.add_result(value=np.mean(loss_list), name='Val_Loss', label='Loss', counter=epoch+1)

        self.clog.show_image_grid(data.float(), name="data_val", normalize=True, scale_each=True, n_iter=epoch)
        self.clog.show_image_grid(target.float(), name="mask_val", title="Mask", n_iter=epoch)
        self.clog.show_image_grid(torch.argmax(pred.data.cpu(), dim=1, keepdim=True), name="unt_argmax_val", title="Unet", n_iter=epoch)
        self.clog.show_image_grid(pred.data.cpu()[:, 1:2, ], name="unt_val", normalize=True, scale_each=True, n_iter=epoch)

    def test(self):
        # TODO
        print('TODO: test() is not implemented yet.')
