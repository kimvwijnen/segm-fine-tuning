from torchvision.utils import save_image

import torch
import numpy as np
from configs.Config_unet import get_config
from experiments.UNetExperiment import UNetExperiment
from PIL import Image

class_color = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 0, 128),
]


class LabelTensorToColor(object):
   def __call__(self, label):
       label = label.squeeze()
       colored_label = torch.zeros(3, label.size(0), label.size(1)).byte()
       for i, color in enumerate(class_color):
           mask = label.eq(i)
           for j in range(3):
               colored_label[j].masked_fill_(mask, color[j])

       return colored_label

color_class_converter = LabelTensorToColor()

# Load data
c = get_config()
exp = UNetExperiment(config=c, name=c.name, n_epochs=c.n_epochs,
                         seed=42, append_rnd_to_name=c.append_rnd_string, globs=globals())
exp.setup()
exp.test_data_loader.do_reshuffle = True

# Load checkpoint
checkpoint = torch.load('./output_experiment/20190529-181944_Basic_Unet/checkpoint/checkpoint_last.pth.tar')
exp.model.load_state_dict(checkpoint['model'])

exp.model.eval()

batch_counter = 0
with torch.no_grad():
    for data_batch in exp.test_data_loader:
        # Get data_batches
        mr_data = data_batch['data'][0].float().to(exp.device)
        mr_target = data_batch['seg'][0].float().to(exp.device)

        pred = exp.model(mr_data)
        pred_argmax = torch.argmax(pred.data.cpu(), dim=1, keepdim=True)

        # Rescale data
        data = (mr_data + mr_data.min()) / mr_data.max() * 256
        data = data.type(torch.uint8)

        # Make classes color
        mr_target = mr_target.cpu()
        target_list = []
        for i in range(mr_data.size()[0]):
            target_list.append(color_class_converter(mr_target[i]))
        target = torch.stack(target_list)

        # Same color as target
        pred_argmax = pred_argmax.cpu()
        pred_list = []
        for i in range(mr_data.size()[0]):
            pred_list.append(color_class_converter(pred_argmax[i]))
        pred = torch.stack(pred_list)

        save_image(torch.cat([data.repeat(1,3,1,1).cpu(), target, pred]), 'data_target_prediction.png', nrow=8)

        break