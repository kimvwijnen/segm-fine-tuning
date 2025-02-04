from torch.optim import Adam
import numpy as np

# dictionary has has key1=high level block {'contracting', 'inner', 'expanding'}
#                    key2=layer number, value is list with layer names: if you extend layer name with '.weight'
#                       or '.bias' then you end up with the parameter name
#                       e.g. model.2.model.3.model.3.model.3.model.2.0.weight (for inner block,
model_block_names = {'contracting':
                         {1: ['model.0.0',  'model.1.0'],
                          2: ['model.2.model.1.0',  'model.2.model.2.0'],
                          3: ['model.2.model.3.model.1.0', 'model.2.model.3.model.2.0'],
                          4: ['model.2.model.3.model.3.model.1.0', 'model.2.model.3.model.3.model.2.0']
                         },
                     'inner': {1: ['model.2.model.3.model.3.model.3.model.1.0',
                                   'model.2.model.3.model.3.model.3.model.2.0',
                                   'model.2.model.3.model.3.model.3.model.3']
                                },
                     'expanding':
                        { 1: ['model.2.model.3.model.3.model.4.0', 'model.2.model.3.model.3.model.5.0',
                              'model.2.model.3.model.3.model.6'],
                          2: ['model.2.model.3.model.4.0', 'model.2.model.3.model.5.0',
                              'model.2.model.3.model.6'],
                          3: ['model.2.model.4.0', 'model.2.model.5.0', 'model.2.model.6'],
                          4: ['model.3.0', 'model.4.0', 'model.5']
                        },
                     }

UNFREEZE_OPTIONS = {'expanding_all':
                      {'block_names': ['expanding'] * 4,
                       'block_numbers': np.arange(1, 5)
                      },
                    'expanding_plus1':
                      {'block_names': ['expanding'] * 4 + ['inner'] + ['contracting'],
                       'block_numbers': np.concatenate((np.arange(1, 5), np.array([1, 4])))
                      } # unfreezes expanding blok 1, 2, 3, 4; inner blok 1; contracting blok 4
                   }


def unfreeze_block_parameters(model, fine_tune_option, verbose=False):
    """
    :param model: pytorch nn.Module object
    :param fine_tune_option: one of the following
              'expanding_all': unfreeze complete expanding path
              'expanding_plus1': unfreeze complete expanding path plus contracting block 4 and inner block
    :return:
    example: only train (unfreeze) last 3 layers of U-net (classifier layers)
    from networks.utilities import unfreeze_block_parameters
    block_names = ['expanding']
    block_layers = [4]
block_names=['contracting', 'contracting', 'expanding', 'expanding'],
        block_numbers=[1, 2, 1, 2],  # 1,2,3,4
    """
    layer_names = []
    block_names = UNFREEZE_OPTIONS[fine_tune_option]['block_names']
    block_numbers = UNFREEZE_OPTIONS[fine_tune_option]['block_numbers']
    for block_name, block_number in zip(block_names, block_numbers):
        layer_names.extend(model_block_names[block_name][block_number])
    print('Layers that will be trained:')
    print(layer_names)
    for name, child in model.named_children():
        for param_name, param in child.named_parameters():
            param_name_stripped = (param_name.strip('.weight')).strip('.bias')
            if param_name_stripped in layer_names:
                if verbose:
                    print("Unfreeze {}".format(param_name))
                param.requires_grad = True
            else:
                param.requires_grad = False

def unfreeze_block_parametersv2(model, block_names, block_numbers, verbose=False):
    """
    :param model: pytorch nn.Module object
    :param block_names: list of block names
    :param block_numbers: list of block numbers (same length as block_names)
    :param layer numbers: list of layer numbers
    :return:
    example: only train (unfreeze) last 3 layers of U-net (classifier layers)
    from networks.utilities import unfreeze_block_parameters
    block_names = ['expanding']
    block_layers = [4]
    block_names=['contracting', 'contracting', 'expanding', 'expanding'],
                 block_numbers=[1, 2, 1, 2],  # 1,2,3,4
    unfreeze_block_parameters(model, block_names, block_layers, verbose=True)
    """
    layer_names = []
    for block_name, block_number in zip(block_names, block_numbers):
        layer_names.extend(model_block_names[block_name][block_number])
    print(layer_names)
    for name, child in model.named_children():
        for param_name, param in child.named_parameters():
            param_name_stripped = (param_name.strip('.weight')).strip('.bias')
            if param_name_stripped in layer_names:
                if verbose:
                    print("Unfreeze {}".format(param_name))
                param.requires_grad = True
            else:
                param.requires_grad = False


def configure_optimizer(model):

    optim = Adam(
        [
            {"params": model.fc.parameters(), "lr": 1e-3},
            {"params": model.agroupoflayer.parameters()},
            {"params": model.lastlayer.parameters(), "lr": 4e-2},
        ],
        lr=5e-4,
    )

    return optim