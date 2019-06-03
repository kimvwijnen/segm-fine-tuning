
from torch.optim import Adam


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


def unfreeze_block_parameters(model, block_names, block_numbers, verbose=False):
    """

    :param model: pytorch nn.Module object
    :param block_names: list of block names
    :param block_numbers: list of block numbers (same length as block_names)
    :param layer numbers: list of layer numbers
    :return:
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