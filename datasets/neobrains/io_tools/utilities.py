
import numpy as np
from datasets.neobrains.settings import neobrain_settings


def rescale_intensities(img_data, percentile=neobrain_settings.int_percentiles):
    min_val, max_val = np.percentile(img_data, percentile)
    return ((img_data.astype(np.float32) - min_val) / (max_val - min_val)).clip(0, 1)
