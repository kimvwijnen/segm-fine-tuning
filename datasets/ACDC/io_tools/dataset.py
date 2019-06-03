import nibabel as nib
import numpy as np
import os
import copy

from datasets.ACDC.io_tools.utilities import split_acdc_dataset, apply_2d_zoom_3d
from datasets.ACDC.settings import acdc_settings


def resample2d(data_item, debug_info=None):
    # spacing has shape [#slices, IH, IW]. We resample to 1.4mm x 1.4mm for the last 2 dimensions
    spacing = data_item['spacing']
    if len(spacing) == 3:
        new_spacing = (tuple((spacing[0],)) + acdc_settings.voxel_spacing)
    elif len(spacing) == 4:
        new_spacing = (tuple((spacing[1],)) + acdc_settings.voxel_spacing)
    else:
        raise ValueError("ERROR - resample2d - spacing not supported ", spacing)

    for item in ['image_ed', 'image_es']:
        data_item[item] = apply_2d_zoom_3d(data_item[item], spacing, do_blur=True,
                                                        new_spacing=acdc_settings.voxel_spacing)

    for item in ['label_ed', 'label_es']:
        mycopy = copy.deepcopy(data_item[item])
        data_item[item] = apply_2d_zoom_3d(data_item[item], spacing, order=0, do_blur=False,
                                                        as_type=np.int, new_spacing=acdc_settings.voxel_spacing)
        for z in range(mycopy.shape[0]):
            if not np.all(np.unique(data_item[item][z, :, :]) == np.unique(mycopy[z, :, :])):
                print("WARNING - slice {} - unique labels not anymore the same! ".format(z) + debug_info)
    data_item['original_spacing'] = spacing
    data_item['spacing'] = new_spacing
    return data_item


class ACDCImageEDES(object):
    def __init__(self, number, root_dir=acdc_settings.data_path, scale_intensities=True):
        """
        IMPORTANT: After loading: numpy array gets reshaped to [z, y, x] z=#slices

        :param number: patient id (1...100)
        :param root_dir: data root dir
        :param scale_intensities: boolean
        """

        self._number = number
        self.patient_id = "patient{:03d}".format(number)
        self._path = os.path.join(root_dir, 'patient{:03d}'.format(number))
        self.info()
        frame_id_ed, frame_id_es = int(self.info()['ED']), int(self.info()['ES'])
        self._img_fname_ed = os.path.join(self._path, 'patient{:03d}_frame{:02d}.nii.gz'.format(self._number,
                                                                                                frame_id_ed))
        self._check_file_exists(self._img_fname_ed)
        self._img_fname_es = os.path.join(self._path, 'patient{:03d}_frame{:02d}.nii.gz'.format(self._number,
                                                                                                frame_id_es))
        self._check_file_exists(self._img_fname_es)
        self._lbl_fname_ed = self._img_fname_ed.replace(".nii.gz",
                                                        "_gt.nii.gz")
        self._check_file_exists(self._lbl_fname_ed)
        self._lbl_fname_es = self._img_fname_es.replace(".nii.gz",
                                                        "_gt.nii.gz")
        self._check_file_exists(self._lbl_fname_es)
        self._image = {'image_ed': None, 'image_es': None}
        self._scale_intensities = scale_intensities
        self.frame_id_ed = frame_id_ed
        self.frame_id_es = frame_id_es

    @staticmethod
    def _check_file_exists(filename):
        if not os.path.isfile(filename):
            raise FileExistsError("ERROR - ARVCEDESImage - file does not exist {}".format(filename))

    def voxel_spacing(self):
        # important we reverse voxel spacing AND shape because after loading data we reshape to [z, y, x]
        # whereas NIFTI images/labels are stored in [x, y, z]
        return self._image['image_ed'].header.get_zooms()[::-1]

    def shape(self):
        return self._image['image_ed'].header.get_data_shape()[::-1]

    def data(self):
        try:
            self._img_data, self._lbl_data
        except AttributeError:
            self._img_data = {'image_ed': None, 'image_es': None}
            self._image = {'image_ed': nib.load(self._img_fname_ed), 'image_es': nib.load(self._img_fname_es)}
            self._lbl_data = {'label_ed': nib.load(self._lbl_fname_ed), 'label_es': nib.load(self._lbl_fname_es)}
            for ikey in self._image.keys():
                # IMPORTANT: numpy array gets reshaped to [z, y, x] z=#slices
                data = self._image[ikey].get_data(caching='fill').transpose(2, 1, 0)
                if self._scale_intensities:
                    data = self._rescale_intensities_per_slice(data)

                self._img_data[ikey] = data
            for ikey in self._lbl_data.keys():
                # also need to reshape labels to [z, y, x]
                self._lbl_data[ikey] = self._lbl_data[ikey].get_data(caching='fill').transpose(2, 1, 0)

        finally:
            return self._img_data, self._lbl_data

    def get_item(self):
        _ = self.data()
        es_apex_base_slices = self._determine_apex_base_slices(self._lbl_data['label_es'])
        ed_apex_base_slices = self._determine_apex_base_slices(self._lbl_data['label_ed'])
        return {'image_ed': self._img_data['image_ed'], 'label_ed': self._lbl_data['label_ed'],
                'image_es': self._img_data['image_es'], 'label_es': self._lbl_data['label_es'],
                'spacing': self.voxel_spacing(), 'origin': None, 'frame_id_ed': self.frame_id_ed,
                'patient_id': self.patient_id, 'patid': self._number, 'num_of_slices': self.shape()[0],
                'frame_id_es': self.frame_id_es, 'type_extra_input': None, 'info': self._info,
                'apex_base_es': es_apex_base_slices, 'apex_base_ed': ed_apex_base_slices}

    @staticmethod
    def _rescale_intensities(img_data, percentile=acdc_settings.int_percentiles):
        min_val, max_val = np.percentile(img_data, percentile)
        return ((img_data.astype(float) - min_val) / (max_val - min_val)).clip(0, 1)

    @staticmethod
    def _rescale_intensities_per_slice(img_data, percentile=acdc_settings.int_percentiles):
        min_val, max_val = np.percentile(img_data, percentile, axis=(1, 2), keepdims=True)
        return ((img_data.astype(float) - min_val) / (max_val - min_val)).clip(0, 1)

    @staticmethod
    def _rescale_intensities_jelmer(img_data, percentile=acdc_settings.int_percentiles):
        min_val, max_val = np.percentile(img_data, percentile)
        data = (img_data.astype(float) - min_val) / (max_val - min_val)
        data -= 0.5
        min_val, max_val = np.percentile(data, percentile)
        return ((data - min_val) / (max_val - min_val)).clip(0, 1)

    @staticmethod
    def _determine_apex_base_slices(labels):
        slice_ab = {'A': None, 'B': None}
        # Note: low-slice number => most basal slices / high-slice number => most apex slice
        # Note: assuming labels has one bg-class indicated as 0-label and shape [z, y, x]
        slice_ids = np.arange(labels.shape[0])
        # IMPORTANT: we sum over x, y and than check whether we'have a slice that has ZERO labels. So if
        # np.any() == True, this means there is a slice without labels.
        binary_mask = (np.sum(labels, axis=(1, 2)) == 0).astype(np.bool)
        if np.any(binary_mask):
            # we have slices (apex/base) that do not contain any labels. We assume that this can only happen
            # in the first or last slices e.g. [1, 1, 0, 0, 0, 0] so first 2 slice do not contain any labels
            slices_with_labels = slice_ids[binary_mask != 1]
            slice_ab['B'], slice_ab['A'] = int(min(slices_with_labels)), int(max(slices_with_labels))
        else:
            # all slices contain labels. We simply assume slice-idx=0 --> base and slice-idx = max#slice --> apex
            slice_ab['B'], slice_ab['A'] = int(min(slice_ids)), int(max(slice_ids))
        return slice_ab

    @staticmethod
    def _normalize(img_data):
        return (img_data - np.mean(img_data)) / np.std(img_data)

    def info(self):
        try:
            self._info
        except AttributeError:
            self._info = dict()
            fname = os.path.join(self._path, 'Info.cfg')
            with open(fname, 'rU') as f:
                for l in f:
                    k, v = l.split(':')
                    self._info[k.strip()] = v.strip()
        finally:
            return self._info


if __name__ == '__main__':

    pass
