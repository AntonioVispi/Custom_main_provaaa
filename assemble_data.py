from __future__ import division, print_function, absolute_import

# Functions which need the PIL

import numpy
import tempfile

from numpy import (amin, amax, ravel, asarray, arange, ones, newaxis,
                   transpose, iscomplexobj, uint8, issubdtype, array)

try:
    from PIL import Image, ImageFilter
except ImportError:
    import Image
    import ImageFilter




import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import os
from tqdm import tqdm

atri_dir = 'attribute_resized/'
mask_dir = 'segmentation_resized/'
output_dir = 'semantic_map/'



@numpy.deprecate(message="`toimage` is deprecated in SciPy 1.0.0, "
                         "and will be removed in 1.2.0.\n"
            "Use Pillow's ``Image.fromarray`` directly instead.")
def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    """Takes a numpy array and returns a PIL image.
    This function is only available if Python Imaging Library (PIL) is installed.
    The mode of the PIL image depends on the array shape and the `pal` and
    `mode` keywords.
    For 2-D arrays, if `pal` is a valid (N,3) byte-array giving the RGB values
    (from 0 to 255) then ``mode='P'``, otherwise ``mode='L'``, unless mode
    is given as 'F' or 'I' in which case a float and/or integer array is made.
    .. warning::
        This function uses `bytescale` under the hood to rescale images to use
        the full (0, 255) range if ``mode`` is one of ``None, 'L', 'P', 'l'``.
        It will also cast data for 2-D images to ``uint32`` for ``mode=None``
        (which is the default).
    Notes
    -----
    For 3-D arrays, the `channel_axis` argument tells which dimension of the
    array holds the channel data.
    For 3-D arrays if one of the dimensions is 3, the mode is 'RGB'
    by default or 'YCbCr' if selected.
    The numpy array must be either 2 dimensional or 3 dimensional.
    """
    data = asarray(arr)
    if iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(numpy.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(asarray(pal, dtype=uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (arange(0, 256, 1, dtype=uint8)[:, newaxis] *
                       ones((3,), dtype=uint8)[newaxis, :])
                image.putpalette(asarray(pal, dtype=uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = amin(ravel(data))
        if cmax is None:
            cmax = amax(ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(numpy.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = numpy.flatnonzero(asarray(shape) == 3)[0]
        else:
            ca = numpy.flatnonzero(asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image



file_name_arr = [] # [ISIC_00000, ISIC_000001, ISIC_000003, ...]
for file in glob.glob(atri_dir+'*.png'):
	temp = file.split('/')[-1].split('_')
	file_name = temp[0]+'_'+temp[1]
	if file_name not in file_name_arr:
	    file_name_arr.append(file_name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for family in tqdm(file_name_arr):
	# Create a zero filled base image
	for i, file in enumerate(glob.glob(atri_dir+family+'*.png')):
		# Read the image
		read_image = misc.imread(file, flatten=True)
		border_color = read_image[0,0]
		read_image[read_image == border_color] = 0
		read_image[read_image > 0] = 255
		read_image = np.int8(read_image/255)

		if i == 0:
			mask = misc.imread(mask_dir+family+'_segmentation.png', flatten=True)
			base_image = np.ones(read_image.shape, dtype=int) # Healthy Skin is 1
			border_mask_color = mask[0,0]
			base_image[mask == border_mask_color] = 0
			mask[mask == border_mask_color] = 0
			mask[mask > 0] = 255
			mask = np.int8(mask/255)
			base_image += mask # Common Lesion is 2

		type_file = file.split('/')[-1].split('_')[3]

		if type_file == 'pigment': # 3
			base_image += read_image
			if base_image[base_image > 3].any():
				base_image[base_image > 3] = 3
		elif type_file == 'negative': # 4
			base_image += read_image*2
			if base_image[base_image > 4].any():
				base_image[base_image > 4] = 4
		elif type_file.startswith('streaks'): # 5
			base_image += read_image*3
			if base_image[base_image > 5].any():
				base_image[base_image > 5] = 5
		elif type_file == 'milia': # 6
			base_image += read_image*4
			if base_image[base_image > 6].any():
				base_image[base_image > 6] = 6
		elif type_file.startswith('globules'): #7
			base_image += read_image*5
			if base_image[base_image > 7].any():
				base_image[base_image > 7] = 7
		else:
			print('ERROR: Invalid File Found!!!!')
	misc.toimage(base_image, cmin=0, cmax=255).save(output_dir+family+'_semantic.png')
