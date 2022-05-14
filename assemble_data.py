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

def fromimage(im, flatten=False, mode=None):
    """
    Return a copy of a PIL image as a numpy array.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    im : PIL image
        Input image.
    flatten : bool
        If true, convert the output to grey-scale.
    mode : str, optional
        Mode to convert image to, e.g. ``'RGB'``.  See the Notes of the
        `imread` docstring for more details.
    Returns
    -------
    fromimage : ndarray
        The different colour bands/channels are stored in the
        third dimension, such that a grey-image is MxN, an
        RGB-image MxNx3 and an RGBA-image MxNx4.
    """
    if not Image.isImageType(im):
        raise TypeError("Input is not a PIL image.")

    if mode is not None:
        if mode != im.mode:
            im = im.convert(mode)
    elif im.mode == 'P':
        # Mode 'P' means there is an indexed "palette".  If we leave the mode
        # as 'P', then when we do `a = array(im)` below, `a` will be a 2-D
        # containing the indices into the palette, and not a 3-D array
        # containing the RGB or RGBA values.
        if 'transparency' in im.info:
            im = im.convert('RGBA')
        else:
            im = im.convert('RGB')

    if flatten:
        im = im.convert('F')
    elif im.mode == '1':
        # Workaround for crash in PIL. When im is 1-bit, the call array(im)
        # can cause a seg. fault, or generate garbage. See
        # https://github.com/scipy/scipy/issues/2138 and
        # https://github.com/python-pillow/Pillow/issues/350.
        #
        # This converts im from a 1-bit image to an 8-bit image.
        im = im.convert('L')

    a = array(im)
    return a



def imread(name, flatten=False, mode=None):
    """
    Read an image from a file as an array.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    name : str or file object
        The file name or file object to be read.
    flatten : bool, optional
        If True, flattens the color layers into a single gray-scale layer.
    mode : str, optional
        Mode to convert image to, e.g. ``'RGB'``.  See the Notes for more
        details.
    Returns
    -------
    imread : ndarray
        The array obtained by reading the image.
    Notes
    -----
    `imread` uses the Python Imaging Library (PIL) to read an image.
    The following notes are from the PIL documentation.
    `mode` can be one of the following strings:
    * 'L' (8-bit pixels, black and white)
    * 'P' (8-bit pixels, mapped to any other mode using a color palette)
    * 'RGB' (3x8-bit pixels, true color)
    * 'RGBA' (4x8-bit pixels, true color with transparency mask)
    * 'CMYK' (4x8-bit pixels, color separation)
    * 'YCbCr' (3x8-bit pixels, color video format)
    * 'I' (32-bit signed integer pixels)
    * 'F' (32-bit floating point pixels)
    PIL also provides limited support for a few special modes, including
    'LA' ('L' with alpha), 'RGBX' (true color with padding) and 'RGBa'
    (true color with premultiplied alpha).
    When translating a color image to black and white (mode 'L', 'I' or
    'F'), the library uses the ITU-R 601-2 luma transform::
        L = R * 299/1000 + G * 587/1000 + B * 114/1000
    When `flatten` is True, the image is converted using mode 'F'.
    When `mode` is not None and `flatten` is True, the image is first
    converted according to `mode`, and the result is then flattened using
    mode 'F'.
    """

    im = Image.open(name)
    return fromimage(im, flatten=flatten, mode=mode)




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
		read_image = imread(file, flatten=True)
		border_color = read_image[0,0]
		read_image[read_image == border_color] = 0
		read_image[read_image > 0] = 255
		read_image = np.int8(read_image/255)

		if i == 0:
			mask = imread(mask_dir+family+'_segmentation.png', flatten=True)
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
	toimage(base_image, cmin=0, cmax=255).save(output_dir+family+'_semantic.png')
