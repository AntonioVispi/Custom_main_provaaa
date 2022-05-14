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
	





# import the necessary packages
from skimage.exposure import rescale_intensity
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage import io
from scipy import misc
import numpy as np
import argparse
from PIL import Image
import glob
from tqdm import tqdm
import os
from joblib import Parallel, delayed


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





# Directories
atri_dir = 'attribute_512p/'
image_dir = 'images_512p/'
segmentation_dir = 'seg_512p/'
output_dir = 'instance_map_no_border/'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_name_arr = []
for file in glob.glob(atri_dir+'*.png'):
	temp = file.split('/')[-1].split('_')
	file_name = temp[0]+'_'+temp[1]
	if file_name not in file_name_arr:
	    file_name_arr.append(file_name)

def create_instance_map(family):
	# Create a zero filled base image
	# Load original image
	image = imread(image_dir+family+'.png')
	instance_map = np.zeros(image.shape[:2], dtype=int)
	segments = slic(img_as_float(image), n_segments=1000,
	slic_zero=True, compactness=1, sigma=2)

	for i, file in enumerate(glob.glob(atri_dir+family+'*.png')):
		# Read Mask
		mask = imread(file)
		type_file = file.split('/')[-1].split('_')[3]
		if i ==0:
			segmentation = imread(segmentation_dir+family+'_segmentation.png', flatten=True)
			last_lesion = 2000
			last_background = 1000
			for v in np.unique(segments):
				union = segmentation[segments==v]
				if float(len(union[union > 0])) / float(len(union)) > 0.5:
					instance_map[segments == v] = last_lesion
					last_lesion += 1
				else:
					instance_map[segments == v] = last_background
					last_background += 1

		if type_file == 'pigment': # 3
			last = 3000
		elif type_file == 'negative': # 4
			last = 4000
		elif type_file.startswith('streaks'): # 5
			last = 5000
		elif type_file == 'milia': # 6
			last = 6000
		elif type_file.startswith('globules'): #7
			last = 7000
		else:
			print('ERROR: Invalid File Found!!!!')
		# For each superpixel in the selected mask, update the pixel values incrementing the value for each superpixel found.
		for v in np.unique(segments):
			union = mask[segments == v]
			if float(len(union[union > 0])) / float(len(union)) > 0.5:
				instance_map[segments == v] = last
				last += 1
	instance_map = instance_map.astype(np.uint32)
	im = Image.fromarray(instance_map)
	im.save(output_dir+family+'_instance.png')

	
results = Parallel(n_jobs=8)(delayed(create_instance_map)(family) for family in tqdm(file_name_arr))
