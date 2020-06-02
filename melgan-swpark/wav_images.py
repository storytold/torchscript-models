from PIL import Image, ImageDraw

def render_histogram(wav, filename='histograms/output.png', row_index=0, col_index=1):
    dimensions = wav.shape
    print('>>> Image input size: {}'.format(dimensions))
    rows = dimensions[row_index]
    cols = dimensions[col_index]

    # Remove 1-sized dimensions
    wav_signal = wav.squeeze()

    print('>>> Generating image (rows: {} cols: {})'.format(rows, cols))
    image = Image.new('RGB', (rows, cols))
    pixels = image.load()

    minimum = wav_signal[0,0]
    maximum = wav_signal[0,0]

    for x in range(0, rows):
        for y in range(0, cols):
            if wav_signal[x,y] > maximum:
                maximum = wav_signal[x,y]
            if wav_signal[x,y] < minimum:
                minimum = wav_signal[x,y]

    print('Minimum: ' + str(minimum))
    print('Maximum: ' + str(maximum))

    if maximum == minimum:
        maximum = 1 # nb: to prevent division by zero below

    for x in range(0, rows):
        for y in range(0, cols):
            v = wav_signal[x,y]
            scaled = int((v - minimum) / (maximum - minimum) * 255)
            pixels[x, y] = (scaled, scaled, scaled)

    # Image Show only works on local X server:
    #image.show()
    image.save(filename)

def rescale_mel(mel, rescaled_min=-1.0, rescaled_max=1.0):
    # NB: Assumes 3 dimensions
    dimensions = mel.shape
    minimum = mel[0,0,0]
    maximum = mel[0,0,0]

    for x in range(0, dimensions[0]):
        for y in range(0, dimensions[1]):
            for z in range(0, dimensions[2]):
                if mel[x,y,z] > maximum:
                    maximum = mel[x,y,z]
                if mel[x,y,z] < minimum:
                    minimum = mel[x,y,z]

    print('Actual Minimum: ' + str(minimum))
    print('Actual Maximum: ' + str(maximum))

    old_range = float(maximum) - minimum
    rescaled_range = float(rescaled_max) - rescaled_min

    print('Old Range: ' + str(old_range))
    print('Rescaled Range: ' + str(rescaled_range))

    for x in range(0, dimensions[0]):
        for y in range(0, dimensions[1]):
            for z in range(0, dimensions[2]):
                d = mel[x,y,z]
                rescaled = (d - minimum) * rescaled_range / old_range + rescaled_min
                mel[x,y,z] = rescaled

"""
# The following is taken from SciPy Cookbook
# https://scipy-cookbook.readthedocs.io/items/Rebinning.html
import numpy as n
import scipy.interpolate
import scipy.ndimage

def congrid(a, newdims, method='linear', centre=False, minusone=False):
    '''Arbitrary resampling of source array to new dimension sizes.
    Currently only supports maintaining the same number of dimensions.
    To use 1-D arrays, first promote them to shape (x,1).

    Uses the same parameters and creates the same co-ordinate lookup points
    as IDL''s congrid routine, which apparently originally came from a VAX/VMS
    routine of the same name.

    method:
    neighbour - closest value from original data
    nearest and linear - uses n x 1-D interpolations using
                         scipy.interpolate.interp1d
    (see Numerical Recipes for validity of use of n 1-D interpolations)
    spline - uses ndimage.map_coordinates

    centre:
    True - interpolation points are at the centres of the bins
    False - points are at the front edge of the bin

    minusone:
    For example- inarray.shape = (i,j) & new dimensions = (x,y)
    False - inarray is resampled by factors of (i/x) * (j/y)
    True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
    This prevents extrapolation one element beyond bounds of input array.
    '''
    if not a.dtype in [n.float64, n.float32]:
        a = n.cast[float](a)

    m1 = n.cast[int](minusone)
    ofs = n.cast[int](centre) * 0.5
    old = n.array( a.shape )
    ndims = len( a.shape )
    if len( newdims ) != ndims:
        print("[congrid] dimensions error. " \
              "This routine currently only support " \
              "rebinning to the same number of dimensions.")
        return None
    newdims = n.asarray( newdims, dtype=float )
    dimlist = []

    if method == 'neighbour':
        for i in range( ndims ):
            base = n.indices(newdims)[i]
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        cd = n.array( dimlist ).round().astype(int)
        newa = a[list( cd )]
        return newa

    elif method in ['nearest','linear']:
        # calculate new dims
        for i in range( ndims ):
            base = n.arange( newdims[i] )
            dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                            * (base + ofs) - ofs )
        # specify old dims
        olddims = [n.arange(i, dtype = n.float) for i in list( a.shape )]

        # first interpolation - for ndims = any
        mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
        newa = mint( dimlist[-1] )

        trorder = [ndims - 1] + list(range( ndims - 1 ))
        for i in range( ndims - 2, -1, -1 ):
            newa = newa.transpose( trorder )

            mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
            newa = mint( dimlist[i] )

        if ndims > 1:
            # need one more transpose to return to original dimensions
            newa = newa.transpose( trorder )

        return newa
    elif method in ['spline']:
        oslices = [ slice(0,j) for j in old ]
        oldcoords = n.ogrid[oslices]
        nslices = [ slice(0,j) for j in list(newdims) ]
        newcoords = n.mgrid[nslices]

        newcoords_dims = list(range(n.rank(newcoords)))
        #make first index last
        newcoords_dims.append(newcoords_dims.pop(0))
        newcoords_tr = newcoords.transpose(newcoords_dims)
        # makes a view that affects newcoords

        newcoords_tr += ofs

        deltas = (n.asarray(old) - m1) / (newdims - m1)
        newcoords_tr *= deltas

        newcoords_tr -= ofs

        newa = scipy.ndimage.map_coordinates(a, newcoords)
        return newa
    else:
        print("Congrid error: Unrecognized interpolation type.\n", \
              "Currently only \'neighbour\', \'nearest\',\'linear\',", \
              "and \'spline\' are supported.")
        return None
"""

