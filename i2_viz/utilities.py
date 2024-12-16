import numpy as np
import holoviews as hv
import nibabel as nib


def normalize_to_symmetric_range(min_val: float, max_val: float):
    max_abs_val = max(abs(min_val), abs(max_val))
    return -max_abs_val, max_abs_val


# Image slicing
# adapted from https://panel.holoviz.org/gallery/vtk_slicer.html
def image_slice(dims, array, low, high, bounds, cmap='seismic', colorbar=True, tools=[]):
    img = hv.Image(array, bounds=bounds, kdims=dims, vdims='Intensity')
    return img.opts(clim=(low, high), cmap=cmap, colorbar=colorbar, tools=tools) 

def image_slice_i(si, vol, low, high, cmap='seismic', colorbar=True, tools=[]):
    array = np.flipud(vol[si, :, :].T)
    bounds = (0, 0, vol.shape[1]-1, vol.shape[2]-1)
    return image_slice(['y','z'], array, low, high, bounds, cmap, colorbar, tools)

def image_slice_j(sj, vol, low, high, cmap='seismic', colorbar=True, tools=[]):
    array = np.flipud(vol[:, sj, :].T)
    bounds = (0, 0, vol.shape[0]-1, vol.shape[2]-1)
    return image_slice(['x','z'], array, low, high, bounds, cmap, colorbar, tools)

def image_slice_k(sk, vol, low, high, cmap='seismic', colorbar=True, tools=[]):
    array = np.flipud(vol[:, :, sk].T)
    bounds = (0, 0, vol.shape[0]-1, vol.shape[1]-1)
    return image_slice(['x','y'], array, low, high, bounds, cmap, colorbar, tools)


underlay_img  = nib.load('/Users/burkelawlor/Repos/i2-viz/data/ch2bet_resampled.nii')
underlay_vol = underlay_img.get_fdata()
underlay_clim = normalize_to_symmetric_range(underlay_vol.min(), underlay_vol.max())
underlay_common = dict(
    vol = underlay_vol,
    low = 0,
    high = 300,
    cmap = 'greys',
    colorbar=False,
    tools=[]
)

def image_slice_i_with_underlay(si, slice_common):
    return image_slice_i(si, **underlay_common) * image_slice_i(si, **slice_common)

def image_slice_j_with_underlay(sj, slice_common):
    return image_slice_j(sj, **underlay_common) * image_slice_j(sj, **slice_common)

def image_slice_k_with_underlay(sk, slice_common):
    return image_slice_k(sk, **underlay_common) * image_slice_k(sk, **slice_common)


# Time series plotting
def vertical_line_callback(time_current):
    return hv.VLine(time_current).opts(color='red', line_width=2, line_dash='dashed')

from bokeh.models import GlyphRenderer, LinearAxis, LinearScale, Range1d
def overlay_hook(plot, element):
    # Adds a secondary y-axis (right)
    p = plot.handles['plot']

    if 'right' not in p.extra_y_scales:
        p.extra_y_scales = {'right': LinearScale()}
        p.extra_y_ranges = {'right': Range1d(start=-0.5, end=4.5)}
        p.add_layout(LinearAxis(y_range_name='right'), 'right')

        # Assign scatterplots to the right axis
        lines = [p for p in p.renderers if isinstance(p, GlyphRenderer)]
        lines[-1].y_range_name = "right"
        lines[-2].y_range_name = "right"
        lines[-3].y_range_name = "right"
