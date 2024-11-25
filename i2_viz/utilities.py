import numpy as np
import holoviews as hv


def normalize_to_symmetric_range(min_val: float, max_val: float):
    max_abs_val = max(abs(min_val), abs(max_val))
    return -max_abs_val, max_abs_val


# Image slicing
# adapted from https://panel.holoviz.org/gallery/vtk_slicer.html
def image_slice(dims, array, low, high, bounds, cmap='seismic', colorbar=True):
    img = hv.Image(array, bounds=bounds, kdims=dims, vdims='Intensity')
    return img.opts(clim=(low, high), cmap=cmap, colorbar=colorbar) 

def image_slice_i(si, vol, low, high, cmap='seismic'):
    array = vol[si, :, :].T
    bounds = (0, 0, vol.shape[1]-1, vol.shape[2]-1)
    return image_slice(['y','z'], array, low, high, bounds, cmap)

def image_slice_j(sj, vol, low, high, cmap='seismic'):
    array = vol[:, sj, :].T
    bounds = (0, 0, vol.shape[0]-1, vol.shape[2]-1)
    return image_slice(['x','z'], array, low, high, bounds, cmap)

def image_slice_k(sk, vol, low, high, cmap='seismic'):
    array = vol[:, :, sk].T
    bounds = (0, 0, vol.shape[0]-1, vol.shape[1]-1)
    return image_slice(['x','y'], array, low, high, bounds, cmap)


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
