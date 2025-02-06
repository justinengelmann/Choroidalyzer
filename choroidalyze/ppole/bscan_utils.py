import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import torch
import pandas as pd
from scipy import interpolate as interp
from skimage import measure
from sklearn.linear_model import LinearRegression

def sort_alphanumeric(name):
    _, fname = os.path.split(name)
    num = fname.split('_')[1].split('.')[0]
    return int(num)


def flatten_dict(nested_dict):
    """
    Recursively flattens a nested dictionary where each value can be a dictionary itself. 
    The function traverses the nested structure and constructs a flat dictionary where 
    the keys represent the hierarchical path to the value.

    Parameters:
    -----------
    nested_dict : dict
        A nested dictionary where the values can be either other dictionaries or non-dictionary values.

    Returns:
    --------
    dict
        A flattened dictionary where the keys are tuples representing the path to the value 
        in the original nested structure, and the values are the corresponding leaf values.

    Example:
    --------
    >>> example_dict = {
        'A': {'A1': 1, 'A2': 2},
        'B': {'B1': 3}
    }
    >>> flatten_dict(example_dict)
    {
        ('A', 'A1'): 1,
        ('A', 'A2'): 2,
        ('B', 'B1'): 3
    }
    """
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict):
    """
    Converts a nested dictionary into a multi-level Pandas DataFrame by first flattening it.
    The resulting DataFrame has a hierarchical column structure, where the dictionary keys 
    define the index and columns.

    Parameters:
    -----------
    values_dict : dict
        A nested dictionary that needs to be converted into a DataFrame. The dictionary is flattened 
        before being transformed into a DataFrame.

    Returns:
    --------
    pandas.DataFrame
        A Pandas DataFrame where the flattened dictionary is represented with a multi-level 
        index and columns. The index reflects the path from the root to the leaf in the nested structure.

    Notes:
    ------
    - This function uses `flatten_dict(...)` to flatten the input dictionary before converting it into a DataFrame.
    - The final DataFrame is "unstacked" to create hierarchical columns, which are formatted based on 
      the second level of the dictionary keys.

    Example:
    --------
    >>> example_dict = {
        'A': {'A1': 1, 'A2': 2},
        'B': {'B1': 3}
    }
    >>> nested_dict_to_df(example_dict)
    A         B
    A1  A2    B1
    1   2     3
    """
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df
    

def extract_bounds(mask):
    """
    Extract the top and bottom boundaries of a binary mask.

    Parameters:
    -----------
    mask : numpy.ndarray
        Binary mask with a connected region of interest.

    Returns:
    --------
    tuple of (numpy.ndarray, numpy.ndarray)
        Top and bottom boundaries of the mask as arrays of coordinates.

    Notes:
    ------
    - Assumes the mask is fully connected and can be sorted along the horizontal axis.
    """
    # Stack of indexes where mask has predicted 1
    where_ones = np.vstack(np.where(mask.T)).T
    
    # Sort along horizontal axis and extract indexes where differences are
    sort_idxs = np.argwhere(np.diff(where_ones[:,0]))
    
    # Top and bottom bounds are either at these indexes or consecutive locations.
    bot_bounds = np.concatenate([where_ones[sort_idxs].squeeze(),
                                 where_ones[-1,np.newaxis]], axis=0)
    top_bounds = np.concatenate([where_ones[0,np.newaxis],
                                 where_ones[sort_idxs+1].squeeze()], axis=0)
    
    return (top_bounds, bot_bounds)


def select_largest_mask(binmask):
    """
    Retain only the largest connected region in a binary mask.

    Parameters:
    -----------
    binmask : numpy.ndarray
        Binary mask with (potentially) multiple connected regions.

    Returns:
    --------
    numpy.ndarray
        Binary mask with only the largest connected region retained.
    """
    # Look at which of the region has the largest area, and set all other regions to 0
    labels_mask = measure.label(binmask)                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1

    return labels_mask


def rebuild_mask(traces, img_shape=None):
    """
    Rebuild a binary mask from upper and lower boundaries. The mask is created by filling in the region between the top and bottom traces.

    Parameters:
    -----------
    traces : tuple of numpy.ndarray
        Top and bottom traces.

    img_shape : tuple of (int, int), optional
        Shape of the output mask. If `None`, the mask size is inferred from traces.

    Returns:
    --------
    numpy.ndarray
        Binary mask reconstructed from the traces.
    """
    # Work out extremal coordinates of traces
    top_lyr, bot_lyr = interp_trace(traces)
    common_st_idx = np.maximum(top_lyr[0,0], bot_lyr[0,0])
    common_en_idx = np.minimum(top_lyr[-1,0], bot_lyr[-1,0])
    top_idx = top_lyr[:,1].min()
    bot_idx = bot_lyr[:,1].max()

    # Initialise binary mask
    if img_shape is not None:
        binmask = np.zeros(img_shape)
    else:
        binmask = np.zeros((bot_idx+100, common_en_idx+100))

    # Fill in region between upper and lower boundaries
    for i in range(common_st_idx, common_en_idx):
        top_i = top_lyr[i-common_st_idx,1]
        bot_i = bot_lyr[i-common_st_idx,1]
        binmask[top_i:bot_i,i] = 1

    return binmask


def interp_trace(traces, align=True):
    """
    Interpolate traces to ensure continuity along the x-axis.

    Parameters:
    -----------
    traces : list of numpy.ndarray
        List of traces, each containing x and y coordinates.

    align : bool, default=True
        Whether to align traces to a common x-range.

    Returns:
    --------
    tuple of numpy.ndarray
        Interpolated traces.
    """
    # Interpolate traces
    new_traces = []
    for i in range(2):
        tr = traces[i]  
        min_x, max_x = (tr[:,0].min(), tr[:,0].max())
        x_grid = np.arange(min_x, max_x)
        y_interp = np.interp(x_grid, tr[:,0], tr[:,1]).astype(int)
        interp_trace = np.concatenate([x_grid.reshape(-1,1), y_interp.reshape(-1,1)], axis=1)
        new_traces.append(interp_trace)

    # Crop traces to make sure they are aligned
    if align:
        top, bot = new_traces
        h_idx=0
        top_stx, bot_stx = top[0,h_idx], bot[0,h_idx]
        common_st_idx = max(top[0,h_idx], bot[0,h_idx])
        common_en_idx = min(top[-1,h_idx], bot[-1,h_idx])
        shifted_top = top[common_st_idx-top_stx:common_en_idx-top_stx]
        shifted_bot = bot[common_st_idx-bot_stx:common_en_idx-bot_stx]
        new_traces = (shifted_top, shifted_bot)

    return tuple(new_traces)



def smart_crop(traces, check_idx=20, ythresh=1, align=True):
    """
    Crop traces to remove discontinuities based on local y-value changes at the end of
    traces.

    Parameters:
    -----------
    traces : list of numpy.ndarray
        List of traces, each containing x and y coordinates.

    check_idx : int, default=20
        Number of points to check at the start and end of each trace.

    ythresh : float, default=1
        Threshold for discontinuity detection in y-values.

    align : bool, default=True
        Whether to align cropped traces to a common x-range.

    Returns:
    --------
    tuple of numpy.ndarray
        Cropped and aligned traces.
    """
    cropped_tr = []
    for i in range(2):
        lyr = traces[i]
        ends_l = np.argwhere(np.abs(np.diff(lyr[:check_idx,1])) > ythresh)
        ends_r = np.argwhere(np.abs(np.diff(lyr[-check_idx:,1])) > ythresh)
        if ends_r.shape[0] != 0:
            lyr = lyr[:-(check_idx-ends_r.min())]
        if ends_l.shape[0] != 0:
            lyr = lyr[ends_l.max()+1:]
        cropped_tr.append(lyr)

    return interp_trace(cropped_tr, align=align)



def get_trace(pred_mask, threshold=0.5, align=False):
    """
    Extract top and bottom traces from a prediction mask.

    The function thresholds the mask, selects the largest connected region, extracts boundaries, and crops discontinuities.

    Parameters:
    -----------
    pred_mask : numpy.ndarray
        Prediction mask, typically probabilistic.

    threshold : float, default=0.5
        Threshold for binarizing the mask.

    align : bool, default=False
        Whether to align the traces to a common x-range.

    Returns:
    --------
    tuple of numpy.ndarray
        Top and bottom traces extracted from the mask.
    """
    if threshold is not None:
        binmask = (pred_mask > threshold).astype(int)
    binmask = select_largest_mask(binmask)
    traces = extract_bounds(binmask)
    traces = smart_crop(traces, align=align)
    return traces


def generate_imgmask(mask, thresh=None, cmap=0):
    """
    Generate a plottable RGBA mask from a binary or probabilistic mask.

    Parameters:
    -----------
    mask : numpy.ndarray
        Input mask, typically binary or probabilistic.

    thresh : float, optional
        Threshold for binarising the mask. Values below the threshold are set to 0, others to 1.

    cmap : int or None, default=0
        Index of the RGB channel to colorise. If `None`, all channels are used equally.

    Returns:
    --------
    numpy.ndarray
        Plottable RGBA mask with transparency for non-mask regions.

    Notes:
    ------
    - The function creates an RGBA image, where the alpha channel corresponds to the mask's binary values.
    """
    # Threshold
    pred_mask = mask.copy()
    if thresh is not None:
        pred_mask[pred_mask < thresh] = 0
        pred_mask[pred_mask >= thresh] = 1
    max_val = pred_mask.max()
    
    # Compute plottable cmap using transparency RGBA image.
    trans = max_val*((pred_mask > 0).astype(int)[...,np.newaxis])
    if cmap is not None:
        rgbmap = np.zeros((*mask.shape,3))
        rgbmap[...,cmap] = pred_mask
    else:
        rgbmap = np.transpose(3*[pred_mask], (1,2,0))
    pred_mask_plot = np.concatenate([rgbmap,trans], axis=-1)
    
    return pred_mask_plot


def construct_line(p1, p2):
    """
    Compute the gradient and intercept of a straight line between two points.

    Parameters
    ----------
    p1 : np.ndarray or list
        A 2D coordinate (x, y) representing the first point.
        
    p2 : np.ndarray or list
        A 2D coordinate (x, y) representing the second point.

    Returns
    -------
    m : float
        The gradient (slope) of the line.
        
    c : float
        The y-intercept of the line.

    Notes
    -----
    If the line is vertical, the function returns `m` and `c` as `np.inf`.
    """
    # Measure difference between x- and y-coordinates of p1 and p2
    delta_x = (p2[0] - p1[0])
    delta_y = (p2[1] - p1[1])

    # Compute gradient and intercept
    try:
        assert delta_x != 0
        m = delta_y / delta_x
        c = p2[1] - m * p2[0]
    except AssertionError:
        m = np.inf
        c = np.inf

    return m, c



def generate_perp_line(pt1, pt2=None, N=None, ref_pt=None):
    """
    Generates a perpendicular line to a given line (tangent) defined by two points.
    The line is evaluated far enough to ensure its intersection with a defined boundary 
    (e.g., Choroid-Sclera boundary) after being rotated by 90 degrees around a reference point.

    Parameters:
    -----------
    pt1 : numpy.ndarray
        A 2D array representing the first point of the tangent line. It must have shape (2,).

    pt2 : numpy.ndarray, optional, default=None
        A 2D array representing the second point of the tangent line. If not provided, 
        a linear model is generated from a single point (`pt1`).

    N : int, optional, default=None
        A scalar value determining how far along the tangent line to evaluate. It defines the 
        range of the tangent line in the x-direction.

    ref_pt : tuple of int, optional, default=None
        A tuple containing the x and y coordinates of the reference point around which the tangent 
        line will be rotated to generate the perpendicular line.

    Returns:
    --------
    tuple of numpy.ndarray
        A tuple containing two 1D numpy arrays: the x and y coordinates of the generated perpendicular line.

    Notes:
    ------
    - The function first fits a linear regression model to the tangent line defined by `pt1` and `pt2` (or just `pt1`).
    - If `N` and `ref_pt` are provided, the function generates the perpendicular line by rotating the tangent line 
      by 90 degrees around `ref_pt` and evaluates the line over the defined range.

    Example:
    --------
    perp_line = generate_perp_line(np.array([1, 2]), pt2=np.array([3, 4]), N=10, ref_pt=(2, 3))
    # This will return the coordinates of the perpendicular line at a given range, rotated around (2, 3).
    """
    # Fit linear model at reference points along tangent
    if pt2 is None:
        X, y = pt1[:,0].reshape(-1,1), pt1[:,1]
    else:
        X, y = np.array([pt1[0], pt2[0]]).reshape(-1,1), np.array([pt1[1], pt2[1]])    
    output = LinearRegression().fit(X, y)

    # Generate perpendicular line if reference point and sample size provided
    if N is not None and ref_pt is not None:

        # Evaluate across tangent
        ref_x, ref_y = ref_pt
        xtan_grid = np.array([ref_x, X[-1,0]+N])
        ytan_grid = output.predict(xtan_grid.reshape(-1,1)).astype(int)

        # Rotate at reference point 90 degrees
        perp_x = (-(ytan_grid - ref_y) + ref_x).reshape(-1,)
        perp_y = (xtan_grid - ref_x + ref_y).reshape(-1,)
        output = (perp_x, perp_y)

        # build output of perpendicular line
        y_grid = np.arange(perp_y[0], perp_y[1])
        x_grid = np.interp(y_grid, perp_y, perp_x)
        output = (x_grid, y_grid)
        
    return output



def detect_orthogonal_coords(reference_pts, traces, offset=15, tol=2):
    """
    Detects coordinates along the lower boundary that intersect with perpendicular lines
    drawn from tangent lines at reference points along the upper boundary. The function calculates
    the points on the lower boundary where the perpendicular lines from the upper boundary's tangent
    lines intersect, within a given tolerance.

    Parameters:
    -----------
    reference_pts : numpy.ndarray
        A 2D array of reference points along the upper boundary. Each point defines the location
        from which a tangent and corresponding perpendicular line will be drawn.

    traces : tuple of numpy.ndarray
        A tuple containing two 2D arrays: the upper and lower boundaries of the segmented layer in xy-space.
        The upper boundary (`top_lyr`) and the lower boundary (`bot_lyr`) are both 2D arrays of shape (N, 2).

    offset : int, optional, default=15
        The distance (in pixels) on either side of each reference point used to define the tangent line.
        This controls how local the tangent lines are defined.

    tol : int, optional, default=2
        The threshold (in pixels) to detect pixels along the lower boundary which intersect as close to the
        perpendicular lines.

    Returns:
    --------
    tuple of numpy.ndarray
        - chorscl_pts : The coordinates along the lower choroid boundary where perpendicular lines from the 
          upper boundary intersect, within the given tolerance.
        - reference_pts : The original reference points along the upper boundary that correspond to the 
          detected intersection points on the lower boundary.
        - perps : The perpendicular lines corresponding to each reference point along the upper boundary, 
          truncated to the detected intersection points.

    Notes:
    ------
    - The function works by generating tangent lines at each reference point on the upper boundary and 
      calculating the corresponding perpendicular lines.
    - These perpendicular lines are then compared to the lower boundary to find the intersection points.
    - The intersection points are accepted if their Euclidean distance to the lower boundary is within the given tolerance.

    Example:
    --------
    chorscl_pts, reference_pts, perps = detect_orthogonal_coords(reference_pts, (top_lyr, bot_lyr), offset=20, tol=3)
    # This will return the coordinates where the perpendicular lines intersect the lower boundary,
    # along with the corresponding reference points on the upper boundary.
    """
    # Extract traces    
    top_lyr, bot_lyr = traces
    toplyr_stx, botlyr_stx = top_lyr[0, 0], bot_lyr[0, 0]

    # total number of candidate points at each reference point to compare with 
    # Choroid-Sclera boundary
    N = max([bot_lyr[ref_x-botlyr_stx, 1] - ref_y for (ref_x, ref_y) in reference_pts])
    perps = []
    for ref_pt in reference_pts:
    
        # Work out local tangent line for each reference point
        # and rotate orthogonally
        ref_x, ref_y = ref_pt
        ref_xidx = ref_x - toplyr_stx
        tan_pt1, tan_pt2 = top_lyr[[ref_xidx - offset, ref_xidx + offset]] 
        (perp_x, perp_y) = generate_perp_line(tan_pt1, tan_pt2, N, ref_pt)
        perps.append(np.array([perp_x, perp_y]))
    
    # Vectorised search for points along Choroid-Sclera boundary where orthogonal 
    # lines from RPE-Choroid intersect
    perps = np.array(perps)
    bot_cropped = bot_lyr[(perps[:,0].astype(int)-botlyr_stx).clip(0, bot_lyr.shape[0]-1)]
    bot_perps_residuals = np.transpose(perps, (0,2,1)) - bot_cropped
    bot_perps_distances = np.sqrt(((bot_perps_residuals)**2).sum(axis=-1))
    endpoint_errors = np.min(bot_perps_distances, axis=-1) <= tol 
    botlyr_indexes = np.argmin(bot_perps_distances, axis=1)
    botlyr_pts = perps[np.arange(botlyr_indexes.shape[0]),:,botlyr_indexes].astype(int)

    return botlyr_pts[endpoint_errors], reference_pts[endpoint_errors], perps[endpoint_errors].astype(int)



def get_fovea(rvfmasks, foveas, predict_fovea):
    """
    Resolves the fovea coordinate based on the provided fovea prediction maps and scan type. 
    This function is designed to handle cases where the fovea prediction is below a threshold 
    or when the scan acquisition is not centered at the fovea.

    Parameters:
    -----------
    rvfmasks : list of numpy arrays
        A list of region/vessel/fovea (RVF) masks for all B-scan slices. 
        Each mask is a 2D array representing the raw pixel-wise predictions of chorioretinal features.

    foveas : list of numpy arrays
        A list of fovea xy-predictions for each scan slice. Each element is an (x,y)-coordinate which
        is the predicted fovea coordinates for the corresponding slice.

    predict_fovea : functiob
        Function to predict the fovea from a raw fovea mask outputted from Choroidalyzer

    Returns:
    --------
    fovea_slice_num : int
        The index of the B-scan slice where the fovea is most likely located.

    fovea : numpy array or None
        The predicted fovea (xy)-coordinates for the selected slice. If no fovea is detected, it can be `None`.
    """
    N_scans = rvfmasks.shape[0]
    fovea_slice_num = N_scans//2
    fovea = foveas[fovea_slice_num]
    if fovea.sum() == 0:
        logging.warning("Prediction threshold for fovea too high or non-centred Ppole scan acquisition.")
        foveas_arr = np.array(foveas)
        fov_idx = np.where(foveas_arr[:,0]>0)[0]

        # If default fovea-centred B-scan prediction is at origin, work out highest score from fovea masks
        # which have detected a fovea coordinate
        if fov_idx.shape[0] > 0:
            fov_scores = []
            fov_preds = []
            for idx in fov_idx:
                fmask = rvfmasks[idx]
                fov_pred = predict_fovea(torch.tensor(rvfmasks).unsqueeze(0))[0]
                fov_scores.append(fmask[fov_pred[1], fov_pred[0]])
                fov_preds.append(fov_pred)
            fovea_slice_num = fov_idx[np.argmax(fov_scores)]
            logging.warning(f"Potentially detected fovea-centred B-scan in Ppole at slice {fovea_slice_num}/{N_scans}.")

        fmask = rvfmasks[fovea_slice_num]
        fovea = predict_fovea(torch.tensor(fmask).unsqueeze(0))

    return fovea_slice_num, fovea



def plot_composite_bscans(bscan_data, 
                          all_rtraces, 
                          vmasks, 
                          fovea_slice_num, 
                          reshape_idx, 
                          fname, 
                          save_path):
    """
    Create and save a composite high-resolution visualization of all B-scans in an OCT stack.

    This function generates a stitched image of all B-scans from an OCT scan stack, with overlaid 
    segmentations, regions of interest (ROI), and optional vessel maps. It supports volume and 
    H-line/V-line/Radial scan formats and can include fovea-centered landmarks and additional ROI overlays if specified.

    Parameters:
    ----------
    bscan_data : ndarray
        3D array containing the OCT B-scan data with dimensions `(num_scans, height, width)`.

    all_rtraces : np.ndarray
        Segmentation traces of the choroid.

    vmasks : ndarray
        3D array of choroidal vessel masks corresponding to the B-scans. Required if `analyse_choroid` is `True`.

    fovea_slice_num : int
        - An integer, specifying the index of the fovea-centered B-scan in volume scans.

    reshape_idx : tuple
        Tuple specifying how the B-scans should be arranged in the composite (e.g., `(rows, cols)`).
        
    fname : str
        The base name of the output file (excluding the file extension).

    save_path : str or pathlib.Path
        Directory path where the resulting composite image will be saved.

    Returns:
    -------
    None
        The function saves the generated composite image to the specified directory.

    Outputs:
    -------
    - A high-resolution PNG file showing the stitched B-scans with overlays:
        - For volume scans: `{fname}_volume_octseg.png`

    Notes:
    -----
    - For volume scans, the fovea-centered B-scan is excluded from the composite image.
    - The figure dimensions and DPI are optimised for high-resolution output.
    """
    img_shape = bscan_data.shape[-2:]
    rtraces = all_rtraces.copy()
    rtraces.pop(fovea_slice_num)
    M, N = img_shape

    # Organise B-scan data
    bscan_list = list(bscan_data.copy())
    bscan_list.pop(fovea_slice_num)
    bscan_arr = np.array(bscan_list)
    bscan_arr = bscan_arr.reshape(*reshape_idx,*img_shape)
    bscan_stacked = np.concatenate(np.concatenate(bscan_arr, axis=-2), axis=-1)

    # Sort out vessel maps if analysing choroid
    vmasks_list = list(vmasks.copy())
    vmasks_list.pop(fovea_slice_num)
    vmasks_arr = np.asarray(vmasks_list)
    vmasks_arr = vmasks_arr.reshape(*reshape_idx,M,N)
    vmask_stacked = np.concatenate(np.concatenate(vmasks_arr, axis=-2), axis=-1)
    all_vcmap = np.concatenate([vmask_stacked[...,np.newaxis]] 
                + 2*[np.zeros_like(vmask_stacked)[...,np.newaxis]] 
                + [vmask_stacked[...,np.newaxis] > 0.01], axis=-1)

    # Figure to be saved out at same dimensions as stacked array
    h,w = bscan_stacked.shape
    fig, ax = plt.subplots(1,1,figsize=(w/1000, h/1000), dpi=100)
    ax.set_axis_off()

    # Overlay stacked B-scans
    ax.imshow(bscan_stacked, cmap='gray')
    
    # Add all traces and fovea (if provided)
    for (i, j) in np.ndindex(reshape_idx):
        
        # Overlay trace
        for tr in rtraces[reshape_idx[1]*i + j]:
            ax.plot(tr[:,0]+j*N, tr[:,1]+i*M, label='_ignore', color='r', zorder=2, linewidth=0.175)

    # Add vessel maps
    ax.imshow(all_vcmap, alpha=0.5)

    # Prepare to save out
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(save_path, f"{fname}_volume_octseg.png"), dpi=1000)
    plt.close()