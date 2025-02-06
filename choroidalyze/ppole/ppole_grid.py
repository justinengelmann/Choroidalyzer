import logging
import os
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.feature as feature
import skimage.measure as meas 
import skimage.transform as trans
import skimage.morphology as morph
from skimage import segmentation
from scipy import interpolate
from skimage import draw
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib

from choroidalyze.ppole import bscan_utils


def rotate_point(point, origin, angle):
    """
    Rotate a point counterclockwise by a specified angle around a given origin.

    Parameters
    ----------
    point : tuple or list of float
        Coordinates of the point to be rotated, in the format (x, y).
        
    origin : tuple or list of float
        Coordinates of the origin point around which the rotation is performed, in the format (x, y).
        
    angle : float
        Rotation angle in radians. Positive values rotate anticlockwise.

    Returns
    -------
    qx, qy : float
        Coordinates of the rotated point.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy



def create_circular_mask(img_shape=(768,768), center=None, radius=None):
    """
    Create a binary circular mask with a specified center, radius, and image shape.

    Parameters
    ----------
    img_shape : tuple of int, optional
        Shape of the output image, specified as (height, width). Default is (768, 768).
        
    center : tuple of int, optional
        (x,y)-coordinates of the circle's center. 
        If None, the center is set to the middle of the image.
        
    radius : int, optional
        Radius of the circle in pixels. If None, the radius is set to the largest value
        that fits within the image boundaries based on the center.

    Returns
    -------
    mask : numpy.ndarray
        Binary mask with the same dimensions as `img_shape`, where pixels inside the circle
        are `True` and others are `False`.

    Examples
    --------
    - 
    """
    h, w = img_shape
    if center is None: # use the middle of the image
        center = (int(h/2), int(w/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask



def create_circular_grids(circle_mask, angle=0):
    """
    Split a binary circular mask into four equally sized quadrants based on a given 
    angle of rotation.

    This function divides the mask diagonally and can optionally rotate the 
    quadrants based on the specified angle. The resulting quadrants are ordered 
    according to the specified angle, permitting angles of rotation within 
    [-120, 120] degrees.

    Parameters
    ----------
    circle_mask : numpy.ndarray
        A binary mask of a filled circle. The mask is assumed to be circular.

    angle : float, optional, default=0
        The angle (in degrees) used to rotate the quadrants. Positive angles 
        rotate counterclockwise.

    Returns
    -------
    etdrs_masks : list of numpy.ndarray
        A list containing four binary masks, each representing one quadrant 
        of the circular mask, ordered based on the specified angle.
    """
    # Detect all angles from center of circular mask indexes
    output_shape = circle_mask.shape
    c_y, c_x = meas.centroid(circle_mask).astype(int)
    radius = int(circle_mask[:,c_x].sum()/2)
    M, N = output_shape
    circ_idx = np.array(np.where(circle_mask)).T
    x, y = circ_idx[:,1], circ_idx[:,0]
    all_angles = 180/np.pi*np.arctan((c_x-x)/(c_y-y+1e-8))

    # Deal with angles of rotation between [-120, 120]
    relabel = 0
    angle_sign = np.sign(angle)
    if angle != 0:
        angle = angle % (angle_sign*360)
    if abs(angle) > 44:
        rem = (abs(angle)-1) // 44
        angle += -1*angle_sign * 89 
        relabel += rem

    # Select pixels which represent superior and inferior regions, based on angle of elevation of points
    # along circular mask relative to horizontal axis (above 45* and below -45*)
    topbot_idx = np.ma.masked_where((all_angles < 45-angle) & (all_angles > -45-angle), 
                                      np.arange(circ_idx.shape[0])).mask

    # Generate superior-inferior and temporal-nasal subregions of circular mask
    top_bot = np.zeros_like(circle_mask)
    topbot_circidx = circ_idx[topbot_idx].copy()
    top_bot[topbot_circidx[:,0], topbot_circidx[:,1]] = 1
    right_left = np.zeros_like(circle_mask)
    rightleft_circidx = circ_idx[~topbot_idx].copy()
    right_left[rightleft_circidx[:,0], rightleft_circidx[:,1]] = 1

    # Split superior-inferior and temporal-nasal into quadrants
    topbot_split = np.concatenate(2*[np.zeros_like(circle_mask)[np.newaxis]]).astype(int)
    rightleft_split = np.concatenate(2*[np.zeros_like(circle_mask)[np.newaxis]]).astype(int)

    # Split two quadrants up - they're connected by a single pixel so 
    # temporarily remove and then replace
    top_bot[c_y, c_x] = 0
    topbot_props = meas.regionprops(meas.label(top_bot))  
    leftright_props = meas.regionprops(meas.label(right_left))  
    for i,(reg_tb, reg_rl) in enumerate(zip(topbot_props, leftright_props)):
        topbot_split[i, reg_tb.coords[:,0], reg_tb.coords[:,1]] = 1
        rightleft_split[i, reg_rl.coords[:,0], reg_rl.coords[:,1]] = 1
    topbot_split[0][c_y, c_x] = 1

    # Order quadrants consistently dependent on angle
    etdrs_masks = [*topbot_split, *rightleft_split]
    if angle >= 0:
        etdrs_masks = [etdrs_masks[i] for i in [0,2,1,3]]
    else:
        etdrs_masks = [etdrs_masks[i] for i in [0,3,1,2]]

    # Relabelling if angle is outwith [-44, 44]
    if relabel == 1:
        if angle_sign > 0:
            etdrs_masks = [etdrs_masks[i] for i in [3,2,1,0]]
        elif angle_sign < 0:
            etdrs_masks = [etdrs_masks[i] for i in [1,2,3,0]]
    elif relabel == 2:
        if angle_sign > 0:
            etdrs_masks = [etdrs_masks[i] for i in [3,2,1,0]]
        elif angle_sign < 0:
            etdrs_masks = [etdrs_masks[i] for i in [1,2,3,0]]
                
    return etdrs_masks




def create_etdrs_grid(scale=11.49, center=(384,384), img_shape=(768,768), 
                      angle=0, etdrs_microns=[1000,3000,6000]):
    """
    Create an ETDRS (Early Treatment Diabetic Retinopathy Study) grid for analysing
    thickness/density maps. The grid is created using binary masks 
    based on predefined circle radii, segmented into regions corresponding to 
    the central, inner, and outer areas of the grid.ß

    Parameters
    ----------
    scale : float, optional, default=11.49
        The scale factor in microns-per-pixel, used to convert from pixel units to 
        physical units (e.g., microns). This scale affects the size of the ETDRS grid
        on the corresponding SLO image.
    
    center : tuple of int, optional, default=(384,384)
        (x,y)-coordinates representing the center of the grid which will be the fovea on
        the SLO image. This point serves as the origin for drawing the circles.
    
    img_shape : tuple of int, optional, default=(768,768)
        The shape of the image (height, width), typically representing the image pixel resolution.
    
    angle : float, optional, default=0
        The angle (in degrees) used to rotate the quadrants of the grid for alignment.
    
    etdrs_microns : list of int, optional, default=[1000, 3000, 6000]
        The diameters of the circles in the ETDRS grid, measured in microns. The grid consists
        of concentric rings corresponding to these radii.

    Returns
    -------
    (circles, quadrants) : tuple of list of numpy.ndarray
        A tuple containing two lists:
        - circles: A list of binary masks representing concentric circles corresponding 
          to the different radii in the ETDRS grid.
        - quadrants: A list of binary masks representing quadrants for each circle.
    
    (central, inner_circle, outer_circle) : tuple of numpy.ndarray
        A tuple containing:
        - central: A binary mask for the innermost circle.
        - inner_circle: A binary mask representing the region between the central and 
          the inner rings of the ETDRS grid.
        - outer_circle: A binary mask representing the outer ring of the grid.

    (inner_regions, outer_regions) : tuple of list of numpy.ndarray
        A tuple containing:
        - inner_regions: A list of binary masks for the inner quadrants.
        - outer_regions: A list of binary masks for the outer quadrants.
    """
    # Standard diameter measureents of ETDRS study grid.
    etdrs_radii = [int(np.ceil((N/scale)/2)) for N in etdrs_microns]

    # Draw circles and quadrants
    circles = [create_circular_mask(img_shape, center, radius=r) for r in etdrs_radii]
    quadrants = [create_circular_grids(circle, angle) for circle in circles[1:]]

    # Subtract different sized masks to get individual binary masks of ETDRS study grid
    central = circles[0]
    inner_regions = [(q-central).clip(0,1) for q in quadrants[0]]
    inner_circle = np.sum(np.array(inner_regions), axis=0).clip(0,1)
    outer_regions = [(q-inner-central).clip(0,1) for (q,inner) in zip(quadrants[1],inner_regions)]
    outer_circle = np.sum(np.array(outer_regions), axis=0).clip(0,1)

    return (circles, quadrants), (central, inner_circle, outer_circle), (inner_regions, outer_regions)



def interp_missing(ctmask):
    """
    Interpolate missing values in a thickness/density map using nearest neighbour interpolation.
    This function is designed to handle missing values in one of the ETDRS study grids, where missing
    values are represented by NaN values.

    Parameters
    ----------
    ctmask : numpy.ndarray
        A 2D array representing the thickness/density map, where missing values are NaNs

    Returns
    -------
    new_ctmask : numpy.ndarray
        The CT map with interpolated values for the missing regions.
    
    Notes
    -----
    - Missing values are those where the CT map contains NaN or zero.
    - The interpolation method uses `scipy.interpolate.NearestNDInterpolator` for nearest neighbour interpolation.
    """
    # Detect where values to be interpolated, values 
    # with known CT measurements, and values outside subregion
    ctmap_nanmask = np.isnan(ctmask)
    ctmap_ctmask = ctmask > 0
    ctmap_ctnone = ctmask != 0

    # Extract relevant coordinates to interpolate and evaluate at
    all_coords = np.array(np.where(ctmap_ctnone)).T
    ct_coords = np.array(np.where(ctmap_ctmask)).T
    ct_data = ctmask[ct_coords[:,0],ct_coords[:,1]]

    # Build new subregion mask with interpolated valuee
    new_ctmask = np.zeros_like(ctmask)
    interp_func = interpolate.NearestNDInterpolator
    ctmask_interp = interp_func(ct_coords, ct_data)
    new_ctmask[all_coords[:,0], all_coords[:,1]] = ctmask_interp(all_coords[:,0], all_coords[:,1])

    return new_ctmask



def measure_grid(thick_map, 
                 fovea, 
                 scale, 
                 eye, 
                 etdrs_size=[1000,3000,6000],
                 plot=False, 
                 slo=None, 
                 dtype=np.uint64, 
                 verbose=True,
                 fname=None, 
                 save_path=""):
    """
    Measure average thickness/density per subfield in the prescribed grid (ETDRS/Square).

    Parameters
    ----------
    thick_map : numpy.ndarray
        A 2D array representing the thickness/density map.
    
    fovea : tuple of int
        The (x, y)-coordinates of the fovea on the SLO.
    
    scale : float
        The scale of the image in microns-per-pixel.
    
    eye : str
        Laterality ("Right" or "Left") for determining the quadrant arrangement.

    etdrs_size : list
        List of micron diameters of the ETDRS circular grids

    rotate : int, optional, default=0
        The angle by which to rotate the grid (in degrees).
    
    grid_kwds : dict, optional, default={"etdrs_microns":(1000,3000,6000)}
        Additional keyword arguments for the grid creation, such as the diameters for ETDRS or the grid size for square grids.
    
    plot : bool, optional, default=False
        If True, plots the grid over the map and optionally an SLO image.
    
    slo : numpy.ndarray, optional, default=None
        The SLO image to be plotted with the grid, if provided.
    
    dtype : numpy.dtype, optional, default=np.uint64
        The data type to use for the grid values (e.g., uint64 for most metrics, float64 for CVI).

    verbose : bool, optiona, default=True
        Logs warning about interpolation needed for ETDRS subfield measurement.
    
    fname : str, optional, default=None
        The filename for logging or saving the plot, if specified.
    
    save_path : str, optional, default=""
        The path where the plot should be saved, if specified.

    Returns
    -------
    grid_dict : dict
        A dictionary containing the average measurements for each subfield.
    
    gridvol_dict : dict
        A dictionary containing the total volume (based on interpolated subfield area and thickness) 
        for each subfield.
    
    logging_list : list
        A list of logging messages that indicate the progress or any issues during processing.

    Notes
    -----
    - The function handles both square and ETDRS grid types, measuring average thickness/density per subfield.
    - Missing values within the grid are interpolated using nearest neighbour interpolation.
    """
    # Initialise logging, specify xy-scaling (if vessel density, this is in square microns, otherwise microns)
    logging_list = []
    rotate = 0
    delta_xy = scale / 1e9
    if fname is not None:
        if "vessel" not in fname:
            delta_xy *= scale

    # Get image shape and ensure fovea is defined properly
    img_shape = thick_map.shape
    if isinstance(fovea, int):
        fovea = (fovea, fovea)

    # Generate subfield binary masks and labelling (according to laterality) for the ETDRS grid
    output = create_etdrs_grid(scale, fovea, img_shape, rotate, etdrs_microns=etdrs_size)
    (circles,_), (central, _, _), (inner_regions, outer_regions) = output
    if eye == 'Right':
        etdrs_locs = ["superior", "temporal", "inferior", "nasal"]
    elif eye == 'Left':
        etdrs_locs = ["superior", "nasal", "inferior", "temporal"]
    etdrs_regions = ["inner", "outer"]
    grid_masks = [central] + inner_regions + outer_regions
    grid_subgrids = ["central"] + ["_".join([grid, loc]) for grid in etdrs_regions for loc in etdrs_locs]

    # All mask is the entire ROI (whole 6mm circle for ETDRS, whole 7mm square grid for Ppole grid)
    all_mask = (np.sum(np.array(grid_masks), axis=0) > 0).astype(int)

    # Most features are measured as integer, except for CVI
    if dtype == np.uint64:
        round_idx = 0
    elif dtype == np.float64:
        round_idx = 3

    # If interpolating (strongly recommended, otherwise -1s will be used in calculation for any missing regions
    # under-estimating true average value)
    all_subr_vals = []
    grid_dict = {}
    gridvol_dict = {}

    # Loop over subfield binary masks and labelling
    for sub,mask in zip(grid_subgrids, grid_masks):
        bool_mask = mask.astype(bool)
        mapmask = thick_map.copy()

        # Check for missing values, replace -1s to NaNs and interpolate using nearest neighbour
        # output to end-user proportion of missing data relative to the size of the subfield.
        if np.any(mapmask[bool_mask] == -1):
            prop_missing = np.round(100*np.sum(mapmask[bool_mask] == -1) / bool_mask.sum(),2)
            msg = f"{prop_missing}% missing values in {sub} region in ETDRS grid. Interpolating using nearest neighbour."
            logging_list.append(msg)
            if verbose:
                logging.warning(msg)
            mapmask[~bool_mask] = 0
            mapmask[mapmask == -1] = np.nan
            mapmask = interp_missing(mapmask)

        # Extract subfield values and interpolate to volume (if not measuring CVI) and take average
        mapmask[~bool_mask] = -1
        all_subr_vals.append(mapmask)
        subr_vals = mapmask[bool_mask]
        if dtype == np.uint64:
            gridvol_dict[sub] = np.round((delta_xy*subr_vals).sum(),3)
        grid_dict[sub] = np.round(dtype(subr_vals.mean()),round_idx)

    # Work out average thickness in the entire grid
    for mapmask in all_subr_vals:
        mapmask[mapmask == -1] = 0
    all_subr_mask = thick_map[all_mask.astype(bool)]
    max_val_etdrs = all_subr_mask.max()
    if dtype == np.uint64:
        gridvol_dict["all"] = np.round((delta_xy*all_subr_mask).sum(),3)
    grid_dict["all"] = np.round(dtype(all_subr_mask.mean()),round_idx)

    # Clipping value for visualisation is measured as 99.5th percentile to
    # ensure no extreme outliers at the edge of the map are included in calculation. 
    clip_val = np.quantile(thick_map[thick_map != -1], q=0.995)

    # Plot grid onto map and SLO
    if plot:
        _ = plot_grid(slo, 
                      thick_map, 
                      grid_dict, 
                      grid_masks,
                      fname=fname, 
                      save_path=save_path, 
                      clip=clip_val)

    return grid_dict, gridvol_dict, logging_list



def plot_grid(slo, 
              ctmap, 
              grid_data, 
              masks=None, 
              scale=11.49, 
              clip=None, 
              eye="Right", 
              fovea=np.array([384,384]),
              rotate=0, 
              etdrs_size=[1000,3000,6000],
              cbar=True, 
              img_shape=(768,768), 
              with_grid=True, 
              fname=None, 
              save_path=None, 
              transparent=False):
    """
    Plot the grid boundaries and associated average values on top of the SLO and thickness/density map.

    Parameters
    ----------
    slo : numpy.ndarray, optional
        The SLO image to be plotted beneath the grid and thickness/density map.
    
    ctmap : numpy.ndarray
        The thickness/density map.
    
    grid_data : dict
        A dictionary containing the grid's measurement values (average thickness or other metrics) for each subfield.
    
    masks : list of numpy.ndarray, optional
        List of binary masks defining subfield, used if the grid is precomputed or passed externally.
    
    scale : float, optional, default=11.49
        The scale of the SLO image in microns-per-pixel.
    
    clip : float, optional, default=None
        The maximum value for clipping the thickness/density map. If None, the 99.5 percentile value is used.
    
    eye : str, optional, default="Right"
        Laterality ("Right" or "Left") for determining the quadrant arrangement in the grid.
    
    fovea : numpy.ndarray, optional, default=np.array([384,384])
        (x,y)-coordinates of the fovea on the SLO.
    
    rotate : int, optional, default=0
        The angle by which to rotate the grid (in degrees).
    
    etdrs_size : dict, optional, default=[1000,3000,6000]
        Diameters of the three circular subfields of the ETDRS grid.
    
    cbar : bool, optional, default=True
        Whether to include a color bar in the plot.
    
    img_shape : tuple of int, optional, default=(768,768)
        The shape of the image (height, width).
    
    with_grid : bool, optional, default=True
        Whether to overlay the grid on the image.
    
    fname : str, optional, default=None
        The filename for saving the plot.
    
    save_path : str, optional, default=None
        The path where the plot should be saved.
    
    transparent : bool, optional, default=False
        Whether to save the plot with a transparent background.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated plot figure.

    Notes
    -----
    - The function supports both "etdrs" and "square" grid types.
    - It overlays grid measurements and boundaries onto the thickness map and optionally onto the SLO image.
    - The plot can be saved to a file if `fname` and `save_path` are provided.
    """
    # Build grid masks 
    if masks is None:
        img_shape = slo.shape
        output = create_etdrs_grid(scale, fovea, img_shape, rotate, etdrs_size)
        (_, _), (central, _, _), (inner_regions, outer_regions) = output
        if eye =='Right':
            etdrs_locs = ["superior", "temporal", "inferior", "nasal"]
        elif eye == 'Left':
            etdrs_locs = ["superior", "nasal", "inferior", "temporal"]
        masks = [central] + inner_regions + outer_regions
    M, N = img_shape
    
    # Detect centroids of masks
    centroids = [meas.centroid(region)[[1,0]] for region in masks]
    all_centroid = np.array([centroids[-1][0], centroids[-4][1]])

    # Generate grid boundaries
    bounds = np.sum(np.array([segmentation.find_boundaries(mask.astype(bool)) for mask in masks]), axis=0).clip(0,1)
    bounds = morph.dilation(bounds, footprint=morph.disk(radius=2))
    bounds = bscan_utils.generate_imgmask(bounds)

    # if clipping heatmap
    mask = ctmap < 0
    if clip is None:
        vmax = np.quantile(ctmap[ctmap != -1], q=0.995)
    else:
        vmax = clip

    # Plot grid on top of thickness map, ontop of SLO
    if cbar:
        figsize=(9,7)
    else:
        figsize=(9,9)
    fig, ax = plt.subplots(1,1,figsize=figsize)
    hmax = sns.heatmap(ctmap,
                    cmap = "rainbow",
                    alpha = 0.75,
                    zorder = 2,
                    vmax = vmax,
                    mask=mask,
                    cbar=cbar,
                    ax = ax)
    # if slo is not None:
    hmax.imshow(slo, cmap="gray",
            aspect = hmax.get_aspect(),
            extent = hmax.get_xlim() + hmax.get_ylim(),
            zorder = 1)
    ax.set_axis_off()
    if with_grid:
        ax.imshow(bounds, zorder=3)
        for (ct, coord) in zip(grid_data.values(), centroids):
            if isinstance(ct, str):
                fontsize=20
            else:
                if ct // 1 == 0:
                    fontsize=13.5 + (2-2*cbar)
                elif ct // 1000 == 0:
                    fontsize=16 + (2-2*cbar)
                else:
                    fontsize=14 + (2-2*cbar)
            ax.text(s=f"{ct}", x=coord[0], y=coord[1], zorder=4,
                    fontdict={"fontsize":fontsize, 
                              "fontweight":"bold", "ha":"center", "va":"center"})
            
        # Plot average CT across whole grid
        ax.text(s=grid_data["all"], 
                x=all_centroid[0] - 50*np.sign(N//2-all_centroid[0]), 
                y=all_centroid[1] - 50*np.sign(M//2-all_centroid[1]),
                zorder=4, fontdict={"fontsize":fontsize, "fontweight":"bold", "ha":"center", "va":"center"})

    # Save out
    if (save_path is not None) and (fname is not None): 
        fig.savefig(os.path.join(save_path, fname), bbox_inches="tight", transparent=transparent, pad_inches=0)
        plt.close()

    return fig



def plot_multiple_grids(all_dict):
    """
    Plot multiple ETDRS grid thickness/density values on top of SLO.

    Parameters
    ----------
    all_dict : dict
        A dictionary containing multiple grid data. The first key ('core') should hold the core image data (SLO, filename, and save path), 
        and subsequent keys should contain the maps and corresponding grid data for each grid to be plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure with multiple grids plotted.

    Notes
    -----
    - The function supports plotting multiple grids (e.g., for comparison) in a single figure.
    - Each grid's measurements (average thickness and grid volume) are overlaid onto both the thickness map and SLO image.
    """
    # Core plotting args
    with_grid = True
    transparent = False
    cbar = False
    measure_type = 'etdrs'
    etdrs_size = [1000,3000,6000]
    interp = True

    # Core map and SLO args
    slo, fname, save_path = all_dict['core']
    img_shape = slo.shape
    map_keys = list(all_dict.keys())[1:]
    fovea, scale, eye, rotate = all_dict[map_keys[0]][1:5]

    # Work out plotting figure subplots
    N = len(list(all_dict.keys()))-1
    figsize=(1,3)
    fig, axes = plt.subplots(1,3, figsize=(21,7))

    # Build grid masks 
    output = create_etdrs_grid(scale, fovea, img_shape, rotate, etdrs_microns=etdrs_size)
    (_, _), (central, _, _), (inner_regions, outer_regions) = output
    if eye == 'Right':
        etdrs_locs = ["superior", "temporal", "inferior", "nasal"]
    elif eye == 'Left':
        etdrs_locs = ["superior", "nasal", "inferior", "temporal"]
    masks = [central] + inner_regions + outer_regions

    # Detect centroids of masks
    centroids = [meas.centroid(region)[[1,0]] for region in masks]
    all_centroid = np.array([centroids[-1][0], centroids[-4][1]])

    # Generate grid boundaries
    bounds = np.sum(np.array([segmentation.find_boundaries(mask.astype(bool)) for mask in masks]), axis=0).clip(0,1)
    bounds = morph.dilation(bounds, footprint=morph.disk(radius=2))
    bounds = bscan_utils.generate_imgmask(bounds)

    plt_indexes = list(np.ndindex(figsize))
    for idx, plt_key in enumerate(map_keys):
        (ctmap, _, _, _, _, dtype, grid_data, gridvol_data) = all_dict[plt_key]

        ax = axes[idx]
        ax.imshow(slo, cmap='gray')
        ax.set_axis_off()
        ax.set_title(plt_key, fontsize=18)

        # clipping heatmap
        mask = ctmap < 0
        vmax = np.quantile(ctmap[ctmap != -1], q=0.995)

        # Plot grid on top of thickness map, ontop of SLO
        hmax = sns.heatmap(ctmap,
                        cmap = "rainbow",
                        alpha = 0.75,
                        zorder = 2,
                        vmax = vmax,
                        mask=mask,
                        cbar=cbar,
                        ax = ax)
        if slo is not None:
            hmax.imshow(slo, cmap="gray",
                    aspect = hmax.get_aspect(),
                    extent = hmax.get_xlim() + hmax.get_ylim(),
                    zorder = 1)
        ax.set_axis_off()
        if with_grid:
            ax.imshow(bounds, zorder=3)
            for (ct, coord) in zip(grid_data.values(), centroids):
                if isinstance(ct, str):
                    fontsize=20
                else:
                    if ct // 1 == 0:
                        fontsize=11
                    elif ct // 1000 == 0:
                        fontsize=12
                    else:
                        fontsize=10
                ax.text(s=f"{ct}", x=coord[0], y=coord[1], zorder=4,
                        fontdict={"fontsize":fontsize, 
                                  "fontweight":"bold", "ha":"center", "va":"center"})
                
            # Plot average CT across whole grid
            ax.text(s=grid_data["all"], 
                    x=all_centroid[0] - 50*np.sign(384-all_centroid[0]), 
                    y=all_centroid[1] - 50*np.sign(384-all_centroid[1]),
                    zorder=4, fontdict={"fontsize":fontsize, "fontweight":"bold", "ha":"center", "va":"center"})

    # Save out
    fig.savefig(os.path.join(save_path, fname+'.png'), bbox_inches="tight", transparent=False)
    plt.close()