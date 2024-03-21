import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import logging
import pandas as pd


def extract_bounds(mask):
    '''
    Given a binary mask, return the top and bottom boundaries,
    assuming the segmentation is fully-connected.
    '''
    # Stack of indexes where mask has predicted 1
    where_ones = np.vstack(np.where(mask.T)).T

    # Sort along horizontal axis and extract indexes where differences are
    sort_idxs = np.argwhere(np.diff(where_ones[:, 0]))

    # Top and bottom bounds are either at these indexes or consecutive locations.
    bot_bounds = np.concatenate([where_ones[sort_idxs].squeeze(),
                                 where_ones[-1, np.newaxis]], axis=0)
    top_bounds = np.concatenate([where_ones[0, np.newaxis],
                                 where_ones[sort_idxs + 1].squeeze()], axis=0)

    return (top_bounds, bot_bounds)


def select_largest_mask(binmask):
    '''
    Enforce connectivity of region segmentation
    '''
    # Look at which of the region has the largest area, and set all other regions to 0
    labels_mask = sk.measure.label(binmask)
    regions = sk.measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:, 0], rg.coords[:, 1]] = 0
    labels_mask[labels_mask != 0] = 1

    return labels_mask


def interp_trace(traces, align=True):
    '''
    Quick helper function to make sure every trace is evaluated
    across every x-value that it's length covers.
    '''
    new_traces = []
    for i in range(2):
        tr = traces[i]
        min_x, max_x = (tr[:, 0].min(), tr[:, 0].max())
        x_grid = np.arange(min_x, max_x)
        y_interp = np.interp(x_grid, tr[:, 0], tr[:, 1]).astype(int)
        interp_trace = np.concatenate([x_grid.reshape(-1, 1), y_interp.reshape(-1, 1)], axis=1)
        new_traces.append(interp_trace)

    # Crop traces to make sure they are aligned
    if align:
        top, bot = new_traces
        h_idx = 0
        top_stx, bot_stx = top[0, h_idx], bot[0, h_idx]
        common_st_idx = max(top[0, h_idx], bot[0, h_idx])
        common_en_idx = min(top[-1, h_idx], bot[-1, h_idx])
        shifted_top = top[common_st_idx - top_stx:common_en_idx - top_stx]
        shifted_bot = bot[common_st_idx - bot_stx:common_en_idx - bot_stx]
        new_traces = (shifted_top, shifted_bot)

    return tuple(new_traces)


def smart_crop(traces, check_idx=20, ythresh=1, align=True):
    '''
    Instead of defining an offset to check for and crop in utils.crop_trace(), which
    may depend on the size of the choroid itself, this checks to make sure that adjacent
    changes in the y-values of each trace are small, defined by ythresh.
    '''
    cropped_tr = []
    for i in range(2):
        _chor = traces[i]
        ends_l = np.argwhere(np.abs(np.diff(_chor[:check_idx, 1])) > ythresh)
        ends_r = np.argwhere(np.abs(np.diff(_chor[-check_idx:, 1])) > ythresh)
        if ends_r.shape[0] != 0:
            _chor = _chor[:-(check_idx - ends_r.min())]
        if ends_l.shape[0] != 0:
            _chor = _chor[ends_l.max() + 1:]
        cropped_tr.append(_chor)

    return interp_trace(cropped_tr, align=align)


def get_trace(binmask, threshold=None, align=False):
    '''
    Helper function to extract traces from a prediction mask.
    This thresholds the mask, selects the largest mask, extracts upper
    and lower bounds of the mask and crops any endpoints which aren't continuous.
    '''
    if threshold is not None:
        binmask = (binmask > threshold).astype(int)
    binmask = select_largest_mask(binmask)
    traces = extract_bounds(binmask)
    traces = smart_crop(traces, align=align)
    return traces


def curve_length(curve, scale=(11.49, 3.87)):
    """
    Calculate the length (in microns) of a curve defined by a numpy array of coordinates.

    This uses the euclidean distance and converts each unit step into the number of microns
    traversed in both axial directions.
    """
    # Scale constants
    xum_per_pix, yum_per_pix = scale

    # Calculate difference between pairwise consecutive coordinates of curve
    diff = np.abs((curve[1:] - curve[:-1]).astype(np.float64))

    # Convert pixel difference to micron difference
    diff[:, 0] *= xum_per_pix
    diff[:, 1] *= yum_per_pix

    # Length is total euclidean distance between all pairwise-micron-movements
    length = np.sum(np.sqrt(np.sum((diff) ** 2, axis=1)))

    return length


def curve_location(curve, distance=2000, ref_idx=400, scale=(11.49, 3.87), verbose=0, image_axis=True):
    """
    Given a curve, what two coordinates are *distance* microns away from some coordinate indexed by
    *ref_idx*.

    This uses the euclidean distance and converts each unit step into the number of microns
    traversed in both axial directions.
    """
    # Work out number of microns per unit pixel movement
    N = curve.shape[0]

    # Scale constants
    xum_per_pix, yum_per_pix = scale

    # If measuring along choroid axis
    if not image_axis:

        # Calculate difference between pairwise consecutive coordinates of curve
        diff_r = np.abs((curve[1 + ref_idx:] - curve[ref_idx:-1]).astype(np.float64))
        diff_l = np.abs((curve[::-1][1 + (N - ref_idx):] - curve[::-1][(N - ref_idx):-1]).astype(np.float64))

        # Convert pixel difference to micron difference
        diff_r[:, 0] *= xum_per_pix
        diff_r[:, 1] *= yum_per_pix
        diff_l[:, 0] *= xum_per_pix
        diff_l[:, 1] *= yum_per_pix

        # length per movement is euclidean distance between pairwise-micron-movements
        length_l = np.sqrt(np.sum((diff_l) ** 2, axis=1))
        cumsum_l = np.cumsum(length_l)
        length_r = np.sqrt(np.sum((diff_r) ** 2, axis=1))
        cumsum_r = np.cumsum(length_r)

        # Work out largest index in cumulative length sum where it is smaller than *distance*
        idx_l = ref_idx - np.argmin(cumsum_l < distance)
        idx_r = ref_idx + np.argmin(cumsum_r < distance)
        if (idx_l == ref_idx) and distance > 200:
            if verbose == 1:
                logging.warning(f"""Segmentation not long enough for {distance}um left of fovea.
                    Extend segmentation or reduce macula_rum to prevent this from happening.
                    Returning 0s.""")
            return None
        if (idx_r == ref_idx) and distance > 200:
            if verbose == 1:
                logging.warning(f"""Segmentation not long enough for {distance}um right of fovea. 
                    Extend segmentation or reduce macula_rum to prevent this from happening.
                    Returning 0s.""")
            return None

    # If measuring along image axis
    else:
        d_px = int(distance / xum_per_pix)

        if ((ref_idx - d_px) < 0) or ((d_px + ref_idx) > N):
            if verbose == 1:
                logging.warning(f"""Segmentation not long enough for {distance}um right of fovea. 
                    Extend segmentation or reduce macula_rum to prevent this from happening.
                    Returning 0s.""")
            return None
        idx_l, idx_r = ref_idx - d_px, ref_idx + d_px

    return idx_l, idx_r


def _check_offset(offset, offsets_lr, N_pts):
    '''
    Quick helper function to check if offset is too large, and deal with it if so
    '''
    (offset_l, offset_r) = offsets_lr
    if offset_l < 0:
        offset_l = 0
        logging.warning(f"Offset {offset} too far to the left, choosing index {offset_l}")

    if offset_r >= N_pts:
        offset_r = N_pts - 1
        logging.warning(f"Offset {offset} too far to the right, choosing index {offset_r}")

    return offset_l, offset_r


def nearest_coord(trace, coord, offset=15, columnwise=False):
    """
    Given a coordinate, find the nearest coordinate along trace and return it.

    INPUTS:
    ------------------------
        trace (np.array) : Upper or lower choroid boundary

        coord (np.array) : Single xy-coordinate.

        offset (int) : Integer index to select either side of reference point
        along trace for deducing locally perpendicular CT measurement.

        columnwise (bool) : If flagged, nearest coordinate on the trace is
            the one at the same column index.

    RETURNS:
    ------------------------
        trace_refpt (np.array) : Point along trace closest to coord.

        offset_pts (np.array) : Points offset distance from trace_refpt for deducing
            local tangent line.
    """
    N_pts = trace.shape[0]

    # Work out closest coordinate on trace to coord
    if not columnwise:
        fovea_argmin = np.argmin(np.sum((trace - coord) ** 2, axis=1))
        trace_refpt = trace[fovea_argmin]
    else:
        fovea_argmin = coord[0] - trace[0, 0]
        trace_refpt = trace[fovea_argmin]

    # Prevent error by choosing maximum offset, if offset is too large for given trace
    offset_l, offset_r = fovea_argmin - offset, fovea_argmin + offset
    offset_l, offset_r = _check_offset(offset, (offset_l, offset_r), N_pts)
    offset_pts = trace[[offset_l, offset_r]]

    return trace_refpt, offset_pts


def construct_line(p1, p2):
    """
    Construct straight line between two points p1 and p2.

    INPUTS:
    ------------------
        p1 (1d-array) : 2D pixel coordinate.

        p2 (1d-array) : 2D pixel coordinate.

    RETURNS:
    ------------------
        m, c (floats) : Gradient and intercept of straight line connection points p1 and p2.
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


def compute_measurement(reg_mask,
                        vess_mask=None,
                        fovea: [tuple, np.ndarray] = None,
                        scale: tuple[int, int] = (11.49, 3.87),
                        macula_rum: int = 3000,
                        N_measures: int = 3,
                        N_avgs=0,
                        offset=15,
                        plottable=False,
                        verbose=0):
    """
    Compute measurements of interest, that is thickness and area using the reg_mask and
    CVI (optional) using the vess_mask.

    Inputs:
    -------------------------
    reg_mask : binary mask segmentation of the choroidal space.
    vess_mask : segmentation of the choroidal space. Need not be binary.
    fovea : Fovea coordinate to define fovea-centred ROI. Default is center column,row of mask
    scale: Microns-per-pixel in x and z directions. Default setting is Heidelberg scaling
        for emmetropic individual.
    macula_rum : Micron radius either side of fovea to measure. Default is the largest region in
        ETDRS grid (3mm).
    N_measures : Number of thickness measurements to make across choroid. Default is three: subfoveal and
        a single temporal/nasal location.
    N_avgs : Number of adjecent thicknesses to average at each location to enforce robustness. Default is
        one column, set as 0.
    offset : Number of pixel columns to define tangent line around upper boundary reference points, for
        accurate, perpendicular detection of lower boundary points.
    plottable : If flagged, returnboundary points defining where thicknesses have been measured, and binary masks
        where choroid area and vascular index have been measured.
    verbose : Log to user regarding segmentation length.

    Outputs:
    -------------------------
    ct : choroid thickness, an integer value per location measured (N_measures, the average of N_avgs adjacent thickness values)
    ca : choroid area in a macula_rum microns, fovea-centred region of interest.
    cvi : choroid vascular index in a macula_rum microns, fovea-centred region of interest.
    """
    measurements = []

    # Constants
    N_measures = max(N_measures + 1, 3) if N_measures % 2 == 0 else max(N_measures, 3)
    N_avgs = N_avgs + 1 if N_avgs % 2 != 0 else N_avgs
    micron_pixel_x, micron_pixel_y = scale
    pixels_per_micron = 1 / micron_pixel_x

    # Organise region mask
    if isinstance(reg_mask, np.ndarray):
        if reg_mask.ndim == 2:
            traces = get_trace(reg_mask, None, align=True)
        else:
            traces = reg_mask.copy()
    elif isinstance(reg_mask, tuple):
        traces = interp_trace(reg_mask)

    # If fovea is known - if not, take as halfway along region
    # segmentation
    if fovea is not None:
        if isinstance(fovea, tuple):
            fovea = np.array(fovea)
        ref_pt = fovea
    else:
        x_N = int(0.5 * (traces[0].shape[0] + traces[1].shape[0]))
        # x_st = int(0.5*(traces[0,0] + traces[1,0]))
        x_st = int(0.5 * (traces[0][0, 0] + traces[1][0, 0]))
        x_half = x_N // 2 + x_st
        y_half = traces[0][:, 1].mean()
        ref_pt = np.array([x_half, y_half])

    # Work out reference point along upper boundary closest to fovea
    # and re-adjust reference point on upper boundary to align with indexing
    top_chor, bot_chor = traces
    rpechor_stx, chorscl_stx = int(top_chor[0, 0]), int(bot_chor[0, 0])
    rpechor_refpt, offset_pts = nearest_coord(top_chor, ref_pt, offset, columnwise=False)
    ref_idx = rpechor_refpt[0] - rpechor_stx

    # Set up list of micron distances either side of reference point, dictated by N_measures
    delta_micron = 2 * macula_rum // (N_measures - 1)
    delta_i = [i for i in range((N_measures - 1) // 2 + 1)]
    micron_measures = np.array([i * delta_micron for i in delta_i])

    # Locate coordinates along the upper boundary at
    # equal spaces away from foveal pit until macula_rum
    curve_indexes = [
        curve_location(top_chor, distance=d, ref_idx=ref_idx, scale=scale, verbose=verbose, image_axis=True) for d in
        micron_measures]

    # To catch if we cannot make measurement macula_rum either side of reference point, return 0s.
    if None in curve_indexes:
        return np.array(N_measures * [0], dtype=np.int64), 0, 0

    # Collect reference points along upper boundary - we can compute more robust thickness as
    # average value of several adjacent thicknesses using N_avgs. ALso compute corresponding
    # perpendicular reference points along lower boundary
    rpechor_pts = np.unique(np.array(
        [top_chor[[idx + np.arange(-N_avgs // 2, N_avgs // 2 + 1)]] for loc in curve_indexes for idx in loc]).reshape(
        -1, 2), axis=0)
    st_Bx = bot_chor[0, 0]
    chorscl_pts = bot_chor[rpechor_pts[:, 0] - st_Bx]
    chorscl_pts = chorscl_pts.reshape(N_measures, N_avgs + 1, 2)
    rpechor_pts = rpechor_pts.reshape(N_measures, N_avgs + 1, 2)

    boundary_pts = np.concatenate([rpechor_pts.reshape(*chorscl_pts.shape), chorscl_pts], axis=-1).reshape(
        *chorscl_pts.shape, 2)

    # Compute choroid thickness at each reference point.
    delta_xy = np.abs(np.diff(boundary_pts, axis=boundary_pts.ndim - 2)) * np.array([micron_pixel_x, micron_pixel_y])
    choroid_thickness = np.rint(np.nanmean(np.sqrt(np.square(delta_xy).sum(axis=-1)), axis=1)).astype(int)[:, 0]
    measurements.append(choroid_thickness)

    # Compute choroid area
    area_bnds_arr = np.swapaxes(boundary_pts[[0, -1], N_avgs // 2], 0, 1).reshape(-1, 2)
    if np.any(np.isnan(area_bnds_arr)):
        logging.warning(f"""Segmentation not long enough for {macula_rum}um using {offset} pixel offset and {N_avgs} column averaging.
            Extend segmentation or reduce macula_rum/N_avgs/offset to prevent under-measurement.
            Returning 0s.""")
        return np.array(N_measures * [0], dtype=np.int64), 0, 0

    choroid_area, plot_output = compute_area_enclosed(traces, area_bnds_arr.astype(int), scale=scale, plot=True)
    chor_pixels, (x_range, y_range), (left_x, right_x) = plot_output
    measurements.append(choroid_area)

    if vess_mask is not None:
        # Compute CVI --- choroidal vessel area is whatever pixels in keep_pixel
        # which are flagged as 1 in vess_mask
        vessel_area = 0
        vessel_pixels = []
        xchor = np.unique(chor_pixels[:, 0]).astype(np.int64)
        chor_pixels = chor_pixels.astype(np.int64)
        for x in xchor:
            col = chor_pixels[chor_pixels[:, 0] == x]
            bmap_col = vess_mask[col[:, 1] - 1, x]

            vessel_area += bmap_col.sum()
            vessel_pixels.append(col[bmap_col == 1])
        vessel_pixels = np.concatenate(vessel_pixels)

        # Total pixel-choroid area is the number of pixels contained with the ROI
        # CVI is just # vessel pixels divided by # choroid area pixels
        total_chor_area = chor_pixels.shape[0]
        choroid_cvi = np.round(vessel_area / total_chor_area, 5)
        measurements.append(choroid_cvi)

        # Other metrics like EVI, vessel area and intersitial area (in mm2)
        micron_area = micron_pixel_x * micron_pixel_y
        choroid_vessel_area = np.round(1e-6 * micron_area * vessel_area, 6)
        measurements.append(choroid_vessel_area)

    if plottable:
        ca_mask = np.zeros_like(reg_mask)
        ca_mask[chor_pixels[:, 1], chor_pixels[:, 0]] = 1
        if vess_mask is not None:
            cvi_mask = np.zeros_like(reg_mask)
            cvi_mask[vessel_pixels[:, 1], vessel_pixels[:, 0]] = 1
        return measurements, (boundary_pts, ca_mask, cvi_mask)
    else:
        return measurements


def compute_area_enclosed(traces,
                          area_bnds_arr,
                          scale=(11.49, 3.87),
                          plot=False):
    """
    Function which, given traces and four vertex points defining the smallest irregular quadrilateral to which
    the choroid area is enclosed in, calculate the area to square millimetres.

    INPUTS:
    ---------------------
        traces (3-tuple) : Tuple storing upper and lower boundaries of trace

        area_bnds_arr (np.array) : Four vertex pixel coordnates defining the smallest irregular quadrilateral
            which contains the choroid area of interest.

        scale (3-tuple) : Tuple storing pixel_x-pixel_y-micron scalar constants.

        plot (bool) : If flagged, output information to visualise area calculation, including the points contained
            in the quadrilateral and the smallest rectangular which contains the irregular quadrilateral.

    RETURNS:
    --------------------
        choroid_mm_area (float) : Choroid area in square millimetres.

        plot_outputs (list) : Information to plot choroid area calculation.
    """
    # Extract reference points scale and traces
    top_chor, bot_chor = traces
    rpechor_stx, chorscl_stx = top_chor[0, 0], bot_chor[0, 0]
    rpechor_ref, chorscl_ref = area_bnds_arr[:2], area_bnds_arr[2:]

    # Compute microns-per-pixel and how much micron area a single 1x1 pixel represents.
    micron_pixel_x, micron_pixel_y = scale
    micron_area = micron_pixel_x * micron_pixel_y

    # Work out range of x- and y-coordinates bound by the area, building the smallest rectangular
    # region which overlaps the area of interest fully
    x_range = np.arange(area_bnds_arr[:, 0].min(), area_bnds_arr[:, 0].max() + 1)
    y_range = np.arange(min(top_chor[x_range[0] - rpechor_stx: x_range[-1] - rpechor_stx + 1, 1].min(),
                            area_bnds_arr[:, 1].min()),
                        max(bot_chor[x_range[0] - chorscl_stx: x_range[-1] - chorscl_stx + 1, 1].max(),
                            area_bnds_arr[:, 1].max()) + 1)
    N_y = y_range.shape[0]

    # This defines the left-most perpendicular line and right-most perpendicular line
    # for comparing with coordinates from rectangular region
    left_m, left_c = construct_line(rpechor_ref[0], chorscl_ref[0])
    right_m, right_c = construct_line(rpechor_ref[1], chorscl_ref[1])
    if left_m != np.inf:
        left_x = ((y_range - left_c) / left_m).astype(np.int64)
    else:
        left_x = np.array(N_y * [rpechor_ref[0][0]])
    if right_m != np.inf:
        right_x = ((y_range - right_c) / right_m).astype(np.int64)
    else:
        right_x = np.array(N_y * [rpechor_ref[1][0]])
    # The rectangular region above needs reduced to only containing coordinates which lie
    # above the Chor-Sclera boundary, below the RPE-Choroid boundary, to the right of the
    # left-most perpendicular line and to the left of the right-most perpendicular line.
    keep_pixel = []

    # We vectorise check by looping across x_range and figuring out if each coordinate
    # in the column satisfies the four checks described above
    for x in x_range:
        # Extract column
        col = np.concatenate([x * np.ones(N_y)[:, np.newaxis], y_range[:, np.newaxis]], axis=1)

        # Define upper and lower bounds at this x-position
        top, bot = top_chor[x - rpechor_stx], bot_chor[x - chorscl_stx]

        # Check all 4 conditions and make sure they are all satisfied
        cond_t = col[:, 1] >= top[1]
        cond_b = col[:, 1] <= bot[1]
        cond_l = x >= left_x
        cond_r = x < right_x
        col_keep = col[cond_t & cond_b & cond_l & cond_r]
        keep_pixel.append(col_keep)

    # All pixels bound within the area of interest
    keep_pixel = np.concatenate(keep_pixel)

    # Calculate area (in square mm)
    choroid_pixel_area = keep_pixel.shape[0]
    choroid_mm_area = np.round(1e-6 * micron_area * choroid_pixel_area, 6)

    # If plotting, reutrn pixels used to compute  area
    if plot:
        plot_outputs = [keep_pixel, (x_range, y_range), (left_x, right_x)]
        outputs = [choroid_mm_area, plot_outputs]
    else:
        outputs = choroid_mm_area

    return outputs
