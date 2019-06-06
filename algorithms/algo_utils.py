def get_mask_data(ref, kernel_size=15, display=False):

    from matplotlib import pyplot as plt
    import numpy as np
    from os.path import basename
    from scipy.ndimage.filters import generic_filter
    from skimage.filters import threshold_otsu
    from scipy.ndimage.measurements import label
    from skimage.morphology import binary_dilation
    from skimage.morphology import binary_erosion
    import nibabel as nib

    ref_np = nib.load(ref).get_data()

    # Computation
    measure = generic_filter(ref_np, np.std, size=kernel_size, mode='mirror')
    otsu_val = threshold_otsu(measure)
    wanted_values = measure < otsu_val
    labeled_mask, num_features = label(wanted_values)
    labeled_mask = np.array(labeled_mask)
    unique_label = np.unique(labeled_mask)
    max_number_of_px = 0
    max_label = 0
    for i, ul in enumerate(unique_label):
        if ul != 0:
            mask_lab = labeled_mask == ul
            sum_px_current_label = np.sum(mask_lab)
            if sum_px_current_label > max_number_of_px:
                max_number_of_px = sum_px_current_label
                max_label = ul
    roi = labeled_mask == max_label
    dilated_roi = binary_dilation(roi, np.ones((10, 10)))
    final_roi = binary_erosion(dilated_roi, np.ones((10, 10)))
    final_ref = np.copy(ref_np)
    final_ref[np.invert(dilated_roi)] = np.nan

    if display:
        fsize = 9
        myfig = plt.figure(3)
        myfig.suptitle("masking data for using std metrics on " + basename(ref), fontsize=fsize)

        plt.subplot(3, 3, 1)
        subplt = plt.imshow(ref, cmap='gray')
        subplt.axes.get_xaxis().set_visible(False)
        subplt.axes.get_yaxis().set_visible(False)
        plt.title(ref, fontsize=fsize)

        plt.subplot(3, 3, 2)
        subplt = plt.imshow(measure, cmap='gray')
        subplt.axes.get_xaxis().set_visible(False)
        subplt.axes.get_yaxis().set_visible(False)
        plt.title("std filter of " + ref + " - kernel size = " + str(kernel_size) + " x " + str(kernel_size),
                  fontsize=fsize)

        plt.subplot(3, 3, 3)
        subplt = plt.hist(measure.ravel(), bins=500, range=(np.nanmin(measure), np.nanmax(measure)), fc='k', ec='k')
        plt.title("Histogram of measure - Otsu threshold = " + str(otsu_val), fontsize=fsize)

        plt.subplot(3, 3, 4)
        subplt = plt.imshow(wanted_values, cmap='gray')
        subplt.axes.get_xaxis().set_visible(False)
        subplt.axes.get_yaxis().set_visible(False)
        plt.title("Filtered image thresholded using Otsu", fontsize=fsize)

        plt.subplot(3, 3, 5)
        subplt = plt.imshow(labeled_mask, cmap='hot')
        subplt.axes.get_xaxis().set_visible(False)
        subplt.axes.get_yaxis().set_visible(False)
        plt.title("Labelled mask - n_label = " + str(num_features), fontsize=fsize)

        plt.subplot(3, 3, 6)
        subplt = plt.imshow(roi, cmap='gray')
        subplt.axes.get_xaxis().set_visible(False)
        subplt.axes.get_yaxis().set_visible(False)
        plt.title("Extraction of largest ROI ", fontsize=fsize)

        plt.subplot(3, 3, 7)
        subplt = plt.imshow(final_roi, cmap='gray')
        subplt.axes.get_xaxis().set_visible(False)
        subplt.axes.get_yaxis().set_visible(False)
        plt.title("Flood-fill", fontsize=fsize)

        plt.subplot(3, 3, 8)
        subplt = plt.imshow(final_ref, cmap='gray')
        subplt.axes.get_xaxis().set_visible(False)
        subplt.axes.get_yaxis().set_visible(False)
        plt.title("final image", fontsize=fsize)
    return final_roi
