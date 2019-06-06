def identify_modality(path):
    from os import listdir
    from os.path import join, basename

    def get_mr_acquisition(file_list, str1, str2):
        from os.path import basename
        res = [file for file in file_list
               if (str1 in basename(file).lower()) and (str2 in basename(file).lower())]
        if len(res) != 1:
            raise IOError(str(len(res)) + ' file(s) found for ' + str1 + ' and ' + str2)
        return res[0]

    mri_list = [join(path, f) for f in listdir(path) if basename(f).lower().endswith('nii')]

    # histo_path = [join(path, f) for f in listdir(path) if basename(f).lower().startswith('histo')]
    # if len(histo_path) != 1:
    #    raise ValueError('There can only be one high rez histological-cut')

    sos_mge_path = get_mr_acquisition(mri_list, 'sos', 'mge')
    sos_msme_path = get_mr_acquisition(mri_list, 'sos', 'msme')
    t1_path = get_mr_acquisition(mri_list, 't1', '')
    t2_msme_path = get_mr_acquisition(mri_list, 't2', 'msme')
    t2s_path = get_mr_acquisition(mri_list, 't2s', '')

    return {"sos_mge": sos_mge_path,
            "sos_msme": sos_msme_path,
            "t1": t1_path,
            "t2_msme": t2_msme_path,
            "t2s": t2s_path}


def coregister(output_path, ref, mov, ref_name):
    import os
    import SimpleITK as sitk

    output_folder = os.path.join(output_path, "coreg_with_" + ref_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    coreg_img = os.path.join(output_folder, "coreg_" + os.path.basename(mov))

    dst = None
    elastixImageFilter = sitk.ElastixImageFilter()
    try:
        elastixImageFilter.SetFixedImage(sitk.ReadImage(ref))
        elastixImageFilter.SetMovingImage(sitk.ReadImage(mov))
    except IOError:
        print("Could not read " + ref + " or " + mov)
    elastixImageFilter.SetParameterMap(sitk.GetDefaultParameterMap("translation"))
    try:
        elastixImageFilter.Execute()
        sitk.WriteImage(elastixImageFilter.GetResultImage(), coreg_img)
        dst = coreg_img
    except RuntimeError:
        print("Error while performing registration between " + ref + " and " + mov)
        dst = mov
    return dst


def normalize_vol(output_path, path_vol, threshold):
    import nibabel as nib
    import os
    import numpy as np

    orig_nifti = nib.load(path_vol)
    img = orig_nifti.get_data()
    img[img > threshold] = np.nan
    img = np.squeeze(img)
    nifti_out = nib.Nifti1Image(img, orig_nifti.affine, header=orig_nifti.header)
    filename = os.path.basename(path_vol)
    if not os.path.exists(os.path.join(output_path, 'normalized')):
        os.mkdir(os.path.join(output_path, "normalized"))
    normalized_filename = os.path.join(output_path, "normalized", "normalized_" + filename)
    nib.save(nifti_out, normalized_filename)
    return normalized_filename