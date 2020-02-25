# histo-mri

The goal of this study is to allow the detection of neurofibrillary tangles (NFT) in magnetic resonance images (MR). Visually, NFT cannot be seen on the MR images. We aimed to test whether NFT can be identified in the signal of MR acquisitions using sophisticated data analysis methods. 
NFT are visible on histological cut of mouse brain. So when histological cuts are provided, matching with MR images can be done to label NFT/NO-NFT. 

- How to use the code ? 

In file `main.py`, set `output_folder` to the desired output folder. 
`input_folder` must link to a folder organized as follows: 
```
├── TG03
│   ├── Histo_B3A1_TG03.jpg
│   ├── Sos_MGE_87.nii
│   ├── Sos_MSME_79.nii
│   ├── T1_80.nii
│   ├── T2_MSME_79.nii
│   └── T2s_MGE_87.nii
├── TG04
│   ├── Histo_B3B1_TG04.jpg
│   ├── Sos_MGE_66.nii
│   ├── Sos_MSME_67.nii
│   ├── T1_71.nii
│   ├── T2_MSME_67.nii
│   ├── T2s_MGE_66.nii
│   └── coreg
├── TG05
│   ├── Histo_B3B3_TG05.jpg
│   ├── Sos_MGE_70.nii
│   ├── Sos_MSME_71.nii
│   ├── T1_73.nii
│   ├── T2_MSME_71.nii
│   └── T2s_MGE_70.nii
├── TG06
│   ├── Histo_B3C2_TG06.jpg
│   ├── Sos_MGE_68.nii
│   ├── Sos_MSME_70.nii
│   ├── T1_71.nii
│   ├── T2_MSME_70.nii
│   └── T2s_MGE_68.nii
├── WT03
│   ├── Histo_B3A2_WT03.jpg
│   ├── Sos_MGE_73.nii
│   ├── Sos_MSME_74.nii
│   ├── T1_75.nii
│   ├── T2_MSME_74.nii
│   └── T2s_MGE_73.nii
├── WT04
│   ├── Histo_B3B2_WT04.jpg
│   ├── Sos_MGE_175.nii
│   ├── Sos_MSME_176.nii
│   ├── T1_171.nii
│   ├── T2_MSME_176.nii
│   └── T2s_MGE_175.nii
├── WT05
│   ├── Histo_B3B4_WT05.jpg
│   ├── Sos_MGE_13.nii
│   ├── Sos_MSME_10.nii
│   ├── T1_16.nii
│   ├── T2_MSME_10.nii
│   └── T2s_MGE_13.nii
└── WT06
    ├── Histo_B3C1_WT06.jpg
    ├── Sos_MGE_69.nii
    ├── Sos_MSME_70.nii
    ├── T1_73.nii
    ├── T2_MSME_70.nii
    └── T2s_MGE_69.nii
 ```
 
 The only thing important in those filenames are the keyword `sos_mge`, `sos_msme`, `t1`, `t2_msme`, `t2s_mge` and `histo`. Numbers do not really matter. 
 
 - What does the code do ?
 
 The different steps carried out in the code are detailled in the document `/network/lustre/dtlake01/aramis/projects/histomri/rapport_Arnaud_histoMRI-2020_02_25_Final.pdf`
