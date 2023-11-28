# Multimodal Models in Oncology: Enhancing Treatment Response Evaluation and Prognostic Accuracy

## Prognosis
<details>
<summary>June 2022, <i>Nature Cancer</i>, <b>Multimodal data integration using machine learning improves risk stratification of high-grade serous ovarian cancer</b></summary>

[Paper](https://www.nature.com/articles/s43018-022-00388-9)
- **Journal:** Nature Cancer
- **Published Date:** June 2022
- **Cancer:** Ovarian Cancer
- **Modalities:** Radiological CTs, Pathological images, Clinical data
- **Data Source:** MSKCC, TCGA-OV
- **Patients**: 444 patients, including 296 patients treated at the Memorial Sloan Kettering Cancer Center (MSKCC) and 148 patients from The Cancer Genome Atlas Ovarian Cancer (TCGA-OV); 40 test cases were randomly sampled from the entire pool of patients with all data modalities available for analysis, and the resting of 404 patients for training
  - 404 training patients: 243 had H&E WSIs, 245 had adnexal lesions on pre-treatment CE-CT, 251 had omental implants on pre-treatment CE-CT
  - 40 test patients: all had omental lesions on CE-CT, H&E WSIs
- **Pipeline**: 
    - using PyRadiomics for Radiological CTs; pre-training a ResNet-18 as histopathological tissue-type classifier and for extracting cell type features and tissue-type features; encoding clinical data as binary variables or one-hot categorical variables
    - using univariate Cox proportional hazards model to select features
    - employing a multivariable Cox model for late fusing
- **Fusion Mode:** Late-fusion, using a multivariate Cox model to integrate unimodal submodels’ predictions

</details>

<details>
<summary>April 2022, <i>Artificial Intelligence in Medicine</i>, <b>A Multi-modal Fusion Framework Based on Multi-task Correlation Learning for Cancer Prognosis Prediction (MultiCoFusion)</b></summary>

[Paper](https://www.sciencedirect.com/science/article/pii/S0933365722000252) [Code]()
- **Journal:** Artificial Intelligence in Medicine
- **Published Date:** April 2022
- **Cancer:** Brain Lower Grade Glioma, Glioblastoma Multiforme
- **Modalities:** Pahological images, Gene (mRNA)
- **Data Source:** TCGA-LGG, TCGA-GBM
- **Patients**: 470 patients
    - For pathological images, [A pre-proposed dataset](https://github.com/mahmoodlab/PathomicFusion), consisting of 954 ROIs from WSIs for 470 patients
    - For gene data, one patient (TCGA-06-0152) is missing mRNA expression data, and the rest of 469 patients contain 953 mRNA samples. For cancer grade classification, i.e., Grade II (393 samples), III (408), IV (152). Each mRNA expression data have 10673 genes.
    - 80% for training and 20% for testing
- **Pipeline**: 
    - pre-trained ResNet-152 for histopathological images; a sparse graph convolutional network (SGCN) for mRNA expression data
    - fusing these representations by a FCN
    - the fused FCN is a multi-task shared network, outputing survival analysis and cancer grade classification simultaneously
- **Fusion Mode:** Middle-fusion

</details>

<details>
<summary>, <i></i>, <b></b></summary>

[Paper]() [Code]()
- **Journal:** 
- **Published Date:** 
- **Cancer:** 
- **Modalities:** 
- **Patients**: 
- **Fusion Mode:** 

</details>

<details>
<summary>, <i></i>, <b></b></summary>

[Paper]() [Code]()
- **Journal:** 
- **Published Date:** 
- **Cancer:** 
- **Modalities:** 
- **Patients**: 
- **Fusion Mode:** 

</details>