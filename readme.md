# Multimodal Models in Oncology: Enhancing Treatment Response Evaluation and Prognostic Accuracy

## Treatment Response Evaluation

## Prognosis
<details>
<summary>June 2022, <i>Nature Cancer</i>, <b>Multimodal data integration using machine learning improves risk stratification of high-grade serous ovarian cancer</b></summary>

[Paper](https://www.nature.com/articles/s43018-022-00388-9)
[Code](https://github.com/kmboehm/onco-fusion)
<!-- - **Journal:** Nature Cancer
- **Published Date:** June 2022 -->
- **Cancer:** Ovarian Cancer
- **Modalities:** Radiological CTs, Pathological images, Clinical data
- **Data Source:** MSKCC, TCGA-OV
- **Patients:** 444 patients, including 296 patients treated at the Memorial Sloan Kettering Cancer Center (MSKCC) and 148 patients from The Cancer Genome Atlas Ovarian Cancer (TCGA-OV); 40 test cases were randomly sampled from the entire pool of patients with all data modalities available for analysis, and the resting of 404 patients for training
  - 404 training patients: 243 had H&E WSIs, 245 had adnexal lesions on pre-treatment CE-CT, 251 had omental implants on pre-treatment CE-CT
  - 40 test patients: all had omental lesions on CE-CT, H&E WSIs
- **Pipeline:**
    - using PyRadiomics for Radiological CTs; pre-training a ResNet-18 as histopathological tissue-type classifier and for extracting cell type features and tissue-type features; encoding clinical data as binary variables or one-hot categorical variables
    - using univariate Cox proportional hazards model to select features
    - employing a multivariable Cox model for late fusing
- **Fusion Mode:** Early/Late-fusion, using a multivariate Cox model to integrate unimodal submodels’ predictions

</details>

<details>
<summary>April 2022, <i>Artificial Intelligence in Medicine</i>, <b>A Multi-modal Fusion Framework Based on Multi-task Correlation Learning for Cancer Prognosis Prediction (MultiCoFusion)</b></summary>

[Paper](https://www.sciencedirect.com/science/article/pii/S0933365722000252) 
<!-- - **Journal:** Artificial Intelligence in Medicine
- **Published Date:** April 2022 -->
- **Cancer:** Brain Lower Grade Glioma, Glioblastoma Multiforme
- **Modalities:** Pahological images, Gene (mRNA)
- **Data Source:** TCGA-LGG, TCGA-GBM
- **Patients:** 470 patients
    - For pathological images, [a pre-proposed dataset](https://github.com/mahmoodlab/PathomicFusion), consisting of 954 ROIs from WSIs for 470 patients
    - For gene data, one patient (TCGA-06-0152) is missing mRNA expression data, and the rest of 469 patients contain 953 mRNA samples. For cancer grade classification, i.e., Grade II (393 samples), III (408), IV (152). Each mRNA expression data have 10673 genes.
    - 80% for training and 20% for testing
- **Pipeline:**
    - pre-trained ResNet-152 for histopathological images; a sparse graph convolutional network (SGCN) for mRNA expression data
    - fusing these representations by a FCN
    - the fused FCN is a multi-task shared network, outputing survival analysis and cancer grade classification simultaneously
- **Fusion Mode:** Middle-fusion

</details>

<details>
<summary>December 2017, <i>Cell Systems</i>, <b>Association of Omics Features with Histopathology Patterns in Lung Adenocarcinoma</b></summary>

[Paper](https://www.cell.com/cell-systems/pdf/S2405-4712(17)30484-2.pdf)

- **Cancer:** Lung Adenocarcinoma
- **Modalities:** Pathological Images, Pathological Reports, Gene (RNA sequencing), Proteomics
- **Data Source:** TCGA
- **Patients:** 538 patients
- **Pipeline:** 
    - converting pathological images into overlapping tiles and selected the ROIs to extract quantitative features (i.e. size, shape, intensity distribution, and texture features); identifing pathology grade from pathology reports; collecting gene and protein expression data by RNA sequencing and reverse-phase protein array
    - employing feature selection on the training set
    - building a random forest model for prognostic prediction
- **Fusion Mode:** Early/Late-fusion, using a random forest model to integrate multimodal features

</details>

## Others

<details>
<summary>June 2023, <i>Nature Biomedical Engineering</i>, <b>A transformer-based representation-learning model with unified processing of multimodal input for clinical diagnostics (IRENE)</b></summary>

[Paper](https://www.nature.com/articles/s41551-023-01045-x)
[Code](https://github.com/RL4M/IRENE)
- **Cancer:** Non-Cancer, predicting the adverse clinical outcomes in patients with COVID-19
- **Modalities:** Chest X-rays, Unstructured Text (i.e. chief complaint, history of present and past illness, and a complete laboratory test report), Structured Text (i.e. demographics)
- **Data Source:** In-house dataset from West China Hospital
- **Patients:** 51511 patients with 72283 data samples
    - 44628 patients for training and 3325 patients for testing
- **Pipeline:** 
    - tokenizing unstructured text into tokens
    - mapping structured text into tokens via linear projection
    - tokenizing images into tokens
    - using the proposed bidirectional multimodal attention block followed by some self-attention block for multimodal fusion
    - a classification head for predicting disease
- **Fusion Mode:** Middle-fusion

</details>



<details>
<summary>, <i></i>, <b></b></summary>

[Paper]()
[Code]()
- **Cancer:** 
- **Modalities:** 
- **Data Source:** 
- **Patients:**
- **Pipeline:** 
- **Fusion Mode:** 

</details>

