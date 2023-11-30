# Multimodal Models in Oncology: Enhancing Treatment Response Evaluation and Prognostic Accuracy

## Treatment Response Evaluation

## Prognosis
<details>
<summary>Jun. 2022, <i>Nature Cancer</i>, <b>Multimodal data integration using machine learning improves risk stratification of high-grade serous ovarian cancer</b></summary>

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
<summary>Oct. 2022, <i>IEEE Transactions on Medical Imaging (TMI)</i>, <b>Clinically-Inspired Multi-Agent Transformers for Disease Trajectory Forecasting from Multimodal Data (CLIMATv2)</b></summary>

[Paper](https://ieeexplore.ieee.org/abstract/document/10242080)
[Code](https://github.com/Oulu-IMEDS/CLIMATv2)

- **Cancer:** Non-Cancer, predicting the development of structural knee osteoarthritis changes and forcasting Alzheimer's disease clinical status
- **Modalities:** Imaging Data (MRI, PET, ...) and Non-Imaging Data (Clinical evaluation, neuropsychological tests, genetic testing, ...)
- **Data Source:** [Osteoarthritis Initiative (OAI) cohort](https://nda.nih.gov/oai/); [Alzheimer's Disease Neuroimaging Initiative (ADNI) cohort](https://ida.loni.usc.edu)
- **Patients:** 4796 patients for knee OA structureal prognosis prediction; 2577 patients for AD clinical status prognosis prediction
- **Pipeline:** 
    - a transformer-based radiologist block to extact imaging features (the agent act as a radiologist)
    - a transformer-based context block to extact non-imaging features 
    - concatenating imaging features and non-imaging features, then employing a transformer-based general practitioner block to fuse multimodal features (the agent act as a general practitioner)
    - the prognostic predictions is temporal, and the first time-point's prognostic prediction is required to be consisted with the diagnostic prediction
- **Fusion Mode:** Middle-fusion, concatenating imaging features and non-imaging features and employing a transformer to fuse multimodal features

</details>

<details>
<summary>Apr. 2022, <i>ISBI</i>, <b>CLIMAT: Clinically-Inspired Multi-Agent Transformers for Knee Osteoarthritis Trajectory Forecasting (CLIMAT)</b></summary>

[Paper](https://ieeexplore.ieee.org/abstract/document/9761545)
[Code](https://github.com/MIPT-Oulu/CLIMAT)
- **Cancer:** Non-Cancer, 
- **Modalities:** Imaging Data (X-ray) and Non-Imaging Data (clinical variables like age, sex, BMI, history injurey, surgey, and total Western Ontario and WOMAC)
- **Data Source:** [Osteoarthritis Initiative (OAI) cohort](https://nda.nih.gov/oai/)
- **Patients:** 4796 patients for knee OA structureal prognosis predictions
- **Pipeline:** The pipeline is similar to CLIMATv2, but does not do the first time-point's prognostic and diagnostic predictions consistency measures.
- **Fusion Mode:** Middle-fusion, concatenating imaging features and non-imaging features and employing a transformer to fuse multimodal features

</details>

<details>
<summary>Apr. 2022, <i>Artificial Intelligence in Medicine</i>, <b>A Multi-modal Fusion Framework Based on Multi-task Correlation Learning for Cancer Prognosis Prediction (MultiCoFusion)</b></summary>

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
<summary>Dec. 2020, <i>MICCAI</i>, <b>Brain Tumor Survival Prediction using Radiomics Features</b></summary>

[Paper](https://link.springer.com/chapter/10.1007/978-3-030-66843-3_28)
- **Cancer:** Brain Tumor
- **Modalities:** MRI-T1-weighted, MRI-T2-weighted, T1-contrast enhanced, FLAIR
- **Data Source:** BraTS 2019 
- **Patients:** 259 subjects diagnosed with HGG and 76 subjects diagnosed with LGG along with ground truth annotations by experts. The data comprises of MRI images from 19 different institutions of four MRI modalities
- **Pipeline:** 
    - extracting image slices corresponding to tumor regions from multiple MRI modalities
    - extracting radiomics features (i.e. first-order statistics, shape features, and texture features) from these 2D slices
    - training machince learning classifiers (i.e. KNN, SVM, DT, RF, and DA) to make prognositic predictions
- **Fusion Mode:** Middle-fusion, using machine learning classifiers to integrate multimodal features from multiple MRIs

</details>


<details>
<summary>Jul. 2019, <i>Bioinformatics</i>, <b>Deep learning with multimodal representation for pancancer prognosis prediction</b></summary>

[Paper](https://academic.oup.com/bioinformatics/article/35/14/i446/5529139?login=false)
[Code](https://github.com/gevaertlab/MultimodalPrognosis)
- **Cancer:** Pancancer
- **Modalities:** Clinical Data, Gene (mRNA, microRNA), Pathological Images (WSIs)
- **Data Source:** TCGA
- **Patients:** 11160 patients, split into training and testing datasets in 85/15 ratio
- **Pipeline:** 
    - for the clinical data, using FC layers with sigmoid activations
    - for the genomic data, using deep highway networks
    - for the WSI images, using the SqueezeNet
    - developing an unsupervised encoder (metric learning) to compress different modalities into a single feature vector for each patient (maximizing cosine similarity between positive samples while minimizing cosine similarity between negative samples)
    - handling missing data through a resilient, mltimodal dropout method
    - averaging different modalities' features into a 512 feature vector and using a prediction layer for survival prediction
- **Fusion Mode:** Middle-fusion, align first and then average

</details>


<details>
<summary>Dec. 2017, <i>Cell Systems</i>, <b>Association of Omics Features with Histopathology Patterns in Lung Adenocarcinoma</b></summary>

[Paper](https://www.cell.com/cell-systems/pdf/S2405-4712(17)30484-2.pdf)

- **Cancer:** Lung Adenocarcinoma
- **Modalities:** Pathological Images, Pathological Reports, Gene (RNA sequencing), Proteomics
- **Data Source:** TCGA
- **Patients:** 538 patients
- **Pipeline:** 
    - converting pathological images into overlapping tiles and selected the ROIs to extract quantitative features (i.e. size, shape, intensity distribution, and texture features); identifing pathology grade from pathology reports; collecting gene and protein expression data by RNA sequencing and reverse-phase protein array
    - employing feature selection on the training set
    - building a random forest model for prognostic prediction
- **Fusion Mode:** Middle-fusion, using a random forest model to integrate multimodal features

</details>

## Others

<details>
<summary>Jun. 2023, <i>Nature Biomedical Engineering</i>, <b>A transformer-based representation-learning model with unified processing of multimodal input for clinical diagnostics (IRENE)</b></summary>

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

## Related Reviews
<details>
<summary>Apr. 2023, <i>Progress in Biomedical Engineering</i>, <b>Deep multimodal fusion of image and non-image data in disease diagnosis and prognosis: a review</b></summary>

[Paper](https://iopscience.iop.org/article/10.1088/2516-1091/acc2fe/meta)

**Content:** 
- Data Modalities: Image data (pathology images, radiology images, camera images); Non-image data (structured data, free-text data)
 - Multimodal fusion methods: Operation-based; Subspace-based; Attention-based; Tensor-based; Graph-based

**View points:**
- It is difficult to compare the performance of different methods directly, since different studies were typically done on different datasets with different settings.
- There is no clue that a fusion method always performance the best. The optimal fusion method might be task/data dependent.
- Fusing multi-modal data typically surpassed the uni-modal counterparts in the downstream tasks, but on the other hand, some studies also mentioned that the model that fused more modalities may not always perform better than the ones with fewer modalities (I think the reason is not doing a good modal fusion)
- Deep-learning methods require a large amount of training data, however, data scaricity, especially multimodal data, is a challenge in the healthcare are.
- Unimodal feature extraction is a essential prerequisite for fusion, especially for multimodal heterogeneity.
- Explainability is a challenge in multimodal diagnosis and prognosis.

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

