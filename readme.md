# Multimodal Models in Oncology: Enhancing Treatment Response Evaluation and Prognostic Accuracy

## Treatment Response Evaluation

<details>
<summary>⭐️ [Aug. 2022] <b>Multimodal integration of radiology, pathology and genomics for prediction of response to PD-(L)1 blockade in patients with non-small cell lung cancer</b>, <i>Nature Cancer</i></summary>

[Paper](https://www.nature.com/articles/s43018-022-00416-8)
[Code](https://github.com/msk-mind/luna/)
- **Cancer:** Non-small Cell Lung Cancer, predicting immunotherapy response
- **Modalities:** Radiological Images (CTs), Pathological Images (digitized programmed death ligand-1 immunohistochemistry slides), Gene Data
- **Data Source:** In-House Dataset
- **Patients:** 249 patients at Memorial Sloan Kettering (MSK) Cancer Center with advanced NSCLC who received PD-(L)1-blockade-based therapy with baseline data and known outcomes between 2014 and 2019
- **Pipeline:** [TODO]
- **Fusion Mode:** Middle-fusion, using a multimodal dynamic attention with masking to integrate multimodal features and address missing data

</details>

## Prognosis

<details>
<summary>[Sep 2023] <b>Med-UniC: Unifying Cross-Lingual Medical Vision-Language Pre-Training by Diminishing Bias</b>, <i>NeurIPS</i></summary>

[Paper](https://arxiv.org/abs/2305.19894)
[Code](https://github.com/SUSTechBruce/Med-UniC)
- **Cancer:** Non-Cancer, make experiments across 5 medical image tasks and 10 datasets encompassing over 30 diseases
- **Modalities:** Radiological Images (CXR images), Free-text Data (radiology reports)
- **Data Source:** MIMIC-CXR, PadChest
- **Patients:** Pre-training on approximately 220k image-text pairs for MIMIC-CXR and 160k pairs for PadChest, then applied to four downstream tasks: medical image linear classification, medical image zero-shot classification, medical image semantic segmentation, and medical image object detection 
- **Pipeline:** 
    - for free-text data, using the corss-lingual medical LM to align different languages
    - for CXR images, using contrastive learning to align image features (apply random augmentations to the original images to create augmented views as postive samples while treating the rest of the images in the mini-batch as negative samples)
    - following CLIP, a contrastive learning is used to align vison-language features
    - introducing Cross-lingual Text Alignment Regularization (CTR) to learn language-independent text representations and neutralize the adverse effects of community bias on other modalitieslearn 
- **Fusion Mode:** Middle-fusion, aligning different modalities' features within hidden space

</details>


<details>
<summary>[Mar. 2023] <b>HGIB: Prognosis for Alzheimer’s Disease via Hypergraph Information Bottleneck</b>, <i>arXiv</i></summary>

[Paper](https://arxiv.org/abs/2303.10390)
- **Cancer:** Non-Cancer, predicting Alzheimer's disease prognosis
- **Modalities:** Radiological Images (MRI and PET), Non-imaging Information
- **Data Source:** Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset (adni.loni.usc.edu)
- **Patients:** 248 patients with complete three modalities from ADNI-2
- **Pipeline:** 
    - using different pre-trained backbones to extract features from different modalities
    - for each modality, building a corresponding hypergraph, whose hyperedge represents the relationship between a subset of the patients, then concatenating all hypergraphs to generate the final hypergraph
    - employing hypergraph convolution to aggregating message in the hypergraph
    - applying hypergraph information bottleneck (HGIB) for requiring the node representation to minimize the information from hypergraph-structured data while maximizing the information to make prognostic prediction
- **Fusion Mode:** Middle-fusion, concatenating hypergraphs from different modalities and employing hypergraph convolution and hypergraph information bottleneck (HGIB) to integrate multimodal information

</details>


<details>
<summary>[Oct. 2022] <b>ICSDA: a multi-modal deep learning model to predict breast cancer recurrence and metastasis risk by integrating pathological, clinical and gene expression data</b>, <i>Briefings in Bioinformatics</i></summary>

[Paper](https://academic.oup.com/bib/article-abstract/23/6/bbac448/6761046)
[Code]()
- **Cancer:** Breast Cancer
- **Modalities:** Pathological Images (H&E), Clinical Information (TNM staging, clinical staging, age, axillary lymph node metastasis), Gene Data
- **Data Source:** TCGA
- **Patients:** 196 patients, divided into the training and testing sets with a ratio of 7:3, in which the distributions of the samples were kept between the two datasets by hierarchical sampling
- **Pipeline:** 
    - applying feature selection to select features from clinical information and sequencing data
    - employing ResNet18 to extract deep image features within the tissue area in the H&E images (patching WSI into tiles); then the attention module is used to aggregate patches' features into a final pathological image deep feature
    - concatenating the pathological image deep feature, sequencing data and clinical data and then predicting prognosis via FC layers
- **Fusion Mode:** Middle-fusion, concatenating different modalities' features

</details>



<details>
<summary>[Jun. 2022] <b>Multimodal data integration using machine learning improves risk stratification of high-grade serous ovarian cancer</b>, <i>Nature Cancer</i></summary>

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
<summary>[Oct. 2022] <b>Clinically-Inspired Multi-Agent Transformers for Disease Trajectory Forecasting from Multimodal Data (CLIMATv2)</b>, <i>IEEE Transactions on Medical Imaging (TMI)</i></summary>

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
<summary>[Apr. 2022] <b>CLIMAT: Clinically-Inspired Multi-Agent Transformers for Knee Osteoarthritis Trajectory Forecasting (CLIMAT), <i>ISBI</i></b></summary>

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
<summary>[Apr. 2022] <b>A Multi-modal Fusion Framework Based on Multi-task Correlation Learning for Cancer Prognosis Prediction (MultiCoFusion)</b>, <i>Artificial Intelligence in Medicine</i></summary>

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
<summary>[Dec. 2020] <b>Brain Tumor Survival Prediction using Radiomics Features</b>, <i>MICCAI</i></summary>

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
<summary>[Jan. 2020] <b>PAGE-Net: Interpretable and Integrative Deep Learning for Survival Analysis Using Histopathological Images and Genomic Data</b>, <i>Pacific Symposium on Biocomputing</i></summary>

[Paper](https://pubmed.ncbi.nlm.nih.gov/31797610/)
[Code](https: //github.com/DataX-JieHao/PAGE-Net)
- **Cancer:** Glioblastoma Multiforme
- **Modalities:** Pathological Images (WSIs), Gene Data, Clinical Data
- **Data Source:** TCGA, TCIA
- **Patients:** 447 GBM patients
- **Pipeline:**
    - patching WSIs into patches; the patch-wise pre-trained CNN is used to extract pathological features; then the pathology hidden layer is used to aggregate these features for as input of Cox layer
    - gene features is extracted by a series layers, inlcuding gene layer, pathway layer, H1 and H2 layers
    - clinical features is extracted by the clinical layer
    - these three modalities' features are concatenated and as the input of the Cox layer for prediction
- **Fusion Mode:** Middle-fusion, concatenating multimodal features and using Cox layer for survival analysis

</details>

<details>
<summary>[Jan. 2020] <b>Integrative analysis of cross-modal features for the prognosis prediction of clear cell renal cell carcinoma</b>, <i>Bioimage informatics</i></summary>

[Paper](https://academic.oup.com/bioinformatics/article/36/9/2888/5716325)
[Code](https://github.com/zhang-de-lab/zhang-lab?from¼singlemessage)
- **Cancer:** Clear Cell Renal Cell Carcinoma
- **Modalities:** Radiological Images (CTs), Pathological Images, Gene Data, Clinical Information
- **Data Source:** 
- **Patients:** 209 patients, randomly divided into training (n=139, 66.51%) and testing cohorts (n=70, 33.49%)
- **Pipeline:** 
    - selecting genes by their variation coefficients and employing the weighted gene co-expression network analysis (WGCNA) for gene analysis
    - using two CNNs with same structure to extract deep features from CT and histopathological images
    - using a parameter-free multivariate feature selection method (called block filtering post-pruning search (BFPS) algorithm) for feature selection; then applying a further faeture selection for the combination of the selected CT features, histopathological features and eigengenes for prognositic prediction via the Cox model
- **Fusion Mode:** Middle-fusion, conbinating the selected CT features, histopathological features and eigengenes

</details>



<details>
<summary>[Jul. 2019] <b>Deep learning with multimodal representation for pancancer prognosis prediction</b>, <i>Bioinformatics</i></summary>

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
<summary>[Dec. 2017] <b>Association of Omics Features with Histopathology Patterns in Lung Adenocarcinoma</b>, <i>Cell Systems</i></summary>

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
<summary>[Jun. 2023] <b>A transformer-based representation-learning model with unified processing of multimodal input for clinical diagnostics (IRENE)</b>, <i>Nature Biomedical Engineering</i></summary>

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
<summary>[Sep. 2022] <b>mmFormer: Multimodal Medical Transformer for Incomplete Multimodal Learning of Brain Tumor Segmentation</b>, <i>MICCAI</i></summary>

[Paper](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_11)
[Code](https://github.com/YaoZhang93/mmFormer)
- **Cancer:** Brain Tumor
- **Modalities:** MRIs (FLAIR, T1c, T1, T2)
- **Data Source:** BraTS 2018
- **Patients:** 285 multi-contrast MRI scans
- **Pipeline:** 
    - using modality-specific encoders to extract modelity-specific features within each modality
    - employing an inter-modal transformer to build and align the long-range correlations across modalities
    - a decoder performs a progressive up-sampling and fusion with the modality-invariant features to generate robust segmentation
- **Fusion Mode:** Middle-fusion, using an inter-modal transformer to integrate multimodal features

</details>


<details>
<summary>[Feb. 2021] <b>Multimodal deep learning models for early detection of Alzheimer’s disease stage</b>, <i>Scientific Reports</i></summary>

[Paper](https://www.nature.com/articles/s41598-020-74399-w)
[Code]()
- **Cancer:** Non-Cancer, early detection of Alzheimer's disease stage
- **Modalities:** Radiological Images (MRI), Gene Data (single nucleotide polymorphisms (SNPs)), Clinical Data
- **Data Source:** ADNI dataset
- **Patients:** ADNI dataset contains SNP (808 patients), MRI imaging (503 patients), and clinical and neurological test data (2004 patients)
- **Pipeline:** 
    - using stacked denoising auto-encoders to extract faetures from clinical and genetic data
    - using 3D0CNNs for imaging data
    - developing a novel data interpretation method to identify top-performing features learned by the deep-models with clustering and perturbation analysis
- **Fusion Mode:** Middle-fusion, concatenating multimodal features and then using a classification layer for prediction

</details>


## Related Reviews
<details>
<summary>[Apr. 2023] <b>Deep multimodal fusion of image and non-image data in disease diagnosis and prognosis: a review</b>, <i>Progress in Biomedical Engineering</i></summary>

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
<summary>[Sep. 2022] <b>Multimodal biomedical AI</b>, <i>Nature Medicine</i></summary>

[Paper](https://www.nature.com/articles/s41591-022-01981-2)

**Content & View points:** 
- Opportunities for leveraging multimodal data (applications)
    - Personalized 'omics' for precision health
    - Digital clinical trials
    - Remote monitoring: the 'hospital-at-home'
    - Pandemic surveillance and outbreak detection
    - Digital twins
    - Virtual health assistant
- Multimodal data collection

| Study                 | Country | Year started | Data modalities                                                                                                                                                         | Access         | Sample size   |
|-----------------------|---------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|---------------|
| [UK Biobank](https://www.ukbiobank.ac.uk)            | UK      | 2006         | Questionnaires, EHR/clinical, Laboratory, Genome-wide genotyping, WES, WGS, Imaging, Metabolites                                                                       | Open access    | ~500,000      |
| [China Kadoorie Biobank](https://www.ckbiobank.org) | China   | 2004         | Questionnaires, Physical measurements, Biosamples, Genome-wide genotyping                                                                                                | Restricted access | ~500,000      |
| [Biobank Japan](https://biobankjp.org/en/index.html#01)         | Japan   | 2003         | Questionnaires, Clinical, Laboratory, Genome-wide genotyping                                                                                                             | Restricted access | ~200,000      |
| [Million Veteran Program](https://www.mvp.va.gov/pwa/) | USA   | 2011         | EHR/clinical, Laboratory, Genome wide                                                                                                                                   | Restricted access | 1 million     |
| [TOPMed](https://topmed.nhlbi.nih.gov/topmed-data-access-scientific-community)                | USA     | 2014         | Clinical, WGS                                                                                                                                                           | Open access    | ~180,000      |
| [All of Us Research Program](https://allofus.nih.gov) | USA | 2017         | Questionnaires, SDH, EHR/clinical, Laboratory, Genome wide, Wearables                                                                                                   | Open access    | 1 million (target) |
| [Project Baseline Health Study](https://ctsi.duke.edu/project-baseline-health-study) | USA | 2015       | Questionnaires, EHR/clinical, Laboratory, Wearables                                                                                                                     | Restricted access | 10,000 (target) |
| [American Gut Project](https://db.cngb.org/search/project/PRJEB11419/)  | USA     | 2012         | Clinical, Diet, Microbiome                                                                                                                                              | Open access    | ~25,000       |
| [MIMIC](https://lcp.mit.edu/mimic)                 | USA     | 2008-2019    | Clinical/EHR, Images                                                                                                                                                    | Open access    | ~380,000      |
| [MIPACT](https://precisionhealth.umich.edu/our-research/mipact/)                | USA     | 2018-2019    | Wearables, clinical/EHR, physiological, laboratory                                                                                                                      | Restricted access | ~6,000        |
| [North American Prodrome Longitudinal Study](https://napls.ucsf.edu) | USA  | 2008 | Clinical, Genetic                                                                                                                                                       | Restricted access | ~1,000        |

- Technical challenges
    - How to leverage multiple different types of data and learn to relate these multiple modalities or combine them for improving prediction performance?
    - Another desirable feature for multimodal learning frameworks is the ability to learn from different modalities without the need for different model architectures.
    - Another important modeling challenge relates to the exceedingly high number of dimensions contained in multimodal health data, collectively termed ‘the curse of dimensionality’.
    - Multimodal fusion is a general concept that can be tackled using any architectural choice.
    - Many other important challenges relating to multimodal model architectures remain (for example, how to extract features from three-dimensional imaging or whole-slide images)
- Data challenges
    - Medical datasets are heterogeneous, which can be described along several axes, including the sample size, depth of phenotyping, the length and intervals of follow-up, the degree of interaction between participants, the heterogeneity and diversity of the participants, the level of standardization and harmonization of the data and the amount of linkage between data sources.
    - Achieving diversity across race/ethnicity, ancestry, income level, education level, healthcare access, age, disability status, geographic locations, gender and sexual orientation has proven difficult in practice.
    - Another frequent problem with biomedical data is the usually high proportion of missing data.
    - The risk of incurring several biases is important when conducting studies that collect health data, and multiple approaches are necessary to monitor and mitigate these biases.
- Privacy challenges
    - The successful development of multimodal AI in health requires breadth and depth of data, which encompasses higher privacy challenges than single-modality AI models.

</details>








