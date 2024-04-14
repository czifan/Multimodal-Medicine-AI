<!-- # Multimodal Models in Oncology: Enhancing Treatment Response Evaluation and Prognostic Accuracy -->
# Multimodal Medicine AI

## Multimodal Medical Datasets
🦘[Multimodal Medical Datasets](https://github.com/czifan/Multimodal-Medicine-AI/blob/main/multimodal_dataset.md)

## Treatment Response Evaluation

| Year | Paper | Code | Cancer | Modalities | Data Source | Patients | Fusion Mode |
|-------|-------|------|--------|------------|-------------|----------|-------------|
| 2023 | [🔗](https://www.biorxiv.org/content/10.1101/2023.11.24.568360v1.abstract)| | ccRCC | Gene | TCGA, In-House | ~1000 | Middle |
| 2023 | [🔗](https://www.biorxiv.org/content/10.1101/2023.07.04.547697v1.abstract) | [🔗](https://github.com/rootchang/ICBpredictor) | 18 solid tumor types | Path, Gene, Clin | In-House | 2881 | Middle |
| 2023 | [🔗](https://www.sciencedirect.com/science/article/pii/S0167814023003316?casa_token=MZeMEY7Dz48AAAAA:9iepZVnJHZdhSU0Hmoq-UyajUchgBk1i1ZpoSZTj0NvvdbUaQhJg5ltcoth-iAC0TaVq9abwWA) | [🔗](https://github.com/vancywx/Immunotherapy-response-prediction-using-multi-modal-semi-superviseddeep-learning/tree/main) | GC | Rad, Clin | In-House | 249 | Middle |
| 2023 | [🔗](https://translational-medicine.biomedcentral.com/articles/10.1186/s12967-023-04004-x) | | NSCLC | Rad, Clin | In-House | 264 | Late |
| 2022 | [🔗](https://www.nature.com/articles/s43018-022-00416-8) | [🔗](https://github.com/msk-mind/luna/) | NSCLC | Rad, Path, Gene | In-House | 249 | Middle |
| 2022 | [🔗](https://www.sciencedirect.com/science/article/pii/S1361841522001128) | | | Rad | UKBB, EchoNet-Dynamic, GSTFT, GSTFT CRT | 62 for response predictions and 10,730 for training segmentation models | Middle |
| 2021 | [🔗](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7868825/) | | NSCLC | Rad, Lab, Clin | In-House | 200 | Middle |
| 2020 | [🔗](https://www.thelancet.com/journals/eclinm/article/PIIS2589-5370(20)30123-1/fulltext) | | HCC | Rad | In-House | 737 | Middle |


<details>
<summary>[Nov. 2023] <b>Multi-omics features-based machine learning method improve immunotherapy response in clear cell renal cell carcinoma</b>, <i>bioRxiv</i></summary>

[Paper](https://www.biorxiv.org/content/10.1101/2023.11.24.568360v1.abstract)
- **Cancer:** Clear Cell Renal Cell Carcinomas
- **Modalities:** Gene Data (bulk RNA, scRNA, DNA)
- **Data Source:** TCGA, In-House dataset
- **Patients:** >1900 patients with immune-mediated kidney discorders; >400 patients with ccRCC treated by ICBs; ~1000 patients as the immune cohort for ccRCC
- **Pipeline:** 
    - extracting six distinct types of features (TIs) from multimodal gene data
    - using XGBoost to predict response based on these features
- **Fusion Mode:** Middle-fusion, using XGBoost to integrate multimodal features
</details>


<details>
<summary>[Jul. 2023] <b>Robust prediction of patient outcomes with immune checkpoint blockade therapy for cancer using common clinical, pathologic, and genomic feature</b>, <i>bioRxiv</i></summary>

[Paper](https://www.biorxiv.org/content/10.1101/2023.07.04.547697v1.abstract)
[Code](https://github.com/rootchang/ICBpredictor)
- **Cancer:** 18 solid tumor types
- **Modalities:** Pathologic, Gene Data, Clinical Data
- **Data Source:** In-House dataset
- **Patients:** 2881 immune checkpoint blockade (ICB)-treated patients across 18 solid tumor types
- **Pipeline:** Using machine learning (i.e., decision tree, random forest) to take the clinical, pathologic, genomic features as inputs and make predictions
- **Fusion Mode:** Middle-fusion, using ML algorithms to integrate multimodal features
</details>

<details>
<summary>⭐️ [Jul. 2023] <b>Cancer immunotherapy response prediction from multi-modal clinical and image data using semi-supervised deep learning</b>, <i>Radiotherapy and Oncology</i></summary>

[Paper](https://www.sciencedirect.com/science/article/pii/S0167814023003316?casa_token=MZeMEY7Dz48AAAAA:9iepZVnJHZdhSU0Hmoq-UyajUchgBk1i1ZpoSZTj0NvvdbUaQhJg5ltcoth-iAC0TaVq9abwWA)
[Code](https://github.com/vancywx/Immunotherapy-response-prediction-using-multi-modal-semi-superviseddeep-learning/tree/main)
- **Cancer:** Gastric Cancer
- **Modalities:** Radiological Images (CTs), Clinical Data
- **Data Source:** In-House
- **Patients:** 249 advanced gastric cancer patients treated with immunotherapy, and an additional dataset of 2029 patients who did not receive immunotherapy in a semi-supervised framework to learn intrinsic imaing phenotypes of the disease
    - 168 advanced GC patients treated with immunotherapy for training
    - two independent cohorts of 81 patients treated with immunotherapy for evaluating model performance
- **Pipeline:** 
    - an MLP for extracting clinical features from clinical data
    - an MLP for mapping radiomics features extracted from CTs
    - a CNN for extracting deep image features from CTs
    - concatenating these features into a multimodal features and predicting response/non-response via an MLP
    - this work innovatively employs a semi-supervised framework to leverage unlabeled examples (patients not treated with immunotherapy). Specially, for labeled example, the consistent loss is employed to consist the teacher model's predictions (predicted by multimodal features) and student model's predictions (predicted by only deep image features); for unlabeled example, the consistent loss is used to consist the teacher model's predictions (applied weak augmentation for CTs) and student model's predictions (applied strong augmentation for CTs). The teacher model is an ema model from student models.
- **Fusion Mode:** Middle-fusion, concatenating multimodal features for predictions via an MLP
</details>


<details>
<summary>[Mar. 2023] <b>Integration of longitudinal deep-radiomics and clinical data improves the prediction of durable benefits to anti-PD-1/PD-L1 immunotherapy in advanced NSCLC patients</b>, <i>Journal of Translational Medicine</i></summary>

[Paper](https://translational-medicine.biomedcentral.com/articles/10.1186/s12967-023-04004-x)
- **Cancer:** Advanced Non-small Cell Lung Cancer (NSCLC)
- **Modalities:** Radiological Images (CTs with follow-ups), Clinical Data (demographic, epidemiologic data, hemogram with follow-ups)
- **Data Source:** In-House dataset
- **Patients:** 264 patients with pathologically confirmed stage IV NSCLC treated with immunotherapy from two institutions, randomly divided into a training (n=221) and an independent test set (n=43)
- **Pipeline:** 
    - using Radiomics and NoduleX to extract time-series CT features and then concatenating them to as the input of Random Forest to predict response
    - clinical data is first encoded by one-hot encoding and then concatenated to as the input of another Random Forest to predict response
    - averaging these two results to get ensemble prediction
- **Fusion Mode:** Late-fusion, averaging multimodal predictions into an ensemble prediction
</details>

<details>
<summary>⭐️ [Aug. 2022] <b>Multimodal integration of radiology, pathology and genomics for prediction of response to PD-(L)1 blockade in patients with non-small cell lung cancer</b>, <i>Nature Cancer</i></summary>

[Paper](https://www.nature.com/articles/s43018-022-00416-8)
[Code](https://github.com/msk-mind/luna/)
- **Cancer:** Non-small Cell Lung Cancer, predicting immunotherapy response
- **Modalities:** Radiological Images (CTs), Pathological Images (digitized programmed death ligand-1 immunohistochemistry slides), Gene Data
- **Data Source:** In-House Dataset
- **Patients:** 249 patients at Memorial Sloan Kettering (MSK) Cancer Center with advanced NSCLC who received PD-(L)1-blockade-based therapy with baseline data and known outcomes between 2014 and 2019
- **Pipeline:**
    - extracting radiomics features using expert segmented thoracic CT scans (Radiology Radiomics per site)
    - extracting image-based IHC texture from original digitized PD-L1 IHC slide via the tumor segmentation mask and several visual transformations (Pathology GLCM and TPS)
    - obtaining genomic alterations and TMB
    - DyAM was used for multimodal integration. CT segmentation-derived features were separated by lesion type (lung PC, PL and LN) with separate attention weights applied. Attention weights are also used for genomics and PD-L1 IHC-derived features to result in a final prediction of response.
- **Fusion Mode:** Middle-fusion, using a multimodal dynamic attention with masking to integrate multimodal features and address missing data

</details>

<details>
<summary>[Apr. 2022] <b>A multimodal deep learning model for cardiac resynchronisation therapy response prediction</b>, <i>Medical Image Analysis</i></summary>

[Paper](https://www.sciencedirect.com/science/article/pii/S1361841522001128)
- **Cancer:** Non-Cancer, predicting cardiac resynchronisation therapy response
- **Modalities:** 2D echocardiography and cardiac magnetic resonace (CMR) data
- **Data Source:** 
    - UK Biobank (UKBB) for pre-training the CMR segmentation model
    - EchoNet-Dynamic dataset for pre-training the echocardiography segmentation model
    - Guys and St Thomas NHS Foundation Trust (GSTFT) for training and validating the CMR and echocardiography segmentation models
    - GSTFT CRT echocardiography database for testing the proposed model in the intended clinical application of using only echocardiography data at test time
- **Patients:** 
    - [UK Biobank (UKBB)](https://www.ukbiobank.ac.uk/): 700 healthy subjects
    - [EchoNet-Dynamic dataset](https://echonet.github.io/echoNet/): 10,030 patients
    - Guys and St Thomas NHS Foundation Trust (GSTFT): 50 HF patients and 50 CRT patients (32/50 patients who were classified as responders to CRT)
    - GSTFT CRT echocardiography database: 12 CRT patients (7/12 patients who were classified as responders to CRT)
    - a total of 62 patients for response predictions
- **Pipeline:**
    - the nnU-Net architecture is used to extract segmentations of the heart over the full cardiac cycle from the two modalities
    - training the multimodal deep learning (MMDL) by maximizing the correlation between two modalities' latent respresents
    - combining the latent spaces of the nnU-Net models from two modalities through average
    - using a SVM classifier for predicting CRT response
- **Fusion Mode:** Middle-fusion, maximizing the correlation between multimodal features and averaging them

</details>

<details>
<summary>⭐️ [Feb. 2021] <b>A multi-omics-based serial deep learning approach to predict clinical outcomes of single-agent anti-PD-1/PD-L1 immunotherapy in advanced stage non-small-cell lung cancer</b>, <i>American Journal of Translational Research</i></summary>

[Paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7868825/)
- **Cancer:** Non-small-cell Lung Cancer (NSCLC)
- **Modalities:** Radiological Images (serial radiomics), Laboratory Data, Baseline Clinical Data
- **Data Source:** In-House Dataset
- **Patients:** 200 advanced stage NSCLC patients with 1633 CT scans and 3414 blood samples who received single anti-PD-1/PD-L1 agent between April 2016 and December 2019
- **Pipeline:** 
    - using the proposed Simple Temporal Attention (SimTA) moduels to process asynchronous clinical time series (i.e. the radiomics and blood tests) separately
    - the encoded features of these time series and static clinical information are then fused by a MLP to get the final output for the assessment prediction of responders/non-responders
- **Fusion Mode:** Middle-fusion, concatenating radiomics and blood test features and then using MLP for predictions
</details>


<details>
<summary>[Jun. 2020] <b>Prediction of prognostic risk factors in hepatocellular carcinoma with transarterial chemoembolization using multi-modal multi-task deep learning</b>, <i>eClinicalMedicine</i></summary>

[Paper](https://www.thelancet.com/journals/eclinm/article/PIIS2589-5370(20)30123-1/fulltext)
- **Cancer:** Hepatocellular Carcinoma
- **Modalities:** Radiological Images (CTs)
- **Data Source:** In-house dataset
- **Patients:** a total 737 patients, 478 patients (64.9%) underwent surgical resection; 16 patients (2.2%) underwent liver transplantation and 243 patients (32.9%) underwent nonsurgical TACE treatment.
- **Pipeline:** 
    - a Random forest feature selection and a SVM predictor used to develop MVI-score and Edmondson' score in 494 HCCs with surgical resection
    - multi-task DL networks to build a prognostic score for HCC survival after TACE
        - first, a DAE is used to reduce and transform 2420 radiomics features from 243 HCCs with TACE into 70 new features from the bottleneck hidden layer of the networks
        - then, six time-varying DL algorithms were used to train the obtained DAE-transformed features and the one perform best was used to build a prognostic score to compute the survival probabilities on the time grid
    - Finally, MVI-score, Edmondson's score, DL-based survival score and evidenced-based clinicoradiologic score were integrated into a Cox-PH model to obtain a precise prediction
- **Fusion Mode:** Middle-fusion, using Cox-PH model to integrate multimodal scores into a prognostic prediction
</details>

## Prognosis Evaluation

| Year  | Paper | Code | Cancer | Modalities | Data Source | Patients | Fusion Mode |
|-------|-------|------|--------|------------|-------------|----------|-------------|
| 2023 | [🔗](https://arxiv.org/abs/2305.19894) | [🔗](https://github.com/SUSTechBruce/Med-UniC) | | Rad, Text | MIMIC-CXR, PadChest | ~380k pairs | Middle |
| 2023 | [🔗](https://arxiv.org/abs/2303.10390) | | | Rad, Non-imaging | ADNI | 248 | Middle |
| 2022 | [🔗](https://academic.oup.com/bib/article-abstract/23/6/bbac448/6761046) | | BC | Path, Clin, Gene | TCGA | 196 | Middle |
| 2022 | [🔗](https://www.nature.com/articles/s43018-022-00388-9) | [🔗](https://github.com/kmboehm/onco-fusion) | OC | Rad, Path, Clin | MSKCC, TCGA-OV | 444 | Late |
| 2022 | [🔗](https://ieeexplore.ieee.org/abstract/document/10242080) | [🔗](https://github.com/Oulu-IMEDS/CLIMATv2) | | Imaging, Non-Imaging | OAI, ADNI | 4796 (knee OA), 2577 (AD) | Middle |
| 2022 | [🔗](https://ieeexplore.ieee.org/abstract/document/9761545) | [🔗](https://github.com/MIPT-Oulu/CLIMAT) | | X-ray, Non-Imaging | OAI | 4796 | Middle |
| 2022 | [🔗](https://www.sciencedirect.com/science/article/pii/S0933365722000252) | | Brain | Path, Gene | TCGA-LGG, TCGA-GBM | 470 | Middle |
| 2020 | [🔗](https://link.springer.com/chapter/10.1007/978-3-030-66843-3_28) | | Brain | MRIs | BraTS 2019 | 335 | Middle |
| 2020 | [🔗](https://ieeexplore.ieee.org/abstract/document/9186053) | [🔗](https://github.com/mahmoodlab/PathomicFusion) | Glioma, ccRCC | Path, Gene | TCGA-GBM, TCGA-LGG | 769 | Middle |
| 2020 | [🔗](https://pubmed.ncbi.nlm.nih.gov/31797610/) | [🔗](https://github.com/DataX-JieHao/PAGE-Net) | GBM | Path, Gene, Clin | TCGA, TCIA | 447 | Middle |
| 2020 | [🔗](https://academic.oup.com/bioinformatics/article/36/9/2888/5716325) | [🔗](https://github.com/zhang-de-lab/zhang-lab) | ccRCC | Rad, Path, Gene, Clin | TCGA | 209 | Middle |
| 2019 | [🔗](https://academic.oup.com/bioinformatics/article/35/14/i446/5529139?login=false) | [🔗](https://github.com/gevaertlab/MultimodalPrognosis) | Pancancer | Clin, Gene, Path | TCGA | 11160 | Middle |
| 2017 | [🔗](https://www.cell.com/cell-systems/pdf/S2405-4712(17)30484-2.pdf) | | LUNA | Path, Path Reports, Gene, Proteomics | TCGA | 538 | Middle |


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
- **Fusion Mode:** Late-fusion, using a multivariate Cox model to integrate unimodal submodels’ predictions

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
<summary>⭐️ [Sep. 2020] <b>Pathomic Fusion: An Integrated Framework for Fusing Histopathology and Genomic Features for Cancer Diagnosis and Prognosis</b>, <i>TMI</i></summary>

[Paper](https://ieeexplore.ieee.org/abstract/document/9186053)
[Code](https://github.com/mahmoodlab/PathomicFusion)
- **Cancer:** Glioma, Clear Cell Renal Cell Carcinoma
- **Modalities:** Pathological Images, Gene Data (mutations, CNV, RNA-Seq)
- **Data Source:** TCGA-GBM, TCGA-LGG
- **Patients:** 769 patients
- **Pipeline:** 
    - using CNNs, parameter efficient GCNs or a combination of the two to extract histology features
    - using a feed-forword network to extract genomic features
    - first training unimodal networks for the respective image and genomic features individually for the corresponding supervised learning task, then used as feature exxtractors for multimodal fusion
    - multimodal fusion is performed by applying an gating-based attention mechanism to first control the expressiveness of each modality, followed by the Kronecker product to model pairwise feature interactions across modalities
    - finally, using cox model for survival analysis and the FC layers for classification
- **Fusion Mode:** Middle-fusion, employing gating-based attention mechanism followed by a Kronecher product to intergate multimodal features
</details>



<details>
<summary>[Jan. 2020] <b>PAGE-Net: Interpretable and Integrative Deep Learning for Survival Analysis Using Histopathological Images and Genomic Data</b>, <i>Pacific Symposium on Biocomputing</i></summary>

[Paper](https://pubmed.ncbi.nlm.nih.gov/31797610/)
[Code](https://github.com/DataX-JieHao/PAGE-Net)
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
- **Data Source:** TCGA
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

| Year | Paper | Code | Cancer | Modalities | Data Source | Patients | Fusion Mode |
|------|-------|------|--------|------------|-------------|----------|-------------|
| 2024 | [🔗](https://arxiv.org/pdf/2402.14252.pdf) | | | Rad | In-House | | |
| 2023 | [🔗](https://arxiv.org/abs/2304.02836) | [🔗](https://github.com/MASILab/lmsignatures) | SPN | Rad, Clin | NLST, EHR-Pulmonary, Image-EHR, In-House | 2668 (public), 1449 (in-house) | Middle |
| 2023 | [🔗](https://www.nature.com/articles/s41551-023-01045-x) | [🔗](https://github.com/RL4M/IRENE) | | X-rays, Text | In-House | 51511 | Middle |
| 2023 | [🔗](https://www.sciencedirect.com/science/article/pii/S0895611123001179) | | LUNA | Rad, Clin | In-House | 199 | Middle |
| 2022 | [🔗](https://arxiv.org/abs/2212.09162) | [🔗](https://github.com/FirasGit/lsmt) | | Rad, Clin | MIMIC | >40,000 | Middle |
| 2022 | [🔗](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_11) | [🔗](https://github.com/YaoZhang93/mmFormer) | Brain | MRIs | BraTS 2018 | 285 | Middle |
| 2021 | [🔗](https://ieeexplore.ieee.org/abstract/document/9366692) |  | | MRIs, PETs | ADNI | 820 | Middle |
| 2021 | [🔗](https://www.nature.com/articles/s41598-020-74399-w) |  | | MRI, Gene, Clin | ADNI | 2004 | Middle |
| 2019 | [🔗](https://pubmed.ncbi.nlm.nih.gov/31586211/) | [🔗](https://cnoc-bwh.shinyapps.io/gbmsurvivalpredictor/) | Glioblastoma | Clinical info. | SEER | 20821 | Early |

<details>
<summary>[Feb. 2024] <b>Multimodal Healthcare AI: Identifying and Designing Clinically Relevant Vision-Language Applications for Radiology</b>, <i>arXiv</i></summary>

[Paper](https://arxiv.org/pdf/2402.14252.pdf)
- **Cancer:** Non-Cancer
- **Modalities:** Radiological Images
- **Data Source:** In-House
- **Contribution:** The first inverstigation into the potential utility and design requirements for leveraging vision-language model (VLM) capabilities with 13 radiologists and clinicians in the context of radiology of four tasks: Draft Report Generation, Augmented Report Review, Visual Search and Querying, and Patient Imaging History Highlights.
</details>

<details>
<summary>[Jun. 2023] <b>Longitudinal Multimodal Transformer Integrating Imaging and Latent Clinical Signatures From Routine EHRs for Pulmonary Nodule Classification</b>, <i>arXiv</i></summary>

[Paper](https://arxiv.org/abs/2304.02836)
[Code](https://github.com/MASILab/lmsignatures)
- **Cancer:** Solitary Pulmonary Nodule (SPN)
- **Modalities:** Radiological Images (chest CTs), Clinical Data (EHR)
- **Data Source:** [NLST](https://cdas.cancer.gov/nlst/), EHR-Pulmonary (the unlabeled dataset used to learn clinical signatures in an unsupervised manner), Image-EHR (a labeled dataset with paired imaging and EHRs), In-House dataset
- **Patients:** Our classifier is pretrained on 2,668 scans from a public dataset and 1,149 subjects with longitudinal chest CTs, billing codes, medications, and laboratory tests from EHRs of our home institution.
- **Pipeline:** 
    - learning independent latent signatures in an unsupervised manner on a large non-imaging cohort (non-imaging features)
    - extracting longitudinal deep image features from CTs via a CNN (imaging features)
    - token embedding is derived from signatures (non-imaging features) and imaging (imaging features); a fixed positional embedding indicating the token's position in the sequence; a learnable segment embedding indicating imaging or non-imaging modality
    - a self-attention is used to integrate multimodal and longitudinal features
- **Fusion Mode:** Middle-fusion, using self-attention to integrate multimodal features
</details>

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
<summary>[Feb. 2023] <b>Development and evaluation of an integrated model based on a deep segmentation network and demography-added radiomics algorithm for segmentation and diagnosis of early lung adenocarcinoma</b>, <i>Computerized Medical Imaging and Graphics</i></summary>

[Paper](https://www.sciencedirect.com/science/article/pii/S0895611123001179)

- **Cancer:** Lung Adenocarcinoma
- **Modalities:** Radiological Images (CT), Clinical Data
- **Data Source:** In-House
- **Patients:** A total of 199 GGN cases, consisting of 168 GGN cases for developing the model and the rest of 31 independent cases for validation
- **Pipeline:** 
    - first, a deep segmentation model is utilized to locate GGNs in CTs and to help categorizing the lesions with a classification model to be subsequently applied
    - then, extracting 1690 quantitative image features via Pyradiomics from lesions, and 28 features from the settings of CTs (i.e., device and modality settings), patients' general characteristics (i.e., age, sex, smoking status), and references were added
    - reducing and selecting the features
    - using a classifier to make prediction
- **Fusion Mode:** Middle-fusion, concatenating CT radiomics features and clinical data and the settings of CTs within feature space
</details>


<details>
<summary>[Dec. 2022] <b>Medical Diagnosis with Large Scale Multimodal Transformers: Leveraging Diverse Data for More Accurate Diagnosis</b>, <i>arXiv</i></summary>

[Paper](https://arxiv.org/abs/2212.09162)
[Code](https://github.com/FirasGit/lsmt)
- **Cancer:** Non-Cancer, focus on intensive care and ophthalmology walk-ins
- **Modalities:** Radiological Images (chest radiographs, fundoscopy images), Clinical Data
- **Data Source:** MIMIC dataset
- **Patients:** MIMIC database comprises retrospectively collected image and non-image data of over 40,000 patients admitted to an intensive care unit or the emergency department at the Beth Israel Deaconess Medical Center between 2008 and 2019. 
    - The authors follow [the previous work](http://arxiv.org/abs/2207.07027) and extract imaging and non-imaging information from the [MIMIC-IV](https://physionet.org/content/mimiciv/1.0/) and [MIMIC-CXR-JPG](https://arxiv.org/abs/1901.07042) database resulting in a subset of 45,676 samples from n=36,542 patients
    - The internal dataset of chest radiographs consisting of 193,556 samples (n=45,016 patients) is thus split into a training set of 122,294 samples (n=28,809 patients), validation set of 31,243 samples (n=7,203 patients) and a test set of 40,028 samples (n=9,004 patients).
    - The the fundoscopy dataset comprised of 3,860 samples (n=1,930 patients) is split into training set of 2,586 samples (n=1,293 patients), a validation set of 502 samples (n=251 patients) and a test set of 772 samples (n=386 patients).
- **Pipeline:** 
    - using a transformer encoder (similar to ViT) to tokenize and encode imaging data into visual tokens (imaging features)
    - using learnable tokens to as query, meanwhile clinical parameters as the key and value, and employing the cross-attention to extract clinical information from clinical parameters into learnable tokens (non-imaging features)
    - the output learnbale tokens and the visual tokens are passed through the transformer encoder, and then the class token is used to make prediction via a MLP
- **Fusion Mode:** Middle-fusion, using a transformer encoder to integrate imaging and non-imaging features
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
<summary>[Mar. 2021] <b>Relation-Induced Multi-Modal Shared Representation Learning for Alzheimer’s Disease Diagnosis</b>, <i>TMI</i></summary>

[Paper](https://ieeexplore.ieee.org/abstract/document/9366692)
- **Cancer:** Non-Cancer, predicting Alzheimer's disease diagnosis
- **Modalities:** Radiological Images (MRIs, PETs)
- **Data Source:** [ADNI](http://www.loni.usc.edu)
- **Patients:** A total of 820 patients, consisting of 93 AD, 99 NC, 121 sMCI, and 79 pMCI from ADNI-1 and 136 AD, 107 NC, 103 sMCI, and 82 pMCI from ADNI-2.
- **Pipeline:** 
    - learning a bi-directional mapping (including projection matrix P and reconstruction matrix Q) to obtain the shared representation matrix U between original space and shared space
    - within this shared space, utilizing several relational regularizers (including feature-feature, feature-label, and sample-sample regularizers) as auxiliary regularizers to encourage learning underlying associations inherent in multi-modal data and alleviate overfitting
    - predict the shared representations into the target space for AD diagnosis
- **Fusion Mode:** Middle-fusion, learning a shared-representation across different modalities 
</details>



<details>
<summary>[Feb. 2021] <b>Multimodal deep learning models for early detection of Alzheimer’s disease stage</b>, <i>Scientific Reports</i></summary>

[Paper](https://www.nature.com/articles/s41598-020-74399-w)
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

<details>
<summary>[Oct. 2019] <b>An Online Calculator for the Prediction of Survival in Glioblastoma Patients Using Classical Statistics and Machine Learning</b>, <i>RESEARCH—HUMAN—CLINICAL STUDIES</i></summary>

[Paper](https://pubmed.ncbi.nlm.nih.gov/31586211/)
[Code](https://cnoc-bwh.shinyapps.io/gbmsurvivalpredictor/)
- **Cancer:** Glioblastoma
- **Modalities:** Clinical information, including continuous variables (age, tumor diameter, ...), categorical variables (sex, race, ...)
- **Data Source:** [Surveillance Epidemiology and end results (SEER) dataset (2005-2015)](https://pubmed.ncbi.nlm.nih.gov/24464362/)
- **Patients:** in total 20821 patients split into a training and hold-out test set in an 80/20 raio
- **Pipeline:** 
    - for censored survival data, using Cox proportional hazards regression (CPHR) and accelerated failure time (AFT) algorithms
    - for predictive analysis, using 15 machine learning and statistical algorithms
- **Fusion Mode:** Early-fusion, taking continuous variables and categorical variables as inputs, actually, it acts as the multi-variables analysis

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
<summary>⭐️ [Oct. 2022] <b>Artificial intelligence for multimodal data integration in oncology</b>, <i>Cancer Cell</i></summary>

[Paper](https://www.cell.com/cancer-cell/pdf/S1535-6108(22)00441-X.pdf)

**Content:** 
- AI methods in oncology
    - Supervised methods
        - Hand-crafted methods
            - 👍：simpler architecture, lower computation cost, may require less training data, and better interpretability
            - 👎：time consuming, translate human bias to the models
        - Representation learning methods
            - 👍：their ability to extract rich feature representations from raw data, resulting in lower preprocessing cost, higher flexibility, and often superior performance over hand-crafted methods
            - 👎：reliance on pixel-level annotations, lack of interpretability
    - Weakly supervised methods: this method can reduce the cost of data preprocessing and mitigate the bias and interrater variability; additionally, they are free to learn from the entire scan, that can indentify predictive features even beyond the regions typically evaluated by clinicians.
        - Graph convolutional networks
            - 👍：can incorporate larger context and spatial tissue structure
            - 👎：higher training costs and memory requirements (since the nodes cannot be processed independently)
        - Multiple-instance learning
            - 👍：no fine annotation is required
            - 👎：overlook patches' correlation
        - Vision transformers
            - 👍：be fully context aware, consider patches' correlation and context, consider spatial structure or relative distances between patches via positional encoding
            - 👎: tend to be more data hungry
    - Unsupervised methods
        - Self-supervised methods
            - 👍：can learn general-purpose features, which can be beneficial for other practical tasks (transfer learning)
            - 👎: (Not mentioned in the paper)
        - Unsupervised feature analysis
            - 👍：can explore structure, similarity and common features across data points
            - 👎: (Not mentioned in the paper)
- Multimodal data fusion
    - Early fusion
        - 👍：only one model is trainied, simplifing the design process
        - 👎: requires a certain level of alignment or synchronization between the modalities
    - Late fusion (decision-level fusion)
        - 👍：allows one to use a different model achitecture for each modality, making it suitable for systems with large data heterogeneity or modalities from different time points; be able to cope  with missing or incomplete data; suitable for weak interdependencies
        - 👎: unsuitable for strong interdependencies
    - Intermediate fusion
        - 👍：flexible—single-level fusion, gradual fusion, guided fusion
    - There is no conclusive evidence that one fusion type is ultimately better than the others, as each type is heavily data and task specific.
- Multimodal interpretability
    - Histopathology: map model architecture attention or probability scores to obtain slide-level attention heatmaps
    - Radiology: is similar to those used in histoloty
    - Molecular data: use the integrated gradient method to analyze, which computes attribution values indicating how changes in specific inputs affect the model outputs
    - Multimodal models: all previously mentioned methods can be used in multimodal models to explore interpretability within each modality. Moreover, shifts in feature importance under unimodal and multimodal settings can be investigated to analyze the impact of the multimodal context.
    - While CAM- or attention-based methods can localize the predictive regions, they cannot specify which features are relevant, i.e., they can explain where but not why.
    - There is no guarantee that all high-attention/attribution regions carry clinical relevance. High scores just mean that the model has considered these regions more important than others.
- Multimodal data interconnection
    - Morphologic associations
    - Non-invasive alternatives
    - Outcome associations
    - Early predictors
- Challenges
    - Missing data
        - Synthetic data generation
        - Dropout-based methods
    - Data alignment
        - Alignment of similar modalities (e.g. MRI and PET brain scans)
        - Alignment of diverse modalities (e.g. data from different scales, timepoints, or measurements)
    - Transparency and prospective clinical trials

</details>


<details>
<summary>⭐️ [Sep. 2022] <b>Multimodal biomedical AI</b>, <i>Nature Medicine</i></summary>

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







<details>
<summary> <b></b>, <i></i></summary>

[Paper]()
[Code]()
- **Cancer:** 
- **Modalities:** 
- **Data Source:** 
- **Patients:** 
- **Pipeline:** 
- **Fusion Mode:** 
</details>


