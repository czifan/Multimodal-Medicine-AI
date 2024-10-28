<!-- <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">

<style>
.custom-tag {
    background-color: #EFEFEF;
    border-radius: 10px;
    padding: 3px;
    display: inline-block;
}
</style> -->

# Multimodal Medicine AI

This repository is dedicated to curating research papers on multimodal medicine AI, encompassing reviews, benchmarks, and cutting-edge AI or machine learning techniques for analyzing clinical tasks using multimodal medical data. These tasks include treatment response assessment, prognosis evaluation, diagnosis, recurrence prediction, and more. We also aim to continually gather and update [relevant open datasets](https://github.com/czifan/Multimodal-Medicine-AI/blob/main/multimodal_dataset.md) to support research in multimodal medicine.

> [October 28, 2024] Articles collected so far: 59

## Table of Contents
- [Reviews](#reviews)
- [Benchmarks](#benchmarks)
- [Treatment Response Evaluation](#treatment-response-evaluation)
- [Prognosis Evaluation](#prognosis-evaluation)
- [Others (Dignosis, recurrence prediction, ...)](#others)
- [Multimodal Medical Datasets](https://github.com/czifan/Multimodal-Medicine-AI/blob/main/multimodal_dataset.md)

<details>
<summary>👈 [Tips!!!] The following article can be expanded by clicking on the left triangle</summary>

[Paper](...)
[Code](...)
- **Cancer:** ...
- **Modalities:** ...
- **Data Source:** ...
- **Patients:** ...
- **Pipeline:** 
    - ...
    - ...
- **Fusion Mode:** ...
</details>

---


## Reviews

<details>
<summary>⭐️ [Oct. 2024] <b>Recent Advances in Data-driven Fusion of Multi-modal Imaging and Genomics for Precision Medicine</b>, <i>Information Fusion</i></summary>

[Paper](https://www.sciencedirect.com/science/article/pii/S1566253524005165?casa_token=C33TkmQ9-uwAAAAA:nQ0kLcxtdkrh-XYYcyZwOQDL4bRwtC4Sez8wWXIZP2t4RCPlQyxGOuWuRIyU3oB5Zcn_Qy1mq45S)

**Highlights:** 
- Providing a systematic overview of imaging genomics across multiple organs throughout the body.
- highlighting the recent advancements in data-driven fusion methods for imaging genomics.
- offering detailed reports on multi-modal imaging, imaging-genomics fusion, and clinical applications.
- discussing the current challenges in imaging genomics, providing insights and future prospects.

**Context:**
- Multi-organ Imaging and IDP Extraction (IDP, imaging-derived phenotypes)
    - Brain
        - Modalities: commonly using MRI, including T1-magnetization prepared rapid gradient echo (MPRAGE), T2 fluid-attenuated inversion-recovery (FLAIR), diffusion-weighted imaging (DWI), diffusion tensor imaging (DTI), task-based functional MRI (tfMRI), and resting-state functional MRI (rfMRI).
        - Clinical applications: brain IDPs serve as valuable biomarkers for assessing complex brain diseases, including Parkinson's disease, Alzheimer's disease (AD), ischemic stroke, and neuropsychiatric disorders.
    - Heart
        - Modalities: cardiac IDPs are usually extracted from various imaging modalities, including cardiac MRI (CMR), echocardiography, and CT. Similar to brain imaging, MRI is the primary cardiac imaging method for population genomics studies in cardiology, including T1 mapping, T2 mapping, cine, DTI, Q-flow, coronary imaging, 2D-PC, and 4D-flow.
        - Clinical applications: cardiac IDPs are valuable for exploring associations with diseases such as coronary artery disease (CAD), dilated cardiomyopathy (DCM), hypertrophic cardiomyopathy (HCM), long QT syndrome, pulmonary hypertension, and heart failure
    - Lung
        - Modalities: pulmonary IDPs can be obtained utilizing a range of imaging modalities, including X-ray, CT, MRI, US, and PET/CT. Among these, CT is particularly advantageous due to its high resolution, faster imaging speed and lower cost, making it the preferred and most widely used method for obtaining pulmonary IDPs.
        - Clinical applications: pulmonary IDPs holds promise in the diagnosis and management of various pulmonary conditions, encompassing bacterial lung infections, pulmonary viral and fungal infections, bronchiectasis, lung abscesses, pulmonary fibrosis (PF), emphysema, chronic obstructive pulmonary disease (COPD), and lung tumors
    - Abdomen
        - Modalities: abdominal IDPs, including several organs like liver, kidney, spleen, pancreas as well as visceral and subcutaneous fat, are predominantly obtained through CT and MRI.
        - Clinical applications: these imaging modalities provide valuable insights into obesity, metabolic disorders, and liver disease.
    - Bone
        - Modalities: skeletal imaging techniques, such as photon, dual-energy X-ray absorptiometry (DXA), CT, MRI, and US, provide a rich source of IDPs related to bone and muscle health.
        - Clinical applications: these IDPs play a crucial role in identifying risk factors monitoring disease progression, and assessing the effectiveness of interventions for conditions like osteoporosis, sarcopenia and fractures.
    - Breast
        - Modalities: breast imaging techniques include mammography, ultrasound, PET, CT, and MRI.
        - Clinical applications: these measurements yield valuable information regarding the size, shape, density, and texture of the breast tissue, in addition to identifying any suspicious lesions or calcifications.
- Genome Sequencing
    - Single nucleotide polymorphisms (SNP) is one of the most common types of genetic variation in the human genome.
    - SNP data processing involve collecting and extracting DNA samples from different individuals, detecting SNPs using high-throughput methods, standardizing and analyzing the data by genomewide association studies.
    - Existing technologies that can be used to acquire SNP data include chip-based analysis and sequencing-based analysis:
        - Chip-based analysis uses specific chips to detect the variation of SNP loci in the sample.
        - Sequencing-based SNP analysis utilizes high-throughput sequencing technologies to detect SNP variation by comparing the sample to the reference genome.
    - Common sequencing technologies include Sanger sequencing, second-generation sequencing (such as Illumina and Ion Torrent), and third-generation sequencing (such as PacBio and Oxford Nanopore)
    - Commonly used file formats to store genetic data include VCF, BED, BAM, FASTQ, GTF/GFF, and BGEN.
- Fusion Strategies:
    - Coorelation analysis
        - Genome-wide Association Analysis (GWAS): GWAS and post-GWAS analysis have become foundational in imaging genomics, playing a crucial role in uncovering associations between imaging features and genes.
        - Pathway Analysis: pathway enrichment analysis is a valuable method for identifying specific biological pathways in a group of genes, facilitating the identification of associations between biological functions and phenotypes. (most common databases are Gene Ontology (GO) and Kyoto Encyclopedia of Genes and Genomes (KEGG))
    - Causal Analysis：Mendelian Randomization (MR) is a robust approach for addressing causal questions about the effects of modifiable exposures on various outcomes.
    - Machine Learning Methods
        - Complex Association: canonical correlation analysis (CCA); structured sparse canonical correlation analysis (SCCA); manifold learning techniques; radiogenomics
        - Prediction Model: early, intermediate, or late of data fusion; deep learning for extracting high spatial correlations; transformer models for modeling conplex dependency within gene expression data
- Public Datasets:

| Dataset | Sample size | Data type |
|---|---|---|
| [UK Biobank](https://www.ukbiobank.ac.uk/) | ~500,000 | Brain, heart, abdomen, genotype, others |
| [FHS](https://www.framinghamheartstudy.org/) | 15,000+ | Brain, heart, lung, abdomen |
| [SHIP](https://ship.community-medicine.de) | 12,000+ | Heart, abdomen, genotype, others |
| [ABCD](https://nda.nih.gov.abcd) | 12,000+ | Brain, others |
| [CAHHM](https://cahhm.mcmaster.ca/) | 9,500+ | Brain, heart, abdomen |
| [PNC](https://www.med.upenn.edu/bbl/philadelphianeurodevelopmentalcohort.html) | 95,00+ | Brain, others |
| [MESA](https://www.mesa-nhlbi.org) | 6,800+ | Heart, others |  
| [IMAGEN](https://imagen-project.org/the-imagen-dataset) | 2,000+ | Brain, others |
| [ADNI](https://adni.loni.usc.edu) | 1,200 | Brain, others |
| [CHCP](https://hupi.fudan.edu.cn) | 300+ | Brain, others |

- Clinical Applications
    - Brain
        - Modalities: T1W, MPRAGE, DTI, rfMRI, T2 FLAIR, SWI/QSM
        - Disease: Alzheimer's disease; Parkinson's disease; ischemic stroke; cognitive dysfunction
    - Heart:
        - Modalities: Cine imaging; black blood, bright blood; coronary imaging; 4D flow
        - Disease: dilated cardiomyopathy; coronary artery disease; heart failure; left ventricular dilation
    - Breast
        - Modalities: mammography, ultrasound, MRI
        - Disease: breast cancer; fibrocystic breast; mastitis
    - Lungs
        - Modalities: NCCT
        - Disease: pulmonary artery embolism; interstitial lung disease; lung cancer; emphysema; COPD
    - Abdomen
        - Modalities: NCCT, T1 mapping, T2*, dixon
        - Disease: liver fibroinflammatory; nonalcohollc fatty liver disease; diabetes and the metabolic syndrome; energy expenditure and obestiy
    - Bone
        - Modalities: DXA
        - Disease: fracture; osteoporosis; osteoporotic fracture
- Challenges and Prospects
    - Improving Data Representation: quality control; exraction of phenotypes; voxel-wise genetic analysis; imaging modalities combination
    - Introducting Other Omics: integration of multi-omics analysis
    - Performing Cross-dataset Analysis: data sharing
    - Exploring Machine Learning Algorithms: new algorithms developed; computational limitation
    - Studying Organ Interactions: exploration of whole-body network connections
</details>

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
<summary>⭐️ [Feb. 2023] <b>Multimodal data fusion for cancer biomarker discovery with deep learning</b>, <i>Nature machine intelligence</i> <i>[Perspective]</i>
</summary>

```Perspective```

[Paper](https://www.nature.com/articles/s42256-023-00633-5)

**Background:**
- In oncology, massive amounts of data are being generated, ranging from molecular, histopathology, radiology to clinical records.
- However, most currently AI approaches focus on single data modalities, leading to slow progress in methods to integrate complementary data types.
- Development of effective multimodal fusion approaches is becoming increasingly important as a single modality might not be consistent and sufficient to capture the heterogeneity of complex diseases to tailor medical care and improve personalized medicine.

**View points:**
- **The need for multimodal data fusion in oncology:** Despite huge investments in cancer research and improved diagnosis and treatments, cancer prognosis is still bleak. Predictive models based on single modalities offer a limited view of disease heterogeneity and might not provide sufficient information to stratify patients and capture the full range of events that take place in response to treatments.
- **Data fusion strategies for multimodal biomarker discovery:** The age of precision medicine demands powerful computational techniques to handle high-dimensional multimodal patient data. Each data source has strengths and limitations in its creation, analysis and interpretation that must be addressed.
    - Medical images (2D in histopathology or 3D in radiology) contain dense information that is encoded at multiple scales. So far, the best performing methods have been based on DL, and specifically convolutional neural networks.
    - EHRs comprise various data types ranging from structured data such as medications, diagnosis codes, vital signs or lab tests, to unstructured data in the form of clinical notes, patient emails and detailed clinical processes. Natural language processing (NLP) algorithms that can extract useful clinical information from structured and unstructured EHR data are being developed.
    - Effective fusion methods must integrate high-dimensional multimodal biomedical data, ranging from quantitative features to images and text.
    - A major decision that must be made is at what specific modelling stage the data fusion takes place: (1) early, (2) intermediate or (3) late.
        - While both early and late fusion approaches are model agnostic, they are not specifically designed to cope with or take full advantage of multiple modalities. Anything between early and late fusion is defined as intermediate or joint data fusion.
        - Intermediate fusion does not merge input data, nor develop separate models for each modality, but instead involves the development of inference algorithms to generate a joint multimodal low-level feature representation that retains the signal and properties of each individual modality. Although dedicated inference algorithms must be developed for each model type, this approach attempts to exploit the advantages of both early and late fusion
- **Advances in multimodal biomarkers for patient stratification:** 
    - Multi-omics data fusion: Although a single omics technology provides insights into the profile of a tumour, one technique alone does not fully capture the underlying biology. The increasing collection of large cohorts of multi-omics cancer data has spurred several efforts to fuse multi-omics data to fully grasp the tumour profile and several models for survival and risk prediction have been proposed
    - Multiscale data fusion: Similar efforts as for multi-omics data fusion have been explored for multiscale data, like integrating histopathology, clinical and expression data to predict patient outcomes.
    - imaging genomics and radiogenomics: When possible, molecular tumour information is nowadays used in cancer prognosis and treatment decisions. Interestingly, multiple studies have shown that phenotypes derived from medical images can act as proxies or biomarkers of molecular phenotypes such as an epidermal growth factor receptor (EGFR) mutation in lung cancer.
- **Current challenges and future directions for multimodal data fusion:**
    - *Data hunger*: DL models requires large amounts of data, and both data sparsity and scarcity present serious challenges, especially for biomedical data.
    - *Modality missing*: in clinical practice, there are often different types of data missing between patients, as not all patients might have all modalities owing to cost, insurance coverage, material availability and lack of systemic collection procedures, among others.
    - *Biased cohort*: the depth of data per patient, that is, many observables per patient are routinely generated and stored, but typical cohort sizes of patients are relatively small.
    - *Lack of large 'golden labelled' cohorts*: the lack of large ‘golden labelled’ cohorts with matched multimodal data, mainly due to the intense labour to annotate cancer datasets combined with privacy concerns.
    - *Lack of model interpretation*: while DL can extract predictive features from complex data, these are usually abstract, and it is not always apparent if they are clinically relevant
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

## Benchmarks
<details>
<summary>⭐️ [Aug. 2024] <b>MultiMed: Massively Multimodal and Multitask Medical Understanding</b>, <i>arXiv</i></summary>

[Paper](https://arxiv.org/pdf/2408.12682v1)

**Background:**
- Biomedical data is inherently multimodal, consisting of electronic health records, medical imaging, digital pathology, genome sequencing, wearable sensors, and more.
- The application of AI tools to these multifaceted sensing technologies has the potential to revolutionize the prognosis, diagnosis, and management of human health and disease.
- However, current approaches to biomedical AI typically only train and evaluate with one or a small set of medical modalities and tasks, hampering the development of comprehensive tools that can leverage the rich interconnected information across many heterogeneous.

**Contributions:**
- Presenting MultiMed, a benchmark designed to evaluate and enable large-scale learning across a wide spectrum of medical modalities and tasks.
- MultiMed consists of 2.56 million samples across medical reports, pathology, genomics, and protein data (ten medical modalities), and is structured into eleven challenging tasks, including disease prognosis, protein structure prediction, and medical question answering.
- Based on MultiMed, conducting comprehensive experiments benckmarking state-of-the-art unimodal, multimodal, and multitask models.

**Benchmark details:**
- Modality diversity
    - Imaging modalities: 84,495 OCT images, 194,922 X-ray images, 617,775 CT scans, 7,023 MRI scans, and 27,560 pathology images.
    - Electrophysiological data: MultiMed consists of 120,000 samples designed for the classification of imagined motor imagery time-series data. 
    - Molecular data: 12,560 samples of genomic sequences and 270,000 samples of scRNA-seq data to support expression prediction at the singlecell level; a total of 131,487 protein sequences for protein structure prediction.
    - Text: Clinical notes that complement raw medical signals with rich, descriptive medical narratives with one million image-text pairs.
- Task diversity
    - Disease classification
    - Brain tumor classification
    - Breast cancer classification
    - Radiographic findings classification
    - Bone age classification
    - Diabetic retinopathy classification
    - Imagined motor imagery classification
    - Cell type classification
    - Expression prediction
    - Protein structure prediction
    - Medical Visual question answering

</details>


<details>
<summary>[May 2016] <b>Data Descriptor: MIMIC-III, a freely accessible critical care database</b>, <i>Scientific Data</i></summary>

[Paper](https://www.nature.com/articles/sdata201635)
[Code](https://github.com/MIT-LCP/mimic-website)
[Jupyter-Notebook](https://github.com/MIT-LCP/mimic-iii-paper/)

**Contributions:**
- MIMIC-III (‘Medical Information Mart for Intensive Care’) is a large, single-center database comprising information relating to patients admitted to critical care units at a large tertiary care hospital.
- Data includes vital signs, medications, laboratory measurements, observations and notes charted by care providers, fluid balance, procedure codes, diagnostic codes, imaging reports, hospital length of stay, survival data, and more.
- MIMIC-III critical care database is unique and notable:
    - it is the only freely accessible critical care database of its kind; 
    - the dataset spans more than a decade, with detailed information about individual patient care; 
    - analysis is unrestricted once a data use agreement is accepted, enabling clinical research and education around the world.

**Benchmark details:**
- Patient characteristics
    - MIMIC-III contains data associated with 53,423 distinct hospital admissions for adult patients (aged 16 years or above) admitted to critical care units between 2001 and 2012.
    - In addition, it contains data for 7870 neonates admitted between 2001 and 2008.
    - The data covers 38,597 distinct adult patients and 49,785 hospital admission. The median age of adult patients is 65.8 years (Q1–Q3: 52.8–77.8), 55.9% patients are male, and in-hospital mortality is 11.5%.
    - The median length of an ICU stay is 2.1 days (Q1–Q3: 1.2–4.6) and the median length of a hospital stay is 6.9 days (Q1-Q3: 4.1–11.9). A mean of 4579 charted observations (’chartevents’) and 380 laboratory measurements (’labevents’) are available for each hospital admission.
    - The top three codes across hospital admissions for patients aged 16 years and above were:
        - 414.01 (‘Coronary atherosclerosis of native coronary artery’), accounting for 7.1% of all hospital admissions; 
        - 038.9 (‘Unspecified septicemia’), accounting for 4.2% of all hospital admissions; and 
        - 410.71 (‘Subendocardial infarction, initial episode of care’), accounting for 3.6% of all hospital admissions.
- Classes of data
    - Data available in the MIMIC-III database ranges from time-stamped, nurse-verified physiological measurements made at the bedside to free-text interpretations of imaging studies provided by the radiology department.
</details>

## Treatment Response Evaluation

| Year | Paper | Code | Cancer | Modalities | Data Source | Patients | Fusion Mode |
|-------|-------|------|--------|------------|-------------|----------|-------------|
| 2024 | [🔗](https://www.nature.com/articles/s41392-024-01932-y)| [🔗](https://github.com/czifan/MuMo) | GC | Rad, Path, Cli | In-House | 429 | Middle |
| 2024 | [🔗](https://pubmed.ncbi.nlm.nih.gov/38845006/)| [🔗](https://github.com/FengAoWang/TMO‑Net) | Pan-Cancer | Multi-omics | TCGA | 8174 | Middle |
| 2023 | [🔗](https://www.biorxiv.org/content/10.1101/2023.11.24.568360v1.abstract)| | ccRCC | Gene | TCGA, In-House | ~1000 | Middle |
| 2023 | [🔗](https://www.biorxiv.org/content/10.1101/2023.07.04.547697v1.abstract) | [🔗](https://github.com/rootchang/ICBpredictor) | 18 solid tumor types | Path, Gene, Clin | In-House | 2881 | Middle |
| 2023 | [🔗](https://www.sciencedirect.com/science/article/pii/S0167814023003316?casa_token=MZeMEY7Dz48AAAAA:9iepZVnJHZdhSU0Hmoq-UyajUchgBk1i1ZpoSZTj0NvvdbUaQhJg5ltcoth-iAC0TaVq9abwWA) | [🔗](https://github.com/vancywx/Immunotherapy-response-prediction-using-multi-modal-semi-superviseddeep-learning/tree/main) | GC | Rad, Clin | In-House | 249 | Middle |
| 2023 | [🔗](https://translational-medicine.biomedcentral.com/articles/10.1186/s12967-023-04004-x) | | NSCLC | Rad, Clin | In-House | 264 | Late |
| 2022 | [🔗](https://www.nature.com/articles/s43018-022-00416-8) | [🔗](https://github.com/msk-mind/luna/) | NSCLC | Rad, Path, Gene | In-House | 249 | Middle |
| 2022 | [🔗](https://www.sciencedirect.com/science/article/pii/S1361841522001128) | | | Rad | UKBB, EchoNet-Dynamic, GSTFT, GSTFT CRT | 62 for response predictions and 10,730 for training segmentation models | Middle |
| 2021 | [🔗](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7868825/) | | NSCLC | Rad, Lab, Clin | In-House | 200 | Middle |
| 2020 | [🔗](https://www.thelancet.com/journals/eclinm/article/PIIS2589-5370(20)30123-1/fulltext) | | HCC | Rad | In-House | 737 | Middle |

<details>
<summary>⭐️ [Aug. 2024] <b>Predicting gastric cancer response to anti-HER2 therapy or anti-HER2 combined immunotherapy based on multi-modal data</b>, <i>Signal Transduction and Targeted Therapy</i></summary>

[Paper](https://www.nature.com/articles/s41392-024-01932-y)
[Code](https://github.com/czifan/MuMo)

- **Cancer:** Gastric Cancer (GC)
- **Modalities:** Radiological CT images, radiological structured reports, pathological whole-slide H&E images (WSIs), pathological structured reports, clinical information (age, sex, treatment line, ...) 
- **Data Source:** In-House
- **Patients:** 429 patients: 310 treated with anti-HER2 therapy and 119 treated with a combination of anti-HER2 and anti-PD-1/PD-L1 inhibitors immunotherapy
- **Pipeline:** 
    - using a deep learning extractor to extract radiological deep image features from 3D CT images according to doctors' ROI annotations
    - using a pre-defined one-hot encoder to map radiological structured reports into high-dim features
    - using a multi-scale transformer-based encoder to extract pathological WSIs' features within doctors' ROI annotations
    - using a pre-defined one-hot encoder to map pathological structure reports into high-dim features
    - employing intra-modal fusion module (attention-based) to fuse pathological features and pathological features, respectively
    - conducting a inter-modal fusion module to first separate model-specific features and then align modal-agnostic features from different modalities for multi-modal fusion
    - unilizing a attention-based module to fuse clinical information and multi-modal feature into a patient-level representation
    - leveraging an MLP layer for patient outcome predictions
- **Fusion Mode:** Middle-fusion, employing attention-based module for multi-modal features intergation, and introducing a operation to first separate modal-specific features and then align modal-agnostic features for better multi-modal aggregation
</details>


<details>
<summary>⭐️ [June 2024] <b>TMO-Net: an explainable pretrained multi-omics model for multi-task learning in oncology</b>, <i>Genome Biology</i></summary>

[Paper](https://pubmed.ncbi.nlm.nih.gov/38845006/)
[Code](https://github.com/FengAoWang/TMO‑Net)
[Dataset](https://zenodo.org/records/11258239)

- **Cancer:** Pan-Cancer
- **Modalities:** Gene mutation, mRNA expression, copy number variation (CNV), and DNA methylation
- **Data Source:** TCGA
- **Patients:** 32 pan-cancers from the TCGA database, which consisted of 8174 samples, including normal tissue samples; All processed datasets used for TMO‑Net model pre‑training and fine‑tuning are deposited in [Zenodo](https://zenodo.org/records/11258239)
- **Pipeline:** 
    - developing the Tumor Multi-Omics pre-trained Network (TMO-Net) model, pre-trained with large-scale pan-cancer multi-omics datasets and learn the relationships among individual omics features
    - the model was engineered to accommodate the learning of incomplete omics data
    - utilizing multiple variational autoencoders (VAEs) for capturing associations within self-modal and cross-modal features
    - integrating a "Cross Fusion Module" to efficiently align latent spaces from different modalities and faciliting the inference of missing modalities
    - fusing each omics type's embeddings into a comprehensive joint multi-omics sample embedding via summation
    - pre-training TMO-Net in large-scale pan-cancer multi-omics datasets and then finetuning it to solve several downstream tasks, i.e. breast cancer subtype prediction, primary and metastatic cancer samples separation, drug response prediction, and pan-cancers prognositic prediction
- **Fusion Mode:** Middle-fusion, using summation to integrate multimodal embeddings
</details>


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
| 2024 | [🔗](https://ieeexplore.ieee.org/document/10669115)| | BLCA, BRCA, GBMLGG, LUAD, UCEC | Path, Gene | TCGA | 372+956+569+453+480 | Middle |
| 2024 | [🔗](https://openaccess.thecvf.com/content/CVPR2024/papers/Jaume_Modeling_Dense_Multimodal_Interactions_Between_Biological_Pathways_and_Histology_for_CVPR_2024_paper.pdf)| [🔗](https://github.com/mahmoodlab/SurvPath) | BRCA, BLCA, COADREAD, HNSC, STAD | Path, Gene | TCGA | 869+359+296+392+317 | Middle |
| 2024 | [🔗](https://pubmed.ncbi.nlm.nih.gov/38445478/)| [🔗](https://github.com/mahmoodlab/clam) | ccRCC | Path, Rad, Clin | In-House, TCGA, CPTAC | 414 | Middle |
| 2024 | [🔗](https://www.google.com/search?client=safari&rls=en&q=MOCAT%3A+multi%E2%80%91omics+integration+with%C2%A0auxiliary+classifiers+enhanced+autoencoder&ie=UTF-8&oe=UTF-8)| [🔗](https://github.com/Yaolab-fantastic/MOCAT) | | mRNA, miRNA, methylation | BRCA, ROSMAP, LGG, KIPAN | 351+875+510+658 samples | Middle |
| 2024 | [🔗](https://www.nature.com/articles/s43018-024-00725-0)| | Breast | Gene, Trans, Prot, Meta, Rad, Path | In-House | 773 | Middle |
| 2023 | [🔗](https://arxiv.org/abs/2305.19894) | [🔗](https://github.com/SUSTechBruce/Med-UniC) | | Rad, Text | MIMIC-CXR, PadChest | ~380k pairs | Middle |
| 2023 | [🔗](https://arxiv.org/abs/2303.10390) | | | Rad, Non-imaging | ADNI | 248 | Middle |
| 2022 | [🔗](https://academic.oup.com/bib/article-abstract/23/6/bbac448/6761046) | | BC | Path, Clin, Gene | TCGA | 196 | Middle |
| 2022 | [🔗](https://www.cell.com/cancer-cell/pdf/S1535-6108(22)00317-8.pdf) | [🔗](https://github.com/mahmoodlab/PORPOISE) | Pan-cancer | Path, Molecular profile data | TCGA | 5720 | Middle |
| 2022 | [🔗](https://www.nature.com/articles/s43018-022-00388-9) | [🔗](https://github.com/kmboehm/onco-fusion) | OC | Rad, Path, Clin | MSKCC, TCGA-OV | 444 | Late |
| 2022 | [🔗](https://ieeexplore.ieee.org/abstract/document/10242080) | [🔗](https://github.com/Oulu-IMEDS/CLIMATv2) | | Imaging, Non-Imaging | OAI, ADNI | 4796 (knee OA), 2577 (AD) | Middle |
| 2022 | [🔗](https://ieeexplore.ieee.org/abstract/document/9761545) | [🔗](https://github.com/MIPT-Oulu/CLIMAT) | | X-ray, Non-Imaging | OAI | 4796 | Middle |
| 2022 | [🔗](https://www.sciencedirect.com/science/article/pii/S0933365722000252) | | Brain | Path, Gene | TCGA-LGG, TCGA-GBM | 470 | Middle |
| 2021 | [🔗](https://pubmed.ncbi.nlm.nih.gov/35035786/) | [🔗](https://github.com/bensteven2/HE_breast_recurrence)  | Breast | Path, Clin | In-House, TCGA | 127+123 | Middle |
| 2021 | [🔗](https://ieeexplore.ieee.org/document/9710773) | [🔗](https://github.com/mahmoodlab/MCAT)  | Five cancer types | Path, Gene | TCGA (BLCA, BRCA, GBMLGG, LUAD, UCEC) | 437+1022+1011+515+538 | Middle |
| 2020 | [🔗](https://link.springer.com/chapter/10.1007/978-3-030-66843-3_28) | | Brain | MRIs | BraTS 2019 | 335 | Middle |
| 2020 | [🔗](https://ieeexplore.ieee.org/abstract/document/9186053) | [🔗](https://github.com/mahmoodlab/PathomicFusion) | Glioma, ccRCC | Path, Gene | TCGA-GBM, TCGA-LGG | 769 | Middle |
| 2020 | [🔗](https://pubmed.ncbi.nlm.nih.gov/31797610/) | [🔗](https://github.com/DataX-JieHao/PAGE-Net) | GBM | Path, Gene, Clin | TCGA, TCIA | 447 | Middle |
| 2020 | [🔗](https://academic.oup.com/bioinformatics/article/36/9/2888/5716325) | [🔗](https://github.com/zhang-de-lab/zhang-lab) | ccRCC | Rad, Path, Gene, Clin | TCGA | 209 | Middle |
| 2019 | [🔗](https://pubmed.ncbi.nlm.nih.gov/31586211/) | [🔗](https://cnoc-bwh.shinyapps.io/gbmsurvivalpredictor/) | Glioblastoma | Clin | SEER | 20821 | Early |
| 2019 | [🔗](https://academic.oup.com/bioinformatics/article/35/14/i446/5529139?login=false) | [🔗](https://github.com/gevaertlab/MultimodalPrognosis) | Pancancer | Clin, Gene, Path | TCGA | 11160 | Middle |
| 2017 | [🔗](https://www.cell.com/cell-systems/pdf/S2405-4712(17)30484-2.pdf) | | LUNA | Path, Path Reports, Gene, Proteomics | TCGA | 538 | Middle |

<details>
<summary>⭐️ [Sep. 2024] <b>Cohort-Individual Cooperative Learning for Multimodal Cancer Survival Analysis</b>, <i>IEEE Transactions on Medical Imaging (TMI)</i></summary>

[Paper](https://ieeexplore.ieee.org/document/10669115)
- **Cancer:** Five cancers: Bladder Urothelial Carcinoma, Breast Invasive Carcinoma, Glioblastoma & Lower Grade Glioma, Lung Adenocarcinoma, and Uterine Corpus Endometrial Carcinoma 
- **Modalities:** Genomics, pathological whole-slide images (WSIs)
- **Data Source:** Five TCGA datasets: Bladder Urothelial Carcinoma (BLCA, n=372), Breast Invasive Carcinoma (BRCA, n=956), Glioblastoma & Lower Grade Glioma (GBMLGG, n=569), Lung Adenocarcinoma (LUAD, n=453) and Uterine Corpus Endometrial Carcinoma (UCEC, n=480).
- **Pipeline:**
    - (1) partioning RNA sequencing (RNA-seq), Copy Number Variation (CNV), Simple Nucleotide Variation (SNV) sequences into six sub-sequences; (2) each sub-sequence is transformed into a feature by two cascaded 256-dim self-normalizing Neural Netwoek (SNN) layers; (3) using a fully-connected layer to aggregate the representations of multiple sequences into a single genomic representation (Fg, 256-dim)
    - (1) splitting tissue regions within each WSI into non-overlapping patches at 20x magnification (256x256 resolution for each patch); (2) utilizing an ImageNet pre-trained ResNet-50 model to extract a 1024-dim embedding for each patch; (3) employing the K-means to cluster all patch embeddings within each WSI into k groups and leveraging the cluster centers as pathology features; (4) using cluster center alignment (CCA) to align cluster centers among WSIs; (5) employing a 256-dim SNN layer and a fully-connected layer to aggregate the cluster centers into a single pathology representation (Fp, 256-dim)
    - (1) employing MLP layers as modality encoders (E_path, E_gene) to extract specific knowledge from pathological images and gene, respectively (G, P); (2) employing co-attention blocks as common and synergistic encoders (E_com, E_syn) to integrate knowledge from different modalities into common fused feature and synergistic features (G, S), respectively
    - incorporating additional supervision signals to guarantee the effective extraction of the intended knowledge components: (1) at the knowledge level: loss =|cos(G, Fp)| − cos(G, Fg) − cos(P, Fp) +|cos(P, Fg)| − cos(C, Fp) − cos(C, Fg) +|cos(S, Fp)| + |cos(S, Fg)|; (2) ath the patient level: using contrastive learning to pull patients' embeddings within same groups, while push embeddings across various groups
    - concatenating a class token U with the decomposed G, P, C, and S as the inputs of Transformer, followed by a fully-connected layer with Sigmoid activation to make predictions
- **Fusion Mode:** Middle, employing a Transformer to intergate multimodal (multiple components) embeddings
</details>


<details>
<summary>⭐️ [June 2024] <b>Modeling Dense Multimodal Interactions Between Biological Pathways and Histology for Survival Prediction</b>, <i>CVPR</i></summary>

[Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Jaume_Modeling_Dense_Multimodal_Interactions_Between_Biological_Pathways_and_Histology_for_CVPR_2024_paper.pdf)
[Code](https://github.com/mahmoodlab/SurvPath)
- **Cancer:** 
- **Modalities:** Pathological whole-slide images (WSIs), Gene (bulk transcriptomics)
- **Data Source:** TCGA (five datasets: BRCA, BLCA, COADREAD, HNSC, STAD)
    - Bladder Urothelial Carcinoma (BLCA) (n=359), Breast Invasive Carcinoma (BRCA) (n=869), Stomach Adenocarcinoma (STAD) (n=317), Colon and Rectum Adenocarcinoma (COADREAD) (n=296), and Head and Neck Squamous Cell Carcinoma (HNSC) (n=392).
- **Patients:** 
- **Pipeline:** 
    - using a pathway encoder to tokenize transcriptomics into biological pathway tokens that are semantically meaning ful, interpretable, and end-to-end learnable.
    - using an SSL pre-trained feature extractor to tokenize the corresponding histology whole-slide image into patch tokens.
    - combining pathway and patch tokens using a memory-efficient multimodal Transformer for survival outcome prediction.
- **Fusion Mode:** Middle-fusion, using a memory-efficient multimodal Transformer for intergating multimodal embeddings
</details>

<details>
<summary>⭐️ [Mar. 2024] <b>Deep learning-based multi-model prediction for disease-free survival status of patients with clear cell renal cell carcinoma after surgery: a multicenter cohort study</b>, <i>International Journal of Surgery</i></summary>

[Paper](https://pubmed.ncbi.nlm.nih.gov/38445478/)
[Code](https://github.com/mahmoodlab/clam)
- **Cancer:** Clear cell renal cell carcinoma (ccRCC) after surgery
- **Modalities:** Pathological whole-slide images, CT images, and clinical data
- **Data Source:** 
    - (General cohort) 238 ccRCC patients receiving radical or partial nephrectomy from January 2008 to December 2016 in Renji hospital were included.
    - (TCGA cohort) 137 patients with ccRCC were recruited from [The Cancer Genome Atlas](https://portal.gdc.cancer.gov/). 
    - (CPTAC cohort) 39 ccRCC patients meeting the criteria mentioned above from the Clinical Proteomic Tumor Analysis Consortium
- **Patients:** A total of 414 patients.
- **Pipeline:** 
    - Deep learning-based prediction score (DLPS): for pathological whole-slide images, dividing them into patches (256x256) according to tissue regions; then employing multiple-isntance learning to learn features; last, a CNN was used to convert these patches into 2048-dim feature vectors.
    - Machine learning-based pathomics signature (MLPS): based on [the authors' previous study](https://pubmed.ncbi.nlm.nih.gov/34824449/), identifying five segmentation features.
    - Radiomics prediction score (RADIS): using PyRadiomics to extract 2400 features from CT images within manually delineated RoIs, then 7 radiomics features were selected via least absolute shrinkage and selection operator regression.
    - Multi-modal prediction signature (MMPS): applying cox regression analysis, developing a multi-modal prediction signature (MMPS) based on DLPS, MLPS, RADIS, and clinicopathological features (tumor stage and tumor grade).
- **Fusion Mode:** Middle-fusion, using a cox model to integrate multimodal features

</details>

<details>
<summary>[Mar. 2024] <b>MOCAT: multi‑omics integration with auxiliary classifiers enhanced autoencoder</b>, <i>BioData Mining</i></summary>

[Paper](https://www.google.com/search?client=safari&rls=en&q=MOCAT%3A+multi%E2%80%91omics+integration+with%C2%A0auxiliary+classifiers+enhanced+autoencoder&ie=UTF-8&oe=UTF-8)
[Code][https://github.com/Yaolab-fantastic/MOCAT]
- **Cancer:** Non-Cancer
- **Modalities:** mRNA, miRNA, methylation
- **Data Source:** BRCA, ROSMAP, LGG, and KIPAN
- **Patients:** 
    - ROSMAP: NC: 169, AD: 182
    - BRCA: Luminal A: 436, Luminal B: 147, HER2enriched: 46, Normal-like: 115, Basal-like: 131
    - LGG: Grade 2: 246, Grade 3: 264
    - KIPAN: KICH: 66, KIRC: 318, KIRP: 274
- **Pipeline:**
    - using modal-specific model with autoencoder to extract three modalities' features, respectively
    - concatenating multimodal features and employing multi-head attention to mine interaction of inter- and intra- modality features
    - using a classifier to predict disease status
    - using a ConfNet to refine the predicted probabilities (similar to multi-head classification ensemble)
- **Fusion Mode:** Middle-fusion, concatenating multimodal features
</details>

<details>
<summary>[Feb. 2024] <b>Integrated multiomic profiling of breast cancer in the Chinese population reveals patient stratification and therapeutic vulnerabilities</b>, <i>Nature Cancer</i></summary>

[Paper](https://www.nature.com/articles/s43018-024-00725-0)
- **Cancer:** Breast Cancer
- **Modalities:** Genomic, Transcriptomic, Proteomic, Metabolomic, Radiomic, and Digital pathological characteristics
- **Data Source:** In-House 
- **Patients:** A total of 773 patients with breast cancer nationwide from China who were treated at Fudan University Shanghai Cancer Center during 2013 and 2014
- **Pipeline:** 
    - most of the content analyzes multimodal data from a clinical perspective
    - using OneHotEncoder in scikit-learn to incorporate multimodal discrete features
    - employing a Cox proportional hazards model to predict outcomes
- **Fusion Mode:** Middle-fusion, using OneHotEncoder to incorporate multimodal features

</details>

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
<summary>⭐️ [Aug. 2022] <b>Pan-cancer integrative histology-genomic analysis via multimodal deep learning</b>, <i>Cancer Cell</i></summary>

[Paper](https://www.cell.com/cancer-cell/pdf/S1535-6108(22)00317-8.pdf)
[Code](https://github.com/mahmoodlab/PORPOISE)
- **Cancer:** Pan-cancer, including 14 cancer types
- **Modalities:** Pathological H&E WSIs, Molecular profile data
- **Data Source:** TCGA
- **Patients:** 6592 gigapixel WSIs from 5720 patien samples across 14 cancer types from the TCGA
- **Pipeline:** 
    - using attention-based MIL to extract WSIs' features
    - using MLPs to extract molecular profile data features
    - employing [pathomic fusion](https://pubmed.ncbi.nlm.nih.gov/32881682/) to integrate dual modalities' features
    - using Shapley Additive Explanation (SHAP)-styled attribution decision plots to visualize the attribution weight and direction of each molecular feature
- **Fusion Mode:** Middle-fusion, employing pathomic fusion to integrate multimodal features

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
<summary>⭐️ [Dec. 2021] <b>Prediction of HER2-positive breast cancer recurrence and metastasis risk from histopathological images and clinical information via multimodal deep learning</b>, <i>Computational and Structural Biotechnology Journal</i></summary>

[Paper](https://pubmed.ncbi.nlm.nih.gov/35035786/)
[Code](https://github.com/bensteven2/HE_breast_recurrence)
- **Cancer:** Breast cancer
- **Modalities:** Histopathological images, clinical information
- **Data Source:** In-house, TCGA
- **Patients:** 127 HER2-positive breast cancer patients with known recurrence and matastasis status from Cancer Hospital of the Chinese Academy of Medical Sciences ([Dataset](https://github.com/bensteven2/HE_breast_recurrence)); 123 HER2-positive breast cancer patients with available H&E image and known recurrence and metastasis status in The Cancer Genome Atlas (TCGA)
- **Pipeline:** 
    - dividing histological images into patches and using CNNs for feature extraction
    - integrating image features and clinical features through multimodal compact bilinear (MCB)
    - using a output layer to predict risk scores
- **Fusion Mode:** Middle-fusion, using MCB to integrate multimodal features

</details>


<details>
<summary>⭐️ [Oct. 2021] <b>Multimodal Co-Attention Transformer for Survival Prediction in Gigapixel Whole Slide Images</b>, <i>ICCV</i></summary>

[Paper](https://ieeexplore.ieee.org/document/9710773)
[Code](https://github.com/mahmoodlab/MCAT)
- **Cancer:** Five cancer types
- **Modalities:** Pathological images (WSIs), Gene
- **Data Source:** TCGA, five largest cancer datasets from TCGA
- **Patients:** Bladder Urothelial Carcinoma (BLCA) (n = 437), Breast Invasive Carcinoma (BRCA) (n = 1022), Glioblastoma & Lower Grade Glioma (GBMLGG) (n = 1011), Lung Adenocarcinoma (LUAD) (n = 515), and Uterine Corpus Endometrial Carcinoma (UCEC) (n = 538).
- **Pipeline:** 
    - dividing WSIs into 256x256 patches, and extracting instance-level patch embeddings
    - mapping genomic features into genomic embeddings
    - applying co-attention to enhancing instance-level patch embeddings into genomic-guided WSI embeddings (Query=Gene embeddings; Key & Value=instance-level patch embeddings)
    - employing Transformer to integrating instance-level embeddings into bag-level embeddings
    - concatenating two-modal features and predicting risk scores
- **Fusion Mode:** Middle-fusion, concatenating multimodal features

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
| 2024 | [🔗](https://pmc.ncbi.nlm.nih.gov/articles/PMC11487165/) | [🔗](https://github.com/BioInforCore-BCI/giExtract) | ESCC | Gene, Path | In-House | 120 | Late |
| 2024 | [🔗](https://arxiv.org/pdf/2407.15362) | | Pan-Cancer | Path, Gene, Text | In-House | 10275 | Middle |
| 2024 | [🔗](https://www.nature.com/articles/s41467-024-51888-4) | [🔗](https://github.com/zqiuak/CoPAS) | Non-Cancer (knee abnormalities) | Rad (multi-sequence MRIs) | In-House | 1748 | Middle |
| 2024 | [🔗](https://www.nature.com/articles/s41591-024-03185-2) | [🔗](https://github.com/taokz/BiomedGPT) | | Path, Rad, Clic, Text | 14 freely avaiable datasets | | Middle |
| 2024 | [🔗](https://www.nature.com/articles/s41467-024-50369-y) | [🔗](https://github.com/guichengpeng1/WSI-based-deep-learning-classifier-in-papillary-renal-cell-carcinoma) | pRCC | Gene, Rad, Clin | In-House, TCGA | 793 + 204 | Late |
| 2024 | [🔗](https://www.nature.com/articles/s41591-024-03118-z) | [🔗](https://github.com/vkola-lab/nmed2024) | Non-Cancer (dementia) | Rad, Clin | In-House | 51269 | Middle |
| 2024 | [🔗](https://www.nature.com/articles/s41591-024-02993-w) | [🔗](https://github.com/AIRMEC/HECTOR) | Endometrial | Path, Clin | In-House, TCGA | 2072 | Middle |
| 2024 | [🔗](https://link.springer.com/chapter/10.1007/978-3-031-72390-2_58) | [🔗](https://github.com/JefferyJiang-YF/M4oE) | Non-Cancer | Rad (CT, MRI, CE-MRI) | FLARE22, AMOS22, ATLAS23 | 50+500+100+60 | Middle |
| 2024 | [🔗](https://ieeexplore.ieee.org/document/10539123) | | BRCA, NSCLC, RCC | Path, Gene | TCGA | 737+158+453+450+512+273+109 | Middle |
| 2024 | [🔗](https://www.nature.com/articles/s41467-024-46700-2) | [🔗](https://github.com/Xiao-OMG/OvcaFinder) | Ovarian | Rad, Clin | In-House | 724 | Middle |
| 2024 | [🔗](https://arxiv.org/pdf/2402.14252.pdf) | | | Rad | In-House | | |
| 2023 | [🔗](https://arxiv.org/abs/2304.02836) | [🔗](https://github.com/MASILab/lmsignatures) | SPN | Rad, Clin | NLST, EHR-Pulmonary, Image-EHR, In-House | 2668 (public), 1449 (in-house) | Middle |
| 2023 | [🔗](https://www.nature.com/articles/s41551-023-01045-x) | [🔗](https://github.com/RL4M/IRENE) | | X-rays, Text | In-House | 51511 | Middle |
| 2023 | [🔗](https://www.sciencedirect.com/science/article/pii/S0895611123001179) | | LUNA | Rad, Clin | In-House | 199 | Middle |
| 2022 | [🔗](https://arxiv.org/abs/2212.09162) | [🔗](https://github.com/FirasGit/lsmt) | | Rad, Clin | MIMIC | >40,000 | Middle |
| 2022 | [🔗](https://link.springer.com/chapter/10.1007/978-3-031-16443-9_11) | [🔗](https://github.com/YaoZhang93/mmFormer) | Brain | MRIs | BraTS 2018 | 285 | Middle |
| 2021 | [🔗](https://www.nature.com/articles/s41467-021-23445-w) | [🔗](https://zenodo.org/records/4719434) | Pediatric | Gene (multi-cfDNA-fragment) | In-House | 95+31 | Middle |
| 2021 | [🔗](https://ieeexplore.ieee.org/abstract/document/9366692) |  | | MRIs, PETs | ADNI | 820 | Middle |
| 2021 | [🔗](https://www.nature.com/articles/s41598-020-74399-w) |  | | MRI, Gene, Clin | ADNI | 2004 | Middle |
| 2020 | [🔗](https://www.nature.com/articles/s43018-020-00121-4) | [🔗](https://github.com/PascaDiMagliano-Lab/MultimodalMappingPDA-scRNASeq) | Pancreas | CyTOF, m-IHC, scRNA-seq | In-House | 18+105+19 | N/A |


<details>
<summary>[Oct. 2024] <b>The integrated molecular and histological analysis defines subtypes of esophageal squamous cell carcinoma</b>, <i>Nature Communications</i></summary>

[Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11487165/)
[Code](https://github.com/BioInforCore-BCI/giExtract)
[Analysis-source-code](https://github.com/Zhong2020/ESCCproject)

- **Cancer:** Esophageal Squamous Cell Carcinoma (ESCC)
- **Modalities:** Genomic-transcriptomic, histopathological images
- **Data Source:** In-House
- **Patients:** 120 Chinese ESCC patients
- **Pipeline:** 
    - extracting sample RNA; constructing the sequencing library; performing 150 bp double-ended sequencing; using SOAPnuke and Salmon for data cleaning and comparison; normalizing RNA-seq data for differential expression and GSEA pathway enrichment analysis
    - using non-negative matrix decomposition (NMF) algorithm for unsupervised clustering of RNA-seq data, and then dividing the samples into 4 subtypes
    - using CIBERSORT and Danaher methods to estimate the level of immune cell infiltration in tumor samples and determine the immune subtypes
    - using Agilent SureSelect to capture Exon regions; sequenced by Illumina X Ten platform, and driver genes were identified using methods such as MutSigCV
    - using CNNs to extract histopathological image features, and then filtering subtype-specific histological markers
    - performing XCL1 gene overexpression in ESCC cell lines in vitro; performing SFRP1 tumor growth in nude mice
- **Fusion Mode:** Late-fusion, correlation analysis refers to the analysis of each modality separately and the interpretation of supporting conclusions through the integration of results
</details>


<details>
<summary>[Aug. 2024] <b>A Multimodal Knowledge-enhanced Whole-slide Pathology Foundation Model</b>, <i>arXiv</i></summary>

[Paper](https://arxiv.org/pdf/2407.15362)
- **Cancer:** Pan-Cancer (32 cancer types)
- **Modalities:** pathologicval whole-slide images (WSIs), pathology reports, and RNA-Seq data
- **Data Source:** In-House (26,169 slide-level modality pairs from 10,275 patients across 32 cancer types)
- **Pipeline:**
    - proposing a novel whole-slide pretraining paradigm which injects multimodal knowledge at the whole-slide context into the pathology FM, called Multimodal Self-TAught PRetraining (mSTAR)
    - stage 1: pretrain slide aggregator for the injection of multimodal knowledge: 
        - 1) first, using patch extractor andh a slide aggregator to extract multi-patch features from pasthological whole-slide images; 2) using a tokenizer and a text encoder to extract pathology reports features; 3) using a Gene2Vec modulle and a gene encoder to extract gene features from gene expression profile
        - employing inter-modality contrastive learning to align inter-modal [CLS] embeddings (pathology and text, pathology and gene, text and gene)
        - employing inter-cancer contrastive learning (triplet loss) to distinguish different cancer types to alleviate the hetegrogeneity of various cancer types
    - stage 2: pretrain patch extractor with self-taught training: propagating multimodal knowledge learnedc at the slide level into the patch extractor by self-taught training, levberaging the slide aggregator pretrained in Stage 1 as "Teacher" and enforces patch extractor to be "Student"
    - evaluation tasks: 1) pathologicial slide classification for disgnosis and treatment; 2) pathological survival analysis for prognosis; 3) multimodal capability (few-shot slide classification; zero-shot slide classification; pathological report generation)
- **Fusion Mode:** Middle-fusion, using contrastive learning to align multimodal embeddings and employing concatenation to fuse multimodal embeddings
</details>


<details>
<summary>[Aug. 2024] <b>Learning co-plane attention across MRI sequences for diagnosing twelve types of knee abnormalities</b>, <i>Nature Communications</i></summary>

[Paper](https://www.nature.com/articles/s41467-024-51888-4)
[Code](https://github.com/zqiuak/CoPAS)
- **Cancer:** Non-Cancer (twelve types of knee abnormalities)
- **Modalities:** multi-sequence magnetic resonance imaging (MRIs)
- **Data Source:** In-House (1748 subjedcts, 5 MRI sequences, and 12 types of abnormalities)
- **Pipeline:**
    - for each PDW sequence, the corresponding images of 2 orthogonal planes are generated by rotation. After data preprocessing, the sagittal plane sequence contained 1 original PDW sequence, 2 synthetic PDW sequences, and 1 T2W sequence; the coronal plane sequence contained 1 original PDW sequence, 2 synthetic PDW sequences, and 1 T1W sequence; and the axial plane sequence contained 1 original PDW sequence and 2 synthetic PDW sequences
    - there is a branch network for sagittal plane, coronal plane and axial plane respectively, including encoder module, cross-plane attention module, cross-sequence attention module and branch integration module
    - using the aggregated cross-plane cross-sequence features to predict knee abnormalities' types
- **Fusion Mode:** Middle-fusion, using a cross-plane attention module and a cross-sequence attention module to intergate multi-plane and multi-MRI-sequence embeddings 
</details>


<details>
<summary>[July 2024] <b>A generalist vision–language foundation model for diverse biomedical tasks</b>, <i>Nature Medicine</i></summary>

[Paper](https://www.nature.com/articles/s41591-024-03185-2)
[Code](https://github.com/taokz/BiomedGPT)
- **Cancer:** Non-Cancer (diverse biomedical tasks)
- **Modalities:** image (MRI, X-ary, EKG, ultrasound, microscopy, CT, endoscopy, pathology images), and text (publications, EHRs, literature, clinical notes)
- **Data Source:** 14 freely available datasets (Peir Gross, MedMNIST-raw, SZ-CXR, MedNLI, MIMIC-CXR, MeQSum, PathVQA, MIMIC-III, IU X-ray, VQA-RAD, CBIS-DDSM, SLAKE)
- **Pipeline:**
    - diverse biomedical tasks:
        - **text understanding:** clinical-trail matching, mortality prediction, and treatment suggestion
        - **text summarization:** report summarization, conversation summarization
        - **captioning:** report generation
        - **image classification:** disease diagnosis, lesion detection
        - **VQA:** pathology and radiology VQA
    - using text encoder, 2D image encoder, 3D image encoder, and text encoder to encode instruction, 2D biomedical image, 3D biomedical image, and biomedical text into input embeddings, respectively
    - using a BiomedGPT encoder as well as a BiomedGPT decoder to integrate input embeddings into a discrete output sequence (Biomed GPT model scale: small with 33 million params, medium with 93 params, and base with 182 millon params)
    - employing a text decoder to solve diverse biomedical tasks 
- **Fusion Mode:** Middle-fusion, using a BiomedGPT encoder and a BiomedGPT decoder to integrate multimodal information
</details>


<details>
<summary>[July 2024] <b>A multi-classifier system integrated by clinico-histology-genomic analysis for predicting recurrence of papillary renal cell carcinoma</b>, <i>Nature Communications</i></summary>

[Paper](https://www.nature.com/articles/s41467-024-50369-y)
[Code](https://github.com/guichengpeng1/WSI-based-deep-learning-classifier-in-papillary-renal-cell-carcinoma)
- **Cancer:** papillary Renal Cell Carcinoma (pRCC)
- **Modalities:** lncRNA, pathological whole-slide image (WSI), clinical info. (age, sex, grade and pathologic stage)
- **Data Source:** In-House (793 patients with pRCC) and TCGA (204 cases)
- **Pipeline:**
    - a lncRNA-based classifier
    - a deep learning whole-slide-image-based classifier, built around multiple instance learning and comprised a MobileNet V3 representation network
    - a clinicopathological classifier, built in univariate and multivariate Cox regression analyses
    - developing a multi-classifier system, driven by Cox regression, to integrate the above three modality models' outputs
- **Fusion Mode:** Late-fusion, using Cox regression coefficients to integrate multimodal outputs: Multi classifier risk score = 1.2924 × the lncRNA-based risk score + 2.6315 × the WSI-basedscore + 0.8646 × (0.5670 × grade + 0.5326 × stage)
</details>

<details>
<summary>⭐️ [June 2024] <b>AAI-based differential diagnosis of dementia etiologies on multimodal data</b>, <i>Nature Medicine</i></summary>

[Paper](https://www.nature.com/articles/s41591-024-03118-z)
[Code](https://github.com/vkola-lab/nmed2024)
- **Cancer:** Non-Cancer (differential diagnosis of dementia)
- **Modalities:** Clinical info (demographics, individual and family medical history, medication use, neuropsychological assessments, functional evaluations), multimodal neuroimaging (multi-series MRIs)
    - Demographics: age, gender, race, primary language
    - health history: family history, medication, hypertension
    - Physical: height, weight, BMI, blood pressure
    - Neurological tests: Unified Parkinson’s Disease Rating Scale, Geriatric Depression Scale, Neuropsychiatric Inventory Questionnaire, neuropsychological battery
    - MRI scans: T1w, T2w, FLAIR, DWI, SWI
- **Data Source:** In-House (51,269 participants across 9 independent, geographically diverse datasets, facilitated the identification of 10 distinct dementia etiologies)
    - Training set: n = 38,319 NACC(36,454), AIBL, PPMI, NIFD, LBDSU, OASIS, 4RTNI
    - Testing set: n = 12,950 NACC*(8,895), ADNI, FHS
- **Pipeline:**
    - using a series of steps (e.g. standardizing the data across all cohorts and formatting the features into numerical or categorical variables), and employing a single linear layer to extract numerical info embeddings while utilizing a lookup table to translate categorical inputs into corresponding embeddings
    - to encounter challenges related to missing features and labels, incorporating several strategies such as random feature masking and masking of missing labels 
    - using a series of pre-processing (e.g. skull stripping, linear registration to the MNI space and intensity normalization), and employing the Swin UNETR to extract image embeddings
    - using a Transformer to aggregate these multimodal embeddings
- **Fusion Mode:** Middle-fusion, using a Transformer to integrate multimodal embeddings
</details>

<details>
<summary>⭐️ [May 2024] <b>Prediction of recurrence risk in endometrial cancer with multimodal deep learning</b>, <i>Nature Medicine</i></summary>

[Paper](https://www.nature.com/articles/s41591-024-02993-w)
[Code](https://github.com/AIRMEC/HECTOR)
- **Cancer:** Endometrial Cancer (EC)
- **Modalities:** Pathological H&E image, image-based molecular classes derived from the H&E-based predictions of im4MEC, anatomical stage
- **Data Source:** In-house and TCGA, a total of 2072 patients from eight EC cohorts including the PORTEC-1/-2/-3 randomized trials
- **Pipeline:**
    - using a vision transformer (modified EsVIT) for patch-level, self-supervised representational learning
    - applying a gating-based attention mechanism with biliear product on the embeddings from different modalities to weight the importance of each modality
    - reducing the multimodal embedding by two fully connected (FC) layers before the survival categorical head of a FC layer with output size as the number of discrete time intervals
- **Fusion Mode:** Middle, employing a gating-based attention mechanism with biliear product to intergate multimodal embeddings
</details>


<details>
<summary>[May 2024] <b>Multimodal Co-attention Fusion Network with Online Data Augmentation for Cancer Subtype Classification</b>, <i>IEEE transactions on medical imaging (TMI)</i></summary>

[Paper](https://ieeexplore.ieee.org/document/10539123)

- **Cancer:** BRCA, NSCLC, RCC
- **Modalities:** pathological whole-slide images (WSIs), multi-omics data (mutation status, copy-number variation, and RNA-seq expression)
- **Data Source:** BRCA (IDC: 737; ILC: 168); NSCLC (LUAD: 453; LUSC: 450); RCC (KIRC: 512; KIRP: 273; KICH: 109)
- **Pipeline:**
    - patching WSIs into patches and mapping them into patch embeddings
    - preprocessing multi-omics data and mapping them into gene embeddings
    - employing a mutual-guided co-attention to enhance both modalities' embeddings
    - for WSI branch, using a online data augmentation module to aggregate information:
        - first, according to co-attention module's attention map to scoring and ranking patches
        - then, split them into an attentive group and an inattentive group according to their scores
        - for attentive group, using a mixup-like strategry to augment patch embeddings
        - for inattentive group, fusing them into one embeddings via their average score
    - for multi-omics branch, using a SNN-Mixer, containing one token-mixing SNN and one channel-mixing SNN, to integrate multi-omics information
    - employing global attention pooling (GAP) for both modalities' token embeddings and then concatenate them into a patient-level representation
    - using a classifier for cancer subtype predictions
- **Fusion Mode:** Middle, employing co-attention for enhancing both modalities' embeddings and finally concatenate multimodal embeddings for predicting cancer subtypes
</details>


<details>
<summary>[May 2024] <b>M4oE: A Foundation Model for Medical Multimodal Image Segmentation with Mixture of Experts</b>, <i>MICCAI 2024</i></summary>

[Paper](https://link.springer.com/chapter/10.1007/978-3-031-72390-2_58)
[Code](https://github.com/JefferyJiang-YF/M4oE)
- **Cancer:** Non-Cacner
- **Modalities:** CTG, MRI, CE-MRI
- **Data Source:** Three multimodal medical image segementation datasets: 1) FLARE22 with 50 labeled CT scan cases; 2) AMOS22 with 600 labeled cases comprising 500 CT and 100 MRI abdominal scans; 3) ATLAS23 with 60 cases of contrast-enhanced MRI (CE-MRI) liver scans
- **Pipeline:**
    - M4oE comprises modality-specific experts; each separately initialized to learn features encoding domain knowledge
    - subsequently, a gating network is integrated during fine-tuning to modulate each expert's contribution to the collective predictions dynamically
- **Fusion Mode:** Middle, employing a gating network to intergate each expers's contribution
</details>

<details>
<summary>[Mar. 2024] <b>Development and validation of an interpretable model integrating multimodal information for improving ovarian cancer diagnosis</b>, <i>Nature Communications</i></summary>

[Paper](https://www.nature.com/articles/s41467-024-46700-2)
[Code](https://github.com/Xiao-OMG/OvcaFinder)
- **Cancer:** Ovarian Cancer
- **Modalities:** Radiological ultrasound images, routine clinical variables (patient's age, lesion diameter, and CA125 concentration), O-RADS scores
- **Data Source:** In-House, there were 3972 B-mode and colour Doppler ultrasound images of 296 (40.9%) benign and 428 (59.1%) malignant lesions from 724 patients in SYSUCC
- **Pipeline:**
    - using six image-based DL models (DenseNet121, DenseNet169, DenseNet201, ResNet34, EfficientNet-b5, and EfficientNet-b6, initialised with ImageNet pretrained weights) to predict image-based scores, then ensembling (averaging) the predictions of these six models
    -  Thress clinical factors, O-RADS scores diagnosed by readers, and DL-based predictions were used to build the input with 5-dim vectors via Random Forest (RF) algorithm
- **Fusion Mode:** Middle-fusion, using RF algorithm to integrate image-based scores and clinical factors
</details>

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
<summary>[May 2021] <b>Multimodal analysis of cell-free DNA whole-genome sequencing for pediatric cancers with low mutational burden</b>, <i>Nature Communications</i></summary>

[Paper](https://www.nature.com/articles/s41467-021-23445-w)
[Code](https://zenodo.org/records/4719434)
[Analysis-source-code](https://medical-epigenomics.org/papers/peneder2020_f17c4e3befc643ffbb31e69f43630748/#code)

- **Cancer:** Pediatric Cancers
- **Modalities:** Gene (multiple cfDNA fragment-based metrics)
- **Data Source:** In-House
- **Patients:** 241 deep whole-genome sequencing profiles of 95 patients with Ewing sarcoma and 31 patients with other pediatric sarcomas
- **Pipeline:** 
    - presenting an integrative analysis and comparison of fragmentation patterns in this data set, including (i) global fragment-size distribution; (ii) regional fragment-size distribution along the genome; and (iii) fragment coverage at predefined regions-of-interest.
    - introducing a bioinformatic method for accurate quantification of these epigenetic signatures in cfDNA.
    - investigating the clinical associations of cfDNA fragmentation patterns, and introducing a machine learning method that integrates multiple cfDNA fragment-based metrics into highly predictive models for the detection and classification of pediatric solid tumors.
- **Fusion Mode:** Middle-fusion, employing a machine learning method to anlysis multiple cfDNA fragment-based metrics 
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
<summary>[Nov. 2020] <b>Multimodal mapping of the tumor and peripheral blood immune landscape in human pancreatic cancer</b>, <i>Nature cancer</i></summary>

[Paper](https://www.nature.com/articles/s43018-020-00121-4)
[Code](https://github.com/PascaDiMagliano-Lab/MultimodalMappingPDA-scRNASeq)
- **Cancer:** Pancreatic cancer
- **Modalities:** CyTOF, single-cell RNA sequencing, and multiplex immunohistochemistry
- **Data Source:** In-house
- **Patients:** 
    - CyTOF: 10 PDA samples / 8 control samples
    - m-IHCs: 71 PDA and 34 chronic pancreatitis samples
    - single-cell RNA-sequencing: 16 PDA samples / 3 control samples (in total, we sequenced 8,541 cells from adjacent/normal samples and 46,244 cells from PDA, while from the blood samples we sequenced 14,240 cells from four healthy subjects and 55,873 cells from 16 patients with PDA.) 
- **Pipeline:** Performing biological analysis for different modalities respectively
- **Fusion Mode:** N/A

</details>