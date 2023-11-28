## Multimodal Models in Oncology: Enhancing Treatment Response Evaluation and Prognostic Accuracy

<details>
<summary>June 2022, Nature Cancer, [Multimodal data integration using machine learning improves risk stratification of high-grade serous ovarian cancer](https://www.nature.com/articles/s43018-022-00388-9)</summary>

- **Journal:** Nature Cancer
- **Published Date:** June 2022
- **Cancer:** Ovarian Cancer
- **Modalities:** CT, H&E slide, HDR-DDR
- **Patients**: 444 patients, including 296 patients treated at the Memorial Sloan Kettering Cancer Center (MSKCC) and 148 patients from The Cancer Genome Atlas Ovarian Cancer (TCGA-OV); 40 test cases were randomly sampled from the entire pool of patients with all data modalities available for analysis, and the resting of 404 patients for training
  - 404 training patients: 243 had H&E WSIs, 245 had adnexal lesions on pre-treatment CE-CT, 251 had omental implants on pre-treatment CE-CT
  - 40 test patients: all had omental lesions on CE-CT, H&E WSIs
- **Fusion Mode:** Late-fusion, using a multivariate Cox model to integrate unimodal submodels’ predictions

</details>
