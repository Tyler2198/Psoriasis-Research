# Psoriasis and Male Infertility: Uncovering Feedback Loops and Comorbidome

This repository accompanies the research project titled **"Psoriasis and Male Infertility: Uncovering Feedback Loops and Comorbidome"**, a data-driven investigation into the intricate relationship between psoriasis and male infertility. The study leverages real-world longitudinal data, statistical testing, machine learning, and epidemiological modeling to uncover patterns, risks, and potential causal pathways.

## 📘 Abstract

Psoriasis, a systemic inflammatory skin disorder, has been increasingly associated with male infertility. This study investigates their bidirectional relationship, aiming to:

- Identify comorbidities associated with both conditions.
- Explore temporal sequences (e.g., psoriasis preceding infertility and vice versa).
- Classify patient subgroups using clustering and latent class modeling.
- Construct a *comorbidome* of psoriasis-related infertility risk factors.

## 📊 Dataset

The data was sourced from the **CLALIT Health Services** database in Israel, covering over **280,000** anonymized patient records from **2002 to 2022**, with equal representation of psoriasis and control groups. Among them:

- 138,684 male patients
- 802 diagnosed with infertility
- 122 comorbidities tracked with temporal annotations

## 📈 Methodology

### 1. **Statistical Analysis**
- Chi-Squared tests for comorbidity significance
- Wilson score confidence intervals for prevalence differences

### 2. **Longitudinal Analysis**
- Split analysis: Psoriasis → Infertility and Infertility → Psoriasis
- Risk profiles based on pre-existing and emerging conditions

### 3. **Clustering and Classification**
- Gaussian Mixture Models (GMM), Agglomerative Clustering, Spectral Clustering
- Dimensionality reduction with Multiple Correspondence Analysis (MCA)
- Classification via Random Forest and StepMix (latent class models)

### 4. **Comorbidome Construction**
- Logistic regression to compute odds ratios (ORs)
- Visual comorbidity mapping for male psoriatic patients

### 5. **Matched Cohort Analysis**
- Age-matching control patients to ensure robust comparisons
- Evaluated influence of comorbidities within temporally constrained windows

## 🔍 Key Findings

- Psoriasis is associated with significantly higher prevalence of metabolic, cardiovascular, autoimmune, psychiatric, and oncologic conditions.
- Smoking, obesity, and hyperlipidemia are strong predictors of psoriasis onset.
- Among psoriatic males, infertility correlates with conditions like **multiple sclerosis**, **hyperthyroidism**, and **joint replacement**.
- Reverse chronology (infertility → psoriasis) reveals comorbidities such as **COPD**, **diabetes**, and **chronic renal failure** as relevant.
- High ORs for **hyperprolactinemia** and **peripheral vascular disease** in refined analyses, suggesting novel links.
- Age-matched cohort analysis confirmed trends but highlighted limitations due to small sample size and underreported infertility.

## ⚠️ Limitations

- Imbalanced data with low infertility diagnosis rates (~3%)
- Control group may include undiagnosed infertile individuals
- Certain findings impacted by rarity of comorbid conditions

## 📌 Future Work

- Validate results using a larger, ethnically diverse dataset (e.g., TriNexT with >1,000 infertile males)
- Cluster comorbidities into macro-categories to enhance statistical power
- Integrate IL-23 treatment outcomes to assess therapeutic impact on fertility
- Explore causal inference models and include genetic data

## 📂 Structure

```text
📁 data/                  # Raw or preprocessed datasets (de-identified)
📁 notebooks/             # Jupyter Notebooks for each analysis step
📁 scripts/               # Core Python scripts (preprocessing, modeling, clustering)
📁 Extracts and Reports/  # Full research extracts paper (this PDF)
📁 Tables and Charts/     # Results from Analysis
📄 README.md              # Project overview
```

## 🧠 Citation & Acknowledgments
This project is based on the scientific insights and methodologies curated by researchers in dermatology, reproductive medicine, and computational biology. Special thanks to the Israeli CLALIT Health Services and reference works by Damiani et al. (2023) for foundational evidence on IL-23 inhibitors and male fertility.

If you use this work or build upon it, please cite the main study and reference this GitHub repository.

Author: Denis L. Cascino
Affiliation: BIDSA-Buffa Lab, Università Luigi Bocconi
License: MIT 
