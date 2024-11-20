# Mamba-based Encoder for Liquid-Liquid Phase Separation (LLPS) Prediction

> A novel framework for accurately identifying phase-separated proteins, leveraging contrastive learning and experimental condition integration.

---

## **Overview**

Liquid–liquid phase separation (LLPS) is a critical mechanism in cellular processes, such as protein aggregation and RNA metabolism, by forming membraneless subcellular structures. Accurate identification of phase-separated proteins is essential for advancing our understanding and control of these processes. However, traditional methods are costly and time-intensive, while existing machine learning approaches are often constrained by experimental conditions.

To overcome these limitations, we developed a **Mamba-based Encoder** using **contrastive learning**, incorporating:
- **Separation probability**
- **Protein type**
- **Experimental conditions**

Our model demonstrated outstanding performance:
- **97.1% accuracy** in predicting phase-separated proteins.
- **ROCAUC score of 0.87** in classifying scaffold and client proteins.

Further validation in the **DGHBP-2 drug delivery system** highlighted its potential for condition modulation in drug development.

---

## **Key Features**

- **High Accuracy:** Achieves 97.1% prediction accuracy.
- **Robust Classification:** Distinguishes scaffold and client proteins with a ROCAUC of 0.87.
- **Context-Aware Learning:** Incorporates experimental conditions for better generalization.
- **Validation in Drug Development:** Proven potential in modulating conditions in the DGHBP-2 system.

---

## **Model Architecture**

Below is the model architecture of the Mamba-based Encoder:

![Model Architecture](model.pdf)

*Figure 1. Schematic representation of the Mamba-based encoder framework.*

- **Input Layer:** Encodes protein sequences and associated experimental conditions.
- **Contrastive Learning Module:** Enhances feature representation through separation probability and protein type.
- **Output Layer:** Predicts phase-separated proteins and classifies scaffold/client proteins.

---

## **Installation**

To use this framework, clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/LLPS-Mamba-Encoder.git](https://github.com/biabiubong/Mambaphase.git
cd Mambaphase
pip install -r requirements.txt
