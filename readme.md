# Reproducing DFF-Adapter for Deepfake Detection

A reproduction and extension of **DFF-Adapter: Fine-Grained DINOv2 Tuning with Dual Supervision for Face Forgery Detection**.

This project re-implements the DFF-Adapter architecture from the research paper and investigates its cross-dataset generalization using parameter-efficient fine-tuning.

## Overview

The implementation uses a frozen **DINOv2** backbone enhanced with multi-head LoRA expert adapters. The model is trained with dual supervision:

- **Authenticity detection:** Classifies a face as real or fake.
- **Forgery-type classification:** Predicts the manipulation category using auxiliary supervision.

Due to computational constraints, DINOv2-Small and DINOv2-Base are used instead of the paper's DINOv2-Large.

## Methodology

- Face detection and cropping using MTCNN.
- Frame sampling from video datasets.
- Frozen DINOv2 backbone with trainable LoRA-based adapters.
- Multi-head expert routing with top-3 expert selection.
- Joint authenticity and forgery-type supervision.
- Video-level predictions through frame-score aggregation.

**Extension:** Label smoothing (ε = 0.1) is applied to the auxiliary forgery-type classification loss to investigate its effect on generalization.

## Datasets

| Dataset | Purpose |
|---|---|
| FaceForensics++ (c23) | Training and validation |
| Celeb-DF-v1 | Cross-dataset evaluation |
| Celeb-DF-v2 | Cross-dataset evaluation |
| DFDC subset | Cross-dataset evaluation |

## Key Results

Best-performing configuration: **DINOv2-Small, 30 epochs**

| Dataset | Video-level AUC |
|---|---:|
| FaceForensics++ (validation) | 99.08% |
| Celeb-DF-v1 | 82.98% |
| Celeb-DF-v2 | 86.25% |
| DFDC subset | 88.03% |

Experiments also compare backbone sizes, training duration, and the effect of label smoothing.

*Note: Results are single-seed. Differences in backbone, training protocol, checkpoint selection, and dataset composition should be considered when comparing against the original paper.*

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- DINOv2
- LoRA / Mixture of Experts
- MTCNN
- OpenCV
- scikit-learn
- Kaggle GPU (NVIDIA Tesla T4)

## Team

- **Sudhiksha Mattewada** — DFF-Adapter Architecture & Experimental Analysis
- **Moguluri Sai Siddhardha** — Data, Backbone & Baseline

## Reference

Zhang et al., *Fine-Grained DINO Tuning with Dual Supervision for Face Forgery Detection*, arXiv:2511.12107v1.

This project is an independent re-implementation based on the paper's description, not the authors' official implementation.