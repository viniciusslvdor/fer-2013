# FER-2013 Facial Emotion Recognition

Facial emotion recognition using the [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) dataset. The dataset consists of 48x48 pixel grayscale, pre-cropped and centered face images, labeled into 7 emotion categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).

This project started as a machine learning practice exercise and is being revisited with three goals:

- [x] Exploratory data analysis of the dataset
- [ ] Benchmarking multiple models (classic ML baselines, CNNs, transfer learning)
- [ ] A web app for real-time emotion classification from a live webcam feed

## About Dataset

![](https://i.imgur.com/lA3PkYX.png)

The dataset is downloaded via the Kaggle API from the `challenges-in-representation-learning-facial-expression-recognition-challenge` competition. `data/external/icml_face_data.csv` is used as the main source (it already gather pixels, emotion label, and the Training/PublicTest/PrivateTest split in a single file).

## Setup

```
uv sync
```

Create a `.env` file in the project root with your Kaggle credentials:

```
KAGGLE_API_TOKEN=...
```

## Building the processed dataset

`src/dataset.py` loads the raw CSV, decodes the pixel data, removes cross-split duplicate/near-duplicate images (a data leakage source: the same image appearing in both `Training` and an evaluation split), and saves the result to `data/processed/fer2013_clean.npz`:

```
python -m src.dataset
```
