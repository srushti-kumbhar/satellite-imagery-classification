\# Deep Learning-Based Satellite Imagery Classification for Terrain and Target Analysis



\## Project Overview



This project uses deep learning and satellite imagery to classify land-cover categories for environmental and terrain analysis.



It uses the EuroSAT RGB dataset and compares two deep learning approaches:

\- Custom Convolutional Neural Network (CNN)

\- MobileNetV2 Transfer Learning



\## Objectives



\- Classify satellite images into land-cover categories.

\- Identify different environmental and geographical features.

\- Compare CNN and transfer learning performance.

\- Provide an interactive image classification dashboard.



\## Technologies Used



\- Python

\- TensorFlow / Keras

\- NumPy

\- Pandas

\- Matplotlib

\- Seaborn

\- Scikit-learn

\- Streamlit

\- Pillow



\## Dataset



\*\*EuroSAT RGB Dataset\*\*



The dataset contains 27,000 satellite images across 10 categories.



| Category | Description |

|---|---|

| AnnualCrop | Annual agricultural crops |

| Forest | Forest regions |

| HerbaceousVegetation | Herbaceous vegetation |

| Highway | Road networks |

| Industrial | Industrial areas |

| Pasture | Pasture land |

| PermanentCrop | Permanent agricultural crops |

| Residential | Residential areas |

| River | River regions |

| SeaLake | Seas and lakes |



\### Dataset Split



\- Training: 18,900 images (70%)

\- Validation: 4,050 images (15%)

\- Testing: 4,050 images (15%)



\## Models Implemented



\### 1. Baseline CNN

A custom CNN with convolutional layers, max pooling, dropout and dense layers.



\### 2. MobileNetV2

A pretrained MobileNetV2 model used for transfer learning, with a custom classification head and data augmentation.



\## Model Performance



Test-set results:



| Metric | Baseline CNN | MobileNetV2 |

|---|---:|---:|

| Accuracy | 87.70% | 87.19% |

| Precision | 87.58% | 87.41% |

| Recall | 87.70% | 87.19% |

| F1 Score | 87.60% | 87.15% |



\*\*Observation:\*\* The Baseline CNN achieved slightly higher performance than MobileNetV2 on the test dataset.



\## Project Features



\- Satellite image preprocessing

\- Dataset exploration and visualization

\- Data augmentation

\- CNN model training

\- MobileNetV2 transfer learning

\- Model evaluation

\- Confusion matrix visualization

\- Model performance comparison

\- Image upload and prediction

\- Interactive Streamlit dashboard



\## Project Structure



```text

satellite-imagery-classification/

│

├── data/

│   ├── EuroSAT\_RGB/

│   └── processed/

│

├── models/

│   ├── best\_cnn.keras

│   └── best\_mobilenetv2.keras

│

├── results/

│   ├── class\_distribution.png

│   ├── sample\_images.png

│   ├── cnn\_accuracy.png

│   ├── cnn\_loss.png

│   ├── cnn\_confusion\_matrix.png

│   ├── mobilenet\_accuracy.png

│   ├── mobilenet\_confusion\_matrix.png

│   └── model\_comparison.png

│

├── src/

│   ├── dataset.py

│   ├── visualize.py

│   ├── preprocessing.py

│   ├── split\_data.py

│   ├── train.py

│   ├── evaluate.py

│   ├── augmentation.py

│   ├── transfer\_learning.py

│   ├── compare\_models.py

│   ├── evaluate\_transfer.py

│   ├── predict.py

│   └── visualize\_comparison.py

│

├── app.py

└── README.md

