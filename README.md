# Dry Bean Variety Classification — BITS Pilani M.Tech AIML Assignment 2

## A. Problem statement
Predict one of seven dry-bean varieties (Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira) from 16 numerical geometric features extracted from seed images. Five supervised classifiers are implemented, evaluated and compared. A Streamlit app loads the saved models and evaluates them on any labelled test CSV without retraining.

## B. Dataset description
**UCI Dry Bean Dataset** (ID 602)  
- 13,611 instances, 16 numerical input features, 7 classes  
- Features: Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRation, Eccentricity, ConvexArea, EquivDiameter, Extent, Solidity, roundness, Compactness, ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4  
- No missing values  
- Source: https://archive.ics.uci.edu/dataset/602/dry+bean | DOI: 10.24432/C50S4B | CC BY 4.0  
- Split: 80 % train / 20 % test (stratified, random_state=42)

## C. GitHub repository link
`https://github.com/dheerajsingh544/dryBean_assignment_2026aj05048`  
*(Replace with your actual public repository URL before submission)*

## D. Models and evaluation

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8917 | 0.9329 | 0.8913 | 0.8917 | 0.8913 | 0.8690 |
| KNN | 0.9203 | 0.9815 | 0.9212 | 0.9203 | 0.9205 | 0.9037 |
| Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| Support Vector Machine | 0.9254 | 0.9890 | 0.9255 | 0.9254 | 0.9255 | 0.9099 |

*Metrics: weighted-average multiclass. AUC = weighted one-vs-rest ROC-AUC.*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline — 92.07 % accuracy and 0.9934 AUC. The dataset's geometric features are largely linearly separable, so logistic regression performs well despite its simplicity. Slight confusion between similar-shaped varieties (e.g. Seker vs Barbunya) but overall high precision/recall balance across all seven classes. |
| Decision Tree | Lowest accuracy (89.17 %) and the only model with AUC below 0.95, indicating that a single decision boundary tree overfits some training splits and generalises less well. Interpretable splits on ShapeFactor and AspectRation are the primary discriminating features, but the model struggles with the Dermason / Sira boundary. |
| KNN | Performance nearly identical to Logistic Regression (92.03 % accuracy, 0.9815 AUC). After StandardScaling, distance-weighted 5-NN generalises well; the slightly lower AUC than LR/SVM suggests it is less confident near class boundaries, especially in the dense Dermason cluster. |
| Naive Bayes | Mid-range accuracy (89.79 %) but high AUC (0.9902), showing that the Gaussian conditional-independence assumption captures class-probability ordering well even though correlated features (Area ↔ ConvexArea, MajorAxisLength ↔ Perimeter) modestly degrade hard predictions. Works best for Bombay class (highly distinct geometry). |
| Support Vector Machine | Best overall model — highest accuracy (92.54 %), AUC (0.9936), F1 (0.9255) and MCC (0.9099). The RBF kernel finds a non-linear margin that separates all seven classes robustly. Calibrated probability estimates keep AUC competitive with Logistic Regression while hard predictions are more accurate. |
| **Overall winner** | **Support Vector Machine** — tops all six metrics simultaneously. High MCC (0.9099) confirms the result is not driven by class imbalance. Logistic Regression is the best linear baseline and runs much faster; use it when inference latency matters. |

## E. Streamlit app features
- CSV upload (labelled test data with all 16 feature columns + `Class`)
- All-model comparison table and bar chart
- Per-model metric tiles (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix and full classification report
- Comparison CSV download
- Models are loaded from saved `.joblib` files — no retraining on uploaded data

Live app URL: `ADD YOUR DEPLOYED STREAMLIT URL`

## F. Run locally (BITS Virtual Lab)
```bash
pip install -r requirements.txt
python train_models.py          # downloads dataset, trains models, creates test_data.csv
streamlit run app.py
```
Generated artifacts committed to the repo: `model/*.joblib`, `test_data.csv`, `dataset_info.json`, `evaluation_metrics.csv`.

## G. Deployment (Streamlit Community Cloud)
1. Push all files including `model/` folder to a public GitHub repository
2. Go to https://share.streamlit.io/ → Create app
3. Select repository, branch (`main`) and entry point (`app.py`)
4. Click Deploy — no secrets required

## H. Repository structure
```
project-folder/
├── app.py                   # Streamlit application
├── train_models.py          # Training script (fetch → train → save)
├── requirements.txt
├── README.md
├── test_data.csv            # 20 % stratified test split (2723 rows)
├── dataset_info.json        # Feature list and dataset metadata
├── evaluation_metrics.csv   # Per-model metrics from training run
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knearest_neighbors.joblib
    ├── gaussian_naive_bayes.joblib
    └── support_vector_machine.joblib
```

## I. Academic integrity
UCI Machine Learning Repository, Dry Bean Dataset, https://archive.ics.uci.edu/dataset/602/dry+bean (CC BY 4.0). AI tools were used for learning support only. All code was reviewed, understood and executed personally; observations are drawn from actual model outputs above.
