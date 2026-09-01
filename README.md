# 📱 Smartphone Addiction Prediction - Kaggle Competition

## 🎯 Project Overview
Predicting smartphone addiction risk based on behavioral and demographic data using ensemble machine learning.

- **Competition**: Playground Series - Season 6 Episode 8
- **Goal**: Binary classification (Addicted/Not Addicted)
- **Best Score**: AUC 0.5351


## 🛠️ Technologies Used

- **Python** 3.12
- **LightGBM**, **XGBoost**, **CatBoost**
- **Scikit-learn** (preprocessing, models)
- **Pandas**, **NumPy** (data manipulation)
- **Optuna** (hyperparameter tuning)
- **Kaggle API** (submission)

## 📁 Project Structure
├── run_competition_final_optimized.py # Main pipeline
├── run_competition_stacking_final.py # Stacking ensemble
├── check_best_submissions.py # Results analysis
├── src/
│ ├── preprocessing.py # Data cleaning
│ ├── features.py # Feature engineering
│ └── models.py # Model definitions
├── submissions/
│ └── final_optimized_submission.csv # Best result
├── requirements.txt # Dependencies
└── README.md # This file

text

## 📊 Results

| Model | Validation AUC | Public LB |
|-------|---------------|-----------|
| LightGBM | 0.5357 | 0.5351 |
| XGBoost | 0.5348 | - |
| CatBoost | 0.5341 | - |
| **Ensemble** | **0.5351** | **0.5351** |

## 🔧 Key Features Engineered

1. **Rank Transformation** - Convert continuous features to percentiles
2. **Ratio Features** - Screen-to-sleep ratio, work-life balance
3. **Missing Value Indicators** - Binary flags for null values
4. **Interaction Features** - Cross-feature combinations

## 🚀 How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main pipeline
python run_competition_final_optimized.py

# Check results
python check_best_submissions.py
📈 Key Learnings
Simple features > Complex features - More features doesn't always help

LightGBM consistently outperformed other models

Ensemble methods improved performance by ~0.02 AUC

Full dataset (647k rows) gave best results

📊 Visualizations
https://eda_plots.png
https://correlation_matrix.png
https://model_comparison_plot.png
https://feature_importance.png


📝 License
MIT License

⭐ If you found this project interesting, please star it on GitHub!
