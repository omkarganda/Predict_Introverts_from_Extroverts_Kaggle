# Predict the Introverts from the Extroverts
Code and experiments for Kaggle Playground Series S5E7 – Predict the Introverts from the Extroverts. Includes EDA, feature engineering, and a tuned RandomForest baseline that achieves competitive log-loss on the public leaderboard, plus reproducible pipeline and submission scripts.

![img.png](imgs/header.png)


## Kaggle Playground Series – Season 5, Episode 7  

This repository contains all code, notebooks, and documentation for my participation in Kaggle’s **Playground Series – S5E7** competition. The challenge is to classify whether a subject is an *introvert* or an *extrovert* given synthetic behavioural and personality features. The competition uses **log‑loss** as the evaluation metric.
    
### Contents
* `PlayGround_Series_S5_E7.ipynb` – complete EDA, feature engineering, model training, and submission workflow.

### To experiment with different methods
```bash
git clone https://github.com/omkarganda/Predict_Introverts_from_Extroverts_Kaggle.git
```

### Approach
1. **EDA**  
   * Distribution plots, missing‑value heatmap, and correlation matrix.  
   * Discovered moderate positive correlation between features F12 & F15 w.r.t. extroversion.  
2. **Pre‑processing**  
   * KNN imputation for numerical and categorical based on the "distance weightage".  
   * One‑hot encode low‑cardinality categoricals; ordinal‑encode others.  
3. **Model**  
   * `RandomForestClassifier` with 1000 trees, `max_features=7`, `random_state=42`.  
   * 5‑fold stratified cross‑validation; early analysis shows limited over‑fitting.  
4. **Performance**  
   * Mean CV Log‑Loss ≈ **0.155**.  
   * Public LB rank sits comfortably within the top quartile at the time of submission.  


### Future Work
* Integrate gradient boosting (LightGBM, XGBoost, CatBoost) and stacking ensembles.  
* Apply SHAP for feature importance analysis.  
* Automate pipeline with DVC or Kedro for better experiment tracking.  

### Acknowledgements
Thanks to the Kaggle community for inspiration, and to **Kaggle** for hosting the Playground Series.


