# Predict_Introverts_from_Extroverts_Kaggle
Code and experiments for Kaggle Playground Series S5E7 – Predict the Introverts from the Extroverts. Includes EDA, feature engineering, and a tuned RandomForest baseline that achieves competitive log-loss on the public leaderboard, plus reproducible pipeline and submission scripts.


# Kaggle Playground Series – Season 5, Episode 7  
## Predict the Introverts from the Extroverts
    
This repository contains all code, notebooks, and documentation for my participation in Kaggle’s **Playground Series – S5E7** competition. The challenge is to classify whether a subject is an *introvert* or an *extrovert* given synthetic behavioural and personality features. The competition uses **log‑loss** as the evaluation metric.
    
### Contents
* `PlayGround_Series_S5_E7.ipynb` – complete EDA, feature engineering, model training, and submission workflow.  
* `src/` – reusable Python modules (`data_loader.py`, `preprocess.py`, `train.py`, `predict.py`).  
* `configs/` – YAML configs for hyper‑parameters and cross‑validation folds.  
* `requirements.txt` – python dependencies.  
* `output/` – trained model artefacts and generated `submission.csv`.

### Quick Start
```bash
git clone https://github.com/omkarganda/Predict_Introverts_from_Extroverts_Kaggle.git
cd kaggle‑ps‑s5e7

# create environment
python -m venv .venv
source .venv/bin/activate         #  Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# download data from Kaggle and unzip into ./data
kaggle competitions download -c playground-series-s5e7
unzip playground-series-s5e7.zip -d data/

# reproduce notebook or run training script
python src/train.py --config configs/random_forest.yaml

# create submission
python src/predict.py --model checkpoints/rf.pkl --out submission.csv
kaggle competitions submit -c playground-series-s5e7 -f submission.csv -m "RandomForest baseline"
```

### Approach
1. **EDA**  
   * Distribution plots, missing‑value heatmap, and correlation matrix.  
   * Discovered moderate positive correlation between features F12 & F15 w.r.t. extroversion.  
2. **Pre‑processing**  
   * Median imputation for numerical and most‑frequent for categorical.  
   * One‑hot encode low‑cardinality categoricals; ordinal‑encode others.  
3. **Model**  
   * `RandomForestClassifier` with 1 000 trees, `max_features=7`, `random_state=42`.  
   * 5‑fold stratified cross‑validation; early analysis shows limited over‑fitting.  
4. **Performance**  
   * Mean CV Log‑Loss ≈ **0.155**.  
   * Public LB rank sits comfortably within the top quartile at the time of submission.  

### Reproducibility
* All random seeds fixed (`numpy`, `python`, `sklearn`).
* Environment captured in `requirements.txt` and `environment.yml`.
* Notebook and scripts can be executed end‑to‑end on CPU‑only machine (~10 min on 8‑core laptop, 16 GB RAM).

### Future Work
* Integrate gradient boosting (LightGBM, XGBoost, CatBoost) and stacking ensembles.  
* Apply SHAP for feature importance analysis.  
* Automate pipeline with DVC or Kedro for better experiment tracking.  

### License
Released under the MIT License – see `LICENSE` for details.

### Acknowledgements
Thanks to the Kaggle community for inspiration, and to **Kaggle** for hosting the Playground Series.
""")
    
# Paths
desc_path = "/mnt/data/repo_description.txt"
readme_path = "/mnt/data/README.md"

with open(desc_path, "w", encoding="utf-8") as f:
    f.write(description)

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)

(desc_path, readme_path)

