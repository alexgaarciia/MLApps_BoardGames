# Repository Structure

This document describes the structure of the repository for the final project. You can find the purpose of each folder and key files.
```
📁 MLApps_BoardGames
├── 📂 assets -> Contains auxiliary assets used for modeling themes using LDA
│   ├── 📂 1783ver -> Models with smaller dataset
│   ├── 📄 lda_best_model.gensim
│   └── 📄 lda_vis.html
├── 📂 datasets -> Data created along the steps in the main notebook
│   ├── 📂 1783ver -> Smaller datasets
│   ├── 📄 boardgames_4122.csv
│   ├── 📄 boardgames_4122_clean.csv
│   ├── 📄 boardgames_4122_clean_glove_dual_tone.csv
│   └── 📄 boardgames_4122_clean_glove_dual_tone_bert_popularity.csv
├── 📂 docs -> Contains the dashboard instructions, report and structure of the repository
│   ├── 📄 DASHBOARD.md
│   ├── 📄 REPORT.md
│   └── 📄 REPO_STRUCTURE.md
├── 📂 images -> Pictures used in the report
├── 📂 models -> Saved models of collaborative filtering
│   ├── 📂 1783ver
│   ├── 📄 knn_model.pkl
│   └── 📄 svd_model.pkl
├── 📄 Board_Games_NLP_Analysis.ipynb -> Main notebook
├── 📄 README.md -> Introduction to the project and repository
├── 📄 dashboard.ipynb -> Dashboard code
└── 📄 data_preparation.ipynb -> Data scrapping and preparation code
```
