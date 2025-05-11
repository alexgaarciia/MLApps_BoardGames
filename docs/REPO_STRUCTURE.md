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


## 📄 Board_Games_NLP_Analysis.ipynb
Before running the notebook, some considerations must be taken into account:

##### Task 1
- Required dataset: boardgames_4122
- To run the GloVe embedding code, two files are needed: glove.6B.100d.txt and glove.6B.300d.txt, which can be found at [GloVe Wikipedia 2014 + Gigaword](https://www.kaggle.com/datasets/gerwynng/glove-wikipedia-2014-gigaword-5)

##### Task 2
- Two options to proceed:
  1. Run the complete first task to obtain the whole process inside the pandas dataframe 'games'
  2. Load directly 'boardgames_4122_clean_glove_dual_tone.csv', which contains all the necessary processing to run this second task.
- Load the 'reviews' dataset that can be found in [BoardGameGeek Reviews on Kaggle](https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews)






