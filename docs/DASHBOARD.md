### 🎨 dashboard.ipynb
Before running the notebook, some things must be considered:

- In case you clone the repository, you can simply run the notebook.
- In case you want to try out only the dashboard, these steps must be followed:

  1. Create a folder called `datasets`.
  2. Download the datasets ['boardgames_4122_clean_glove_dual_tone_bert_popularity.csv'](https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/datasets/boardgames_4122_clean_glove_dual_tone_bert_popularity.csv) and ['filtered_reviews.csv'](https://github.com/alexgaarciia/MLApps_BoardGames/releases/tag/datasets), and introduce them in the `datasets` folder.
  3. Create a folder called `assets`, which will be used to let Dash use pyLDAvis.
  4. Download the file ['lda_vis.html'](https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/assets/lda_vis.html) and place it inside.
  5. Create a folder called `models`.
  6. Download the files ['knn_model.pkl'](https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/models/knn_model.pkl) and ['svd_model.pkl'](https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/models/svd_model.pkl) and put them inside.
  7. Run the notebook.

In case the second path is followed, your folder must look like this:
```
📁 root
├── 📂 assets
│   └── 📄 lda_vis.html
├── 📂 datasets
│   ├── 📄 boardgames_4122_clean_glove_dual_tone_bert_popularity.csv
│   └── 📄 filtered_reviews.csv
├── 📂 models
│   ├── 📄 knn_model.pkl
│   └── 📄 svd_model.pkl
└── 📄 dashboard.ipynb
```
