### 📄 Board_Games_NLP_Analysis.ipynb
Before running the notebook, some considerations must be taken into account:

For task 1:
- Required dataset: ['boardgames_4122.csv'](https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/datasets/boardgames_4122.csv).
- To run the GloVe embedding code, two files are needed: 'glove.6B.100d.txt' and 'glove.6B.300d.txt', which can be found at [GloVe Wikipedia 2014 + Gigaword](https://www.kaggle.com/datasets/gerwynng/glove-wikipedia-2014-gigaword-5).

For task 2:
- There are two options to proceed:
  1. Run the complete first task to obtain the whole process inside the pandas dataframe 'games'.
  2. Load directly ['boardgames_4122_clean_glove_dual_tone.csv'](https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/datasets/boardgames_4122_clean_glove_dual_tone.csv), which contains all the necessary processing to run this second task.
- Load one of the BGG reviews datasets that can be found in [BoardGameGeek Reviews on Kaggle](https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews), such as 'bgg-15m-reviews.csv'. Depending on the computational resources available, more reviews or less can be downloaded (15, 19 or 26 million) . Note that any of these datasets must be renamed to 'reviews' to run the notebook.
