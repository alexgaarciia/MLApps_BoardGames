<div align="center">

# Board Games NLP Analysis  
**Final Project Report**

Miguel Fernández Lara (100473125)  
Alejandro Leonardo García Navarro (100472710)

</div>
<br>

## Table of Contents

1. [Introduction](#1-introduction)  
2. [Task 1: Text Preprocessing and Vectorization](#2-task-1-text-preprocessing-and-vectorization)  
   2.1. [Data Collection](#21-data-collection)  
   2.2. [Data Cleaning and Preprocessing](#22-data-cleaning-and-preprocessing)  
   2.3. [Word Vectorization](#23-word-vectorization)  
   2.4. [Word Embeddings](#24-word-embeddings)  
   2.5. [Topic Detection using LDA](#25-topic-detection-using-lda)  
   2.6. [Sentiment Analysis](#26-sentiment-analysis)  
3. [Task 2: Recommender Systems](#3-task-2-recommender-systems)  
   3.1. [Content-Based Recommender System](#31-content-based-recommender-system)  
   3.2. [Hybrid Model Predictions](#32-hybrid-model-predictions)  
   3.3. [Collaborative Filtering](#33-collaborative-filtering)  
4. [Dashboard](#4-dashboard)  
5. [Final Conclusions](#5-final-conclusions)  
6. [References](#6-references)
   
<br>

## 1. Introduction
A way to observe creativity is exploring board games. There are thousands of different board games created to entertain and bring out the competitive spirit of the people around us. Board games come in all types of shapes and themes. Therefore, a thorough analysis of the descriptions of thousands of board games is performed to collect insightful information about them.

To explore this at scale, NLP is used to uncover hidden patterns, identify themes with topic detection, and perform recommender systems for users to get suggestions of board games. 

<br>

## 2. Task 1: Text Preprocessing and Vectorization
### 2.1. Data Collection
The board games are scrapped from the website BGG [<a href="#ref1">1</a>], in which a complete collection of worldwide games is shown. To do so, the BoardGameGeek XML API [<a href="#ref2">2</a>] is accessed to help us scrape the corresponding fields about different games. 

A database of around **4122** games is created with the following fields: 
| **Text Variables** | **Numerical Variables** |
|--------------------|--------------------------|
| `name`             | `year`: Game’s year of publication. |
| `description`: Long texts containing the games’ characteristics and rules. These descriptions are highly detailed and contain a vast amount of information. | `rating`: Average rating given by users on BGG. |
| `categories`: Labels that describe the game’s genre. | `complexity`: How difficult the game is to play. |
|                    | `minplayers` / `maxplayers`: Recommended range of players. |
|                    | `minage`: Suggested minimum age. |

For example, for the game ‘chess’, the following description is found: “Chess is a two-player, abstract strategy board game that represents medieval warfare on an 8x8 board with alternating light and dark squares. Opposing pieces, [...]".

Besides, another database containing a couple million of reviews from users is obtained from Kaggle to deepen the recommender system, called BoardGameGeek Reviews [<a href="#ref3">3</a>]. This dataset contains the user name, id of the game, review, and rating. 

### 2.2. Data Cleaning and Preprocessing
The library SpaCy is used to create the NLP pipeline to process the descriptions. The steps taken include:

1. Token Normalization: The descriptions are set into lower case and HTML tags, extra whitespaces and special characters are removed. Homogenization of unicode characters is not needed, since all the descriptions are in English.
   
2. Token Filtering: Stopwords, numbers and punctuation marks are removed. Note that some custom stopwords have been removed too, including common and uninformative words such as game, player, move, turn and others.
   
3. Token Lemmatization and Tokenization: The words are lemmatized to keep the root. Besides, contractions are expanded to ensure grammatically complete sentences. Finally, the description is tokenized to ensure each word is an individual token.


### 2.3. Word Vectorization
Once the descriptions are clean, vectorization techniques can be applied. To analyze the semantic content of the descriptions, the first and simplest approach to take is to create BoW (Bag of Words) and TF-IDF (Term Frequency-Inverse Document Frequency). To do so, a dictionary is built using Gensim, including all unique terms in the corpus.

The BoW model represents each of the documents as a sparse vector of word counts. Therefore, a list of _(word_id, count)_ pairs can be easily generated, which indicates how many times a word appears in the whole collection of documents. 

On the other hand, TF-IDF is the refinement of BoW, which is done by giving a weight to the words based on their term frequency. Words that are very common are given a small weight, whereas the rare and distinctive words are bound to have a higher weight. In _Figure 1_ one can observe the most relevant words applying TF-IDF vectorization.

<div align="center">
  <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/tfidf.png" width="800"/>
  <p><em>Figure 1: Top terms by TF-IDF Score</em></p>
</div>


### 2.4. Word Embeddings
GloVe (Global Vectors for Word Representation) is used for word embeddings, capturing the semantic similarity between words by learning co-occurrence statistics. Gensim pretrained models _glove-wiki-gigaword-100_ [<a href="#ref4">4</a>] and _glove-wiki-gigarword-300_ [<a href="#ref5">5</a>] are used to provide a 100 and 300-dimensional representation of the corpus. Two versions of GloVe are compared to assess how embedding dimensionality affects the semantic differences between descriptions and analyse the trade-off between expressiveness and computational cost.

To capture the general meaning of the description, the average vector is computed by averaging the embedding number of all known words in a document. To interpret and understand the structure of these embeddings, 2D PCA is performed to reduce their high dimensionality, as observed in _Figure 2_.

<br>
<table align="center">
  <tr>
    <td>
      <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/glove100.png?raw=true" width="500"/>
    </td>
    <td>
      <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/glove300.png?raw=true" width="500"/>
    </td>
  </tr>
</table>
<p align="center"><em>Figure 2: PCA projection of 100 (left) and 300 (right) GloVe embeddings</em></p>
<br>

Taking an initial look, the overall distribution in both plots is entered, reflecting shared vocabulary across many descriptions. However, the 300-dimensional embedding is slightly more spread-out, suggesting that such representation captures more subtle semantic differences between descriptions. This is why the 300-dimensional embedding version is chosen as the appropriate one for the project analysis.

It is important to clarify, however, that these plots help observing patterns in the semantic space. For instance, it can be seen that most points are clustered in the center, indicating that many descriptions have a similar vocabulary. Outliers, on its side, may be games with more unique descriptions.

In order to capture semantics in a different way, Doc2Vec is also used, which is an extension of Word2Vec but it learns embeddings for the whole document directly. The descriptions are encoded into a 300-dimensional vector. Doc2Vec, compared to GloVe, can take into account word order and co-occurrence in the same context window, which provides more detailed embeddings. 

To compare both Glove and Doc2Vec, the 5 nearest neighbours are printed down below to obtain interpretability with a well-known game, ‘Catan’:
<div align="center">

<table>
  <tr>
    <th><strong>Similar games using GloVe</strong></th>
    <th><strong>Similar games using Doc2Vec</strong></th>
  </tr>
  <tr>
    <td>CATAN: Cities & Knights (distance=0.0722)</td>
    <td>New England (distance=0.3499)</td>
  </tr>
  <tr>
    <td>New England (distance=0.0880)</td>
    <td>CATAN: Cities & Knights (distance=0.3579)</td>
  </tr>
  <tr>
    <td>Targui (distance=0.0899)</td>
    <td>Die Magier von Pangea (distance=0.3889)</td>
  </tr>
  <tr>
    <td>Magna Grecia (distance=0.1037)</td>
    <td>Elfenroads (distance=0.4065)</td>
  </tr>
  <tr>
    <td>Space Race (distance=0.1046)</td>
    <td>Magna Grecia (distance=0.4162)</td>
  </tr>
</table>

</div>
<br>

One can observe that some of the neighbours are similar, whereas the distance ranges are completely different. Despite this, GloVe is used in the following sections as the preferred embedding. This is because Glove embeddings are pre-trained on a massive corpus, which allows for generalization even when applied to short texts like game descriptions. 

Most importantly, even though Doc2Vec models can capture word order and context more precisely, they require task-specific training and may not perform well without a large and diverse dataset. Given that the dataset contains only 4122 rows, it is not worth taking the risk.

### 2.5. Topic Detection using LDA
To discover the themes of the board game descriptions, LDA (Latent Dirichlet Allocation) is applied using Gensim. The goal is to obtain the optimal number of themes to describe the data and analyze the distribution across them.

In order to determine the optimal number of themes, a coherence score is used to measure the relevance of themes. It rewards topics where top words frequently co-occur together in the text, which usually means that the topic is semantically meaningful. 

Higher scores (0.4 to 0.6 or higher) generally indicate better topics, while very low scores (< 0.2) often mean that topics are noisy. The optimal number of themes observed is **5**, with a maximum coherence score of **0.44** approximately. Some of the topics obtained are:
- Topic 0: Games centered around movement and positioning on a board (dice, pieces, squares, colors…).
- Topic 1: Resource trading and island exploration (island, buy, sell, market…).
- Topic 2: Trivia games (word, letter, question, answer, category…).
- Topic 3: War or combat simulation games (unit, battle, map, scenario, combat…).
- Topic 4: Card-based games (card, deck, hand, draw, tile…).

<br>
<div align="center">
  <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/coherence_score.png" width="500"/>
  <p><em>Figure 3: Coherence score within topic number</em></p>
</div>

### 2.6. Sentiment Analysis
To understand the emotion and vibe the board games have, sentiment analysis is applied. This is useful for the recommender system to use as extra metadata. As no labels are included in the original dataset, a zero-shot pretrained model is used to classify the games without the need of labeled training data. The labels designed are:

- Emotional tones: _funny, dark, happy, serious, nostalgic, intense_.
- Game vibes: _strategic, cooperative, competitive, family-friendly, chaotic, educational_.

The model used is _facebook/bart-large-mnli_ [<a href="#ref6">6</a>], found in Hugging Face. The main reason why this pretrained transformer is chosen is because it was trained on the Multi-Genre Natural Language Inference (MNLI) dataset [<a href="#ref7">7</a>], which enables it to decide whether a given label can be inferred from the text, making it ideal for the project case. Some considerations must be taken into account about the model before predicting the labels:

- They are trained on natural language, not preprocessed or lemmatized text.
- They expect raw sentences to understand semantic tones.
- They perform better when given more expressive and emotional content, which can get lost when lemmatizing everything.

Because of this, another preprocessing pipeline is included to perform a light data cleaning of the descriptions. 

There is an important aspect to consider here, which is that zero-shot classification with large transformer models is computationally expensive and slow when done sequentially. To optimize performance, the data is converted to a Hugging Face Dataset object, which allows classification in efficient batches. Batching not only reduces runtime significantly, but also allows the model to process multiple descriptions in parallel without affecting the accuracy of individual predictions.

After the classification process with the zero-shot model, two new variables are introduced into the dataset: _predicted_emotion_ and _predicted_vibe_. The former reflects the emotional or narrative mood of the game, while the latter characterizes the gameplay dynamics or player experience. Below can be seen the distribution of these variables:

<br>
<div align="center">
<table>
  <tr>
    <td style="text-align: center;">
      <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/distr_pred_em.png" width="500"/><br>
      <em>Figure 4: Predicted emotion distribution</em>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/distr_pred_vibe.png" width="500"/><br>
      <em>Figure 5: Predicted tone distribution</em>
    </td>
  </tr>
</table>

</div>
<br>

## 3. Task 2: Recommender Systems

### 3.1. Content-Based Recommender System
The first approach used to personalize game suggestions is a content-based recommender system. These models are based on the computation of similarity between games. In particular, the similarity between the embeddings of the different descriptions. 

For this task, GloVe embeddings are used and, to observe how similar the descriptions are, the cosine similarity is measured. Due to the complexity and detail of the descriptions, the similarity between descriptions is high, with a left skewed distribution with mean 0.70 and low variance. 

Another embedding (Sentence-BERT [<a href="#ref8">8</a>]) is applied to try to improve the relationships between descriptions by capturing the context more accurately. This captures deep semantic context using attention but, as observed in _Figure 7_, the distribution of cosine similarities is much broader and centered at lower values compared to GloVe.

This indicates that SBERT is better at distinguishing between descriptions, since it spreads out the similarity scores more. However, because the recommender system depends on having higher similarity values to find related games, the original GloVe embeddings are kept for building the recommendations and provide more stable and intuitive suggestions.

<br>
<div align="center">
<table>
  <tr>
    <td style="text-align: center;">
      <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/distr_cosine_glove.png" width="500"/><br>
      <em>Figure 6: Cosine similarity of GloVe embeddings</em>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/distr_cosine_bert.png" width="500"/><br>
      <em>Figure 7: Cosine similarity of BERT embeddings</em>
    </td>
  </tr>
</table>

</div>
<br>

To implement the content-based recommender system, other metadata is used for filtering: minimum age, number of players, rating and playing time. Therefore, given a user and the former metadata, and including the games the user has rated positively, corresponding recommendations are returned to the user. The predicted score for a candidate game is computed using a similarity-weighted average of the user’s past ratings, as shown in the formula below:

<div align="center">
  <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/formula1.png" width="300"/>
</div>

After testing a few iterations of the recommender system with different users, it is important to note that very similar games in theme are suggested. To fix this, some diversity approaches are implemented. 

MMR (Maximal Marginal Relevance) balances relevance and novelty. It avoids choosing redundant items by selecting documents that are coherent with the user but diverse from the items that have been already selected. MMR is applied iteratively to re-rank the top candidate items. At each step, the item with the highest MMR score is selected using the formula:

<div align="center">
  <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/formula_mmr.png" width="300"/>
</div>

Also, an unsupervised machine learning technique like clustering can be helpful for solving the diversity issue. In particular, Kmeans is applied to obtain different groups based on similarity on the embedding space. The number of clusters is set to 5, since this is the number of themes obtained beforehand. This way, the clusters represent different categories of games. When suggesting games with the recommender system, the model ensures that the adequate games for the user are strictly from different clusters. 

To compare both methods, precision@10 and recall@10 metrics are applied. Some games highly rated by the user are used as a test set. The goal is to observe how many of these diverse games appear in the top 10 suggested games. The results obtained are:

<div align="center">

<table>
  <tr>
    <th></th>
    <th><strong>Basic RecSys</strong></th>
    <th><strong>MMR RecSys</strong></th>
    <th><strong>Clustering RecSys</strong></th>
  </tr>
  <tr>
    <td><strong>Precision@10</strong></td>
    <td>0.0450</td>
    <td>0.0600</td>
    <td>0.0500</td>
  </tr>
  <tr>
    <td><strong>Recall@10</strong></td>
    <td>0.0270</td>
    <td>0.0285</td>
    <td>0.0274</td>
  </tr>
</table>

</div>

The recall@10 measures how many of the relevant items are successfully found in the top 10 recommendations, whereas precision@10 measures how many of the top 10 recommendations are actually relevant. As one can observe, the approach with the best results is MMR. Therefore, this is the content-based recommendation system to be implemented in the dashboard.

To add explainability of the recommended games, some descriptions are added to each game suggested with the reason why it is chosen. This explanation is based on selecting the game with the maximum similarity that the user liked. Besides, it contains the tone and vibe extracted in the sentiment analysis phase. 

For example, an explanation for a recommendation can be: “I recommend this game because you liked ‘CATAN’ (you rated it 8.8/10). This game has a ‘dark’ feeling and is ‘competitive’”. Such explanations support the dashboard by helping users understand the reasoning behind each recommendation, thereby reducing the perception of the system as a black box.

### 3.2. Hybrid Model Predictions
Users may wonder what could be the rating predicted if they choose a new game they haven’t played before. This is the idea behind the implementation of the hybrid model: based on the content of the games and the ratings the user has given to the games in its profile, the new game obtains a prediction of the rating using a weighted average between similarity and ratings.

A key aspect to note is that users with a high number of ratings and diverse games experience some bias in the predicted outcome because the prediction is close to the approximation of the mean of the rated games. To avoid this problem, only the top-k similar games are used as the set for prediction. 

Another problem to take into consideration is cold start. When the user is trying to predict a rating of a game and it does not have any rating yet, it is impossible to accurately suggest a rating. Therefore, some methods are used to tackle this problem: 

**1. Cold start with the user choosing some games:** The user can provide some games they like, which are automatically set to a rating of 9 to simulate prior interactions.
**2. Cold start with popularity-based recommender:** Another option implemented is a popularity based recommender system, in which the user is suggested games liked by the vast majority of people. 

To improve the diversity of the recommendations and avoid suggesting only famous games, clustering is applied using their GloVe embeddings. The Elbow Method (_Figure 8_) is used to choose the number of clusters by analyzing the within-cluster variance. Based on the curve, 8 clusters are selected. To support this decision, t-SNE is implemented to confirm that the games formed different clusters.

<br>
<div align="center">
<table>
  <tr>
    <td style="text-align: center;">
      <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/elbow.png" width="500"/><br>
      <em>Figure 8: Elbow Method</em>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/tsne.png" width="430"/><br>
      <em>Figure 9: t-SNE of K-Means Clusters</em>
    </td>
  </tr>
</table>

</div>
<br>

One top-rated game is then selected from each cluster, based on a popularity score that balances the average rating and the number of ratings:

<div align="center">

$$
\text{Popularity Score} = \frac{\text{Average Rating} \times \text{Number of Ratings}}{\text{Number of Ratings} + 10}
$$

</div>

This ensures that games with many high ratings are favored, while games with only a few ratings are down-weighted to avoid overestimating their quality. For example, if Game A has 1 review and it is a perfect 10, but Game B has 500 reviews and its average rating is 8.9, only considering the average rating is risky and can’t be trusted, that is why games with fewer reviews are down-weighted.

### 3.3. Collaborative Filtering
Contrary to content-based systems, collaborative filtering recommender systems are based solely on user-item interactions. To explore the functionalities of this technique, the ‘Surprise’ library is used. The surprise dataset contains just three fields: User ID, Game ID and Rating. Two parallel methods are implemented: neighbour and latent methods.

The different collaborative filtering neighbour methods consists of predicting relevant games based on the ratings of similar users (user-based) or similar items (item-based). However, for this project only the item-based approach is explored. Two algorithms are implemented: 

**1. KNNBasic:** Uses raw ratings and predicts a user’s rating for a game by computing a weighted average of ratings from similar games.
**2. KNNWithMeans:** An extension of the basic approach by adjusting ratings based on item mean ratings for user bias normalization.

The first step to find the best neighbours method is to implement hyperparameter exploration: the maximum and minimum number of neighbours. A wide range of parameters are tested and plotted the influence in the RMSE metric (_Figures 10-11_), which allowed for a reduction in options in the GridSearch. 

<br>
<div align="center">
<table>
  <tr>
    <td style="text-align: center;">
      <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/exploration_kmax.png" width="500"/><br>
      <em>Figure 10: k_max parameter exploration</em>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/exploration_kmin.png" width="500"/><br>
      <em>Figure 11: min_k parameter exploration</em>
    </td>
  </tr>
</table>

</div>
<br>

During the model selection, two similarity metrics are explored: cosine and pearson, which, contrary to the cosine similarity, calculates the correlation coefficients between all pairs _(user, item)_. The RMSE and MAE are computed using a 3-fold cross validation. The model that is selected as the most optimal is KNNWithMeans with k = 60, min_k = 3 and cosine similarity.

Moving on with the second collaborative filtering approach, latent factor methods are explored. These models uncover hidden relations and patterns in the interaction of the user-item. The goal is to find the optimal latent space size to encode the data.

The first latent factor method used is SVD, which is a powerful matrix factorization technique used to decompose the user-item  matrix in a lower-dimensional space. This way one can easily predict missing ratings effectively. In order to train the SVD model, some parameters must be fine tuned. Firstly, the number of latent dimensions is assessed. Two approaches are tested:

**1. Biased = False:** Only the user and item embeddings are used to reconstruct ratings.
**2. Biased = True:** Apart from using the latent embeddings, bias is incorporated into the user and items. This bias controls the diversity of ratings the users can give. For example, some users can give significantly higher ratings to some games than another user with the same likes, but more moderate mindset. 

The approach with the ‘biased’ parameter provided lower ranges of RMSE when training the SVD model, as observed in the figures below:

<br>
<div align="center">
<table>
  <tr>
    <td style="text-align: center;">
      <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/exploration_biased_false.png" width="500"/><br>
      <em>Figure 12: RMSE Exploration (biased=False)</em>
    </td>
    <td style="text-align: center;">
      <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/exploration_biased_true.png" width="500"/><br>
      <em>Figure 13: RMSE Exploration (biased=True)</em>
    </td>
  </tr>
</table>

</div>
<br>

Besides, other parameters such as ‘lr_all’ (learning rate) and ‘reg_all’ (regularization) are fine tuned to obtain the least error possible.

Furthermore, Non-negative Matrix Factorization (NMF) is also included as an alternative latent factor model. While SVD allows both positive and negative values in the learned user and item factors, NMF restricts them to be non-negative. This results in features that are easier to interpret, because each value reflects how strongly a user or game shows a certain characteristic, without introducing negative influence. For this model, the following parameters are fine tuned: ‘n_factors’ (number of latent dimensions), ‘reg_pu’ (regularization strength for user factors), and ‘reg_qi’ (regularization strength for item factors).

The model selected as the most optimal is SVD with n_factors=50, lr_all=0.005, reg_all=0.1 and biased = True.

To better understand how the model represents games and users, two separate 2D versions of the SVD model are also trained for visualization purposes. One model is used to extract the latent positions of games to show how games are grouped based on user rating patterns. The other is trained to visualize users’ preferences, showing how different user profiles are distributed across the same space.

As observed in _Figure 14_ below, the games can be represented using a 2 dimension latent space, to get a visual interpretable representation. The games that appear near to each other are rated similarly by the same users, indicating some similarity in mechanics, theme or style. Most games lie in a visually dense diagonal band that goes from the lower left up to the right, which suggests a possible high correlation between the two latent dimensions.

<div align="center">
  <img src="https://github.com/alexgaarciia/MLApps_BoardGames/blob/main/images/games_latent_dim.png" width="600"/>
  <p><em>Figure 14: 2-Latent dimension representation of games</em></p>
</div>

As a final remark, comparing the outputs of the collaborative filtering and the content-based suggestions, almost no games match when performing the analysis for the same user. This is expected, since the former is based on similarity of content and the latter on ratings.

<br>

## 4. Dashboard
Finally, as a closing part of the project, a dashboard is developed using Dash. Such a dashboard allows users to explore the dataset and receive personalized game suggestions based on the recommendation strategies explained beforehand.

The main page offers two main pathways: data exploration or personalized recommendation. The exploration section includes interactive visualizations such as TF-IDF bar chart and word cloud to show important keywords in game descriptions, along with an LDA topic map that provides a semantic overview of game clusters.

In case the recommendation pathway is chosen, existing users can enter their username to obtain personalized suggestions, while new users are guided through a cold-start process where they either choose 5 games they like or view popular games. Once a profile is set, 3 recommendation strategies are available: Content-Based Filtering (MMR), Collaborative Filtering, and  Hybrid Recommender. Each method provides personalized and explained results.

<br>

## 5. Final Conclusions
This project combined NLP and recommender systems to analyze board games and build personalized recommendations. We processed a large amount of game descriptions, extracted semantic features, and implemented content-based, collaborative, and hybrid recommenders. A dashboard was also done to show all this work, allowing users to explore not only the dataset, but also game suggestions. 

One of the main difficulties we faced is finding the correct embedding for the descriptions, due to their complex and detailed nature. However, we successfully detected themes and suggested coherent recommendations. Overall, we have learnt how to build reliable recommender systems by combining NLP, user history, and explainability.

<br>

## 6. References
<a name="ref1"></a> [1] BoardGameGeek, “BoardGameGeek - The world's largest source for board game information,” accessed: 09/05/2025. [Online]. Available: https://boardgamegeek.com/  

<a name="ref2"></a> [2] BoardGameGeek, “BGG XML API2,” accessed: 09/05/2025. [Online]. Available: https://boardgamegeek.com/wiki/page/BGG_XML_API2  

<a name="ref3"></a> [3] J. van Elteren, “BoardGameGeek Reviews,” Kaggle, 2022, accessed: 09/05/2025. [Online]. Available: https://www.kaggle.com/datasets/jvanelteren/boardgamegeek-reviews  

<a name="ref4"></a> [4] fse, “glove-wiki-gigaword-100,” Hugging Face, accessed: 09/05/2025. [Online]. Available: https://huggingface.co/fse/glove-wiki-gigaword-100  

<a name="ref5"></a> [5] GerwynnG, “GloVe Wikipedia 2014 + Gigaword 5,” Kaggle, 2023, accessed: 09/05/2025. [Online]. Available: https://www.kaggle.com/datasets/gerwynng/glove-wikipedia-2014-gigaword-5 

<a name="ref6"></a> [6] Facebook AI, “facebook/bart-large-mnli,” Hugging Face, accessed: 09/05/2025. [Online]. Available: https://huggingface.co/facebook/bart-large-mnli  

<a name="ref7"></a> [7] NYU MLL, “multi_nli,” Hugging Face Datasets, accessed: 09/05/2025. [Online]. Available: https://huggingface.co/datasets/nyu-mll/multi_nli

<a name="ref8"></a> [8] Sentence-Transformers, “all-MiniLM-L6-v2,” Hugging Face, accessed: 09/05/2025. [Online]. Available: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
