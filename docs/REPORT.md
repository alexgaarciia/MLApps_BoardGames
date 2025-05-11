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
The board games are scrapped from the website BGG [1], in which a complete collection of worldwide games is shown. To do so, the BoardGameGeek XML API [2] is accessed to help us scrape the corresponding fields about different games. 

A database of around **4122** games is created with the following fields: 
| **Text Variables** | **Numerical Variables** |
|--------------------|--------------------------|
| `name`             | `year`: Game’s year of publication. |
| `description`: Long texts containing the games’ characteristics and rules. These descriptions are highly detailed and contain a vast amount of information. | `rating`: Average rating given by users on BGG. |
| `categories`: Labels that describe the game’s genre. | `complexity`: How difficult the game is to play. |
|                    | `minplayers` / `maxplayers`: Recommended range of players. |
|                    | `minage`: Suggested minimum age. |

For example, for the game ‘chess’, the following description is found: “Chess is a two-player, abstract strategy board game that represents medieval warfare on an 8x8 board with alternating light and dark squares. Opposing pieces, [...]".

Besides, another database containing a couple million of reviews from users is obtained from Kaggle to deepen the recommender system, called BoardGameGeek Reviews [3]. This dataset contains the user name, id of the game, review, and rating. 

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
GloVe (Global Vectors for Word Representation) is used for word embeddings, capturing the semantic similarity between words by learning co-occurrence statistics. Gensim pretrained models _glove-wiki-gigaword-100_ [4] and _glove-wiki-gigarword-300_ [5] are used to provide a 100 and 300-dimensional representation of the corpus. Two versions of GloVe are compared to assess how embedding dimensionality affects the semantic differences between descriptions and analyse the trade-off between expressiveness and computational cost.

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

The model used is _facebook/bart-large-mnli_ [6], found in Hugging Face. The main reason why this pretrained transformer is chosen is because it was trained on the Multi-Genre Natural Language Inference (MNLI) dataset [7], which enables it to decide whether a given label can be inferred from the text, making it ideal for the project case. Some considerations must be taken into account about the model before predicting the labels:

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

[Write your section here.]

### 3.2. Hybrid Model Predictions

[Write your section here.]

### 3.3. Collaborative Filtering

[Write your section here.]

## 4. Dashboard

[Write your section here.]

## 5. Final Conclusions

[Write your section here.]

## 6. References

[Add references here.]
