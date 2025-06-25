# Bias_Recommender_Team2
This is our public repository for our project "Reproducing Popularity and Gender Bias in Music Recommenders with Cross-Domain Extension to Books"

Git Hub link: https://github.com/elifdeg/Bias_Recommender_Team2

This project reproduces the results of the paper "Reproducing Popularity and Gender Bias in Music Recommenders" and extends the analysis to the book domain. Our goal is to evaluate whether recommendation algorithms reinforce popularity bias and gender bias, and whether these biases persist across different domains. All work was conducted using Python in Jupyter Notebooks.

We added all of the code we used in a jupyter notebook. Our Notebook includes code for all 7 algorithms we ran and the metrics we used for both music and books datasets. It also includes code for bias mitigation applied. Results and evaluation of the results can also be find in the notebook. 

Project Goals: 
- Reproduce key algorithms and bias metrics in music recommender systems using the dataset LFM 2b.
- Extend the methodology to the book domain to assess cross-domain consistency.
- Evaluate multiple algorithms and compare their behavior with respect to fairness and bias.
- Apply mitigation techniques to the best, middle and worst performing algorithms.

Methodology:
The paper we reproduced investigates popularity bias in music recommender systems and how this bias differently impacts male and female users. Used data set is "LFM-2b-DemoBias", a curated subset of the Last.fm "LFM-2b" dataset. Filters applied to the dataset: Interactions with play count > 1, tracks with ≥ 5 listeners, users with ≥ 5 tracks. Final dataset included: ~20k users, ~100k tracks, with gender labels. After that we ran 7 recommender algorithms and used delta and similarity metrics to measure, first for the music dataset and then for the books dataset. Evaluation protocol included User-based 5-fold cross-validation and train/validation/test split of users (60/20/20). We evaluated recommendations using NDCG@10 for ranking quality. Metrics were analyzed separately for female and male users to assess if popularity bias varies by gender. At the end we choose the best, middle and worst performing algorithms and applied bias mitigation. All of these were applied first to the music dataset and then to the books datastet.


Datasets: 
- Music Domain: [LFM-2b Dataset] from: https://www.cp.jku.at/datasets/LFM-2b/
- Book Domain: [Book-Crossing] Dataset from: https://www.kaggle.com/datasets/syedjaferk/book-crossing-dataset/data

Algorithms Used:
- Random Item (RAND)
- Most Popular Items (POP)
- Item k-Nearest Neighbors (ItemKNN)
- Sparse Linear Method (SLIM)
- Alternating Least Squares (ALS)
- Matrix factorization with Bayesian Personalized Ranking (BPR)
- Variational Autoencoder (VAE)

Bias Metrics: 
- %Δ𝑀𝑒𝑎𝑛 
- %Δ𝑀𝑒𝑑𝑖𝑎𝑛 
- %Δ𝑉𝑎𝑟. 
- %Δ𝑆𝑘𝑒𝑤 
- %Δ𝐾𝑢𝑟𝑡𝑜𝑠𝑖𝑠 
- 𝐾𝐿 
- Kendall’s 𝜏 
- NDCG@10



