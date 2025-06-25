# Bias_Recommender_Team2
This is our public repository for our project "Reproducing Popularity and Gender Bias in Music Recommenders with Cross-Domain Extension to Books"

Git Hub link: https://github.com/elifdeg/Bias_Recommender_Team2

This project reproduces the results of the paper "Reproducing Popularity and Gender Bias in Music Recommenders" and extends the analysis to the book domain. Our goal is to evaluate whether recommendation algorithms reinforce popularity bias and gender bias, and whether these biases persist across different domains. All work was conducted using Python in Jupyter Notebooks.

Project Goals: 
- Reproduce key algorithms and bias metrics in music recommender systems using the dataset LFM 2b.
- Extend the methodology to the book domain to assess cross-domain consistency.
- Evaluate multiple algorithms and compare their behavior with respect to fairness and bias.
- Apply mitigation techniques to the best, middle and worst performing algorithms.

Code and Notebook:
- We only used Pyhton.
- We added all of the code we used in a jupyter notebook. Our Notebook includes code for all 7 algorithms we ran and the metrics we used for both music and books datasets.
- We ran some of the code directly in jupyter notebooks web, and some using locally Pycharm.  
- It also includes code for bias mitigation applied.
- Results and evaluation of the results can also be find in the notebook.
- Python libraries used in the code:
    - Standard Library:os, random
    - Data Manipulation & Computation: numpy (np), pandas (pd)
    - Statistics: scipy.stats, scipy.sparse.csr_matrix
    - Machine Learning: scikit-learn (sklearn), linear_model.ElasticNet
    - Linear Algebra: numpy.linalg.solve
    - Deep Learning (PyTorch) : torch, torch.nn, torch.nn.functional, torch.utils.data: Dataset, DataLoader

Methodology:
- The paper we reproduced investigates popularity bias in music recommender systems and how this bias differently impacts male and female users. Used data set is "LFM-2b-DemoBias", a curated subset of the Last.fm "LFM-2b" dataset.
- Filters applied to the dataset: Interactions with play count > 1, tracks with ≥ 5 listeners, users with ≥ 5 tracks. Final dataset included: ~20k users, ~100k tracks, with gender labels.
- After that we ran 7 recommender algorithms and used delta and similarity metrics to measure, first for the music dataset and then for the books dataset. Evaluation protocol included User-based 5-fold cross-validation and train/validation/test split of users (60/20/20).
-  We evaluated recommendations using NDCG@10 for ranking quality. Metrics were analyzed separately for female and male users to assess if popularity bias varies by gender.
-  At the end we choose the best, middle and worst performing algorithms and applied bias mitigation.
-  All of these were applied first to the music dataset and then to the books datastet.

Datasets: 
- Music Domain: [LFM-2b Dataset] from: https://www.cp.jku.at/datasets/LFM-2b/ .The authors state that beacuse of licencing issues the dataset is not available to public. We contacted the authors and they sent us the respective dataset. That is why we only include the link to the webpage where the dataset used to be available, and didn't upload the dataset itself in our repository.
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




