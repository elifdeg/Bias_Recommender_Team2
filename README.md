# Bias_Recommender_Team2
This is our public repository for our project "Reproducing Popularity and Gender Bias in Music Recommenders with Cross-Domain Extension to Books"

Git Hub link: https://github.com/elifdeg/Bias_Recommender_Team2

This project aims to reproduce the results of the paper "Analyzing Item Popularity Bias of Music Recommender Systems: Are Different Genders Equally Affected?" and extends the analysis to the book domain. Our goal is to evaluate whether recommendation algorithms have popularity bias, and whether these biases also persist across different domains. All work was conducted using Python in Jupyter Notebooks.

Project Goals: 
- Reproduce key algorithms and bias metrics in music recommender systems using the dataset LFM 2b.
- Extend the methodology to the book domain to assess cross-domain consistency.
- Evaluate multiple algorithms and compare their behavior with respect to popularity bias.
- Apply mitigation techniques to the best, middle and worst performing algorithms.

Code and Notebooks:
- We created two jupyter notebook files. Notebook "Code_Data_Processing" includes steps and code we used to pre-process two datadets before applying recommendation algorithms. Notebook "Code_Algorithms_Mitigation" includes the code we used to model the recommendation algorithms and metrics as well as their evaluation. It also includes bias mitigation techniques used for both datasets.
- We only used Pyhton.
- We ran some of the code directly in jupyter notebooks web, and some using locally Pycharm.  
- Python libraries used in our code:
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
- Music Domain: [LFM-2b Dataset] from: https://www.cp.jku.at/datasets/LFM-2b/ .The LFM-2b dataset used in our study is considered derivative work according to paragraph 4.1 of Last.fm’s API Terms of Service (https://www.last.fm/api/tos). The Last.fm Terms of Service further grant us a license to use this data (according to paragraph 4). The exact dataset we are using is LFM-2b Dataset, which is a subset of LAST FM dataset and an extension of the LFM-1b dataset and was created by the respective authors of the paper we are replicating "Analyzing Item Popularity Bias of Music Recommender Systems: Are Different Genders Equally Affected?". Unfortunately due to licensing issues (see: https://www.cp.jku.at/datasets/LFM-2b/) the dataset is not avaliable to public. We had to contact the authors ourselves, and they were very kind to provide us with the dataset.

- Book Domain: [Book-Crossing] Dataset from: https://www.kaggle.com/datasets/syedjaferk/book-crossing-dataset/data. The Book-Crossing dataset used in our study is publicly available and labeled as CC0: Public Domain (as stated on its Kaggle distribution page: https://www.kaggle.com/datasets/syedjaferk/book-crossing-dataset?utm_source=chatgpt.com). This permits unrestricted use, including for research and derivative work, without the need for explicit permission or attribution.

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

References:

- Borges, R., & Stefanidis, K. (2021). On mitigating popularity bias in recommendations via variational autoencoders. In Proceedings of the 36th Annual ACM Symposium on Applied Computing (SAC '21) (pp. 1383–1389). Association for Computing Machinery. https://doi.org/10.1145/3412841.3442123
- Cima, R. (2017, June). Bias, she wrote: The gender balance of The New York Times best seller list. The Pudding.Retrieved June 26, 2025, from https://pudding.cool/2017/06/best-sellers/
- Ekstrand, M. D., Tian, M., Kazi, M. R. I., Mehrpouyan, H., & Kluver, D. (2018). Exploring author gender in book rating and recommendation. In Proceedings of the 12th ACM Conference on Recommender Systems (RecSys '18) (pp. 242–250). Association for Computing Machinery. https://doi.org/10.1145/3240323.3240373
- Lesota, O., Melchiorre, A., Rekabsaz, N., Brandl, S., Kowald, D., Lex, E., & Schedl, M. (2021, September). Analyzing item popularity bias of music recommender systems: Are different genders equally affected? In Proceedings of the 15th ACM conference on recommender systems (pp. 601–606).
- Liu, B., Chen, E., & Wang, B. (2023). Reducing popularity bias in recommender systems through AUC-optimal negative sampling. arXiv preprint arXiv:2306.01348. https://arxiv.org/abs/2306.01348
- Melchiorre, A. B., Rekabsaz, N., Parada-Cabaleiro, E., Brandl, S., Lesota, O., & Schedl, M. (2025). Investigating gender fairness of recommendation algorithms in the music domain [GitHub repository]. https://github.com/CPJKU/recommendation_systems_fairness
- Pérez Posada, S. (2025, March 13). Exposing the music industry’s gender bias. Skoove.https://www.skoove.com/blog/music-gender-bias/
- Xv, G., Lin, C., Li, H., Su, J., Ye, W., & Chen, Y. (2022). Neutralizing popularity bias in recommendation models. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '22) (pp. 2623–2628). Association for Computing Machinery. https://doi.org/10.1145/3477495.3531907


