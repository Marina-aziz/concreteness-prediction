import pandas as pd
from embedding_loader import FastTextEmbeddings
from concreteness_predictor import ConcretenessPredictor

# Configuration: file paths and parameters
fasttext_bin_path = 'cc.ar.300.bin'        # Path to FastText .bin model for Arabic
train_csv         = 'training.csv'         # CSV with 'word' and 'concreteness' columns for training
query_csv         = 'ar_knn_disagreement.csv'  # Query CSV with a 'word' column (words of interest)
output_csv        = 'knn_neighbors_detailed.csv'
word_col_train    = 'word'                 # Column name for words in training set
target_col        = 'concreteness'         # Column name for concreteness ratings in training set
word_col_query    = 'word'                 # Column name for query words
k_neighbors       = 20                     # Number of nearest neighbors

# Load embeddings
emb_loader = FastTextEmbeddings(fasttext_bin_path, file_format='bin')
emb_loader.load_embeddings()

# Fit KNN predictor on the full dataset
train_df = pd.read_csv(train_csv, encoding='utf-8')
predictor = ConcretenessPredictor(emb_loader, k=k_neighbors, weighting='distance')
predictor.fit(train_df, word_column=word_col_train, target_column=target_col)

# Read query words
query_df = pd.read_csv(query_csv, encoding='utf-8')
queries = query_df[word_col_query].astype(str).tolist()

# Retrieve nearest neighbors for each query word
records = []
for query in queries:
    # Get vector for query word and find k nearest neighbors (cosine distance)
    vec = emb_loader.get_vector(query)
    distances, indices = predictor.model.kneighbors([vec], n_neighbors=k_neighbors)
    
    for dist, idx in zip(distances[0], indices[0]):
        neighbor_word = predictor.words_[idx]
        cosine_sim   = 1 - dist  # Convert cosine distance to similarity
        records.append({
            'query_word': query,
            'neighbor': neighbor_word,
            'cosine_similarity': round(cosine_sim, 4)
        })

# Save results to CSV
out_df = pd.DataFrame(records, columns=['query_word', 'neighbor', 'cosine_similarity'])
out_df.to_csv(output_csv, index=False, encoding='utf-8')
print(f"Saved neighbor details to {output_csv}")
