"""
Default model parameters to use when skipping grid search.
These parameters are optimized for producing a reasonable number of topics
without going through the time-consuming grid search process.
"""
# BEST UMAP PARAMETERS - Updated with optimal values from grid search results
BEST_UMAP_PARAMS = {
    'n_neighbors': 5,           # Based on grid search results
    'n_components': 50,         # Based on grid search results
    'min_dist': 0.0,            # Based on grid search results
    'metric': 'cosine',
    'random_state': 42,
    'verbose': True
}

# BEST HDBSCAN PARAMETERS - Updated with optimal values from grid search results
BEST_HDBSCAN_PARAMS = {
    'min_samples': 5,           # Based on grid search results
    'min_cluster_size': 30,     # Based on grid search results
    'metric': 'euclidean',
    'cluster_selection_method': 'eom',
    'prediction_data': True,
    'core_dist_n_jobs': 1
}

# Default UMAP parameters
DEFAULT_UMAP_PARAMS = {
    'n_neighbors': 15,          # Lower value = more local structure = more topics
    'n_components': 50,         # Higher dimensionality = more topics
    'min_dist': 0.0,            # Lower value = tighter clusters = more topics
    'metric': 'cosine',
    'random_state': 42,
    'verbose': True
}

# Default HDBSCAN parameters
DEFAULT_HDBSCAN_PARAMS = {
    'min_samples': 5,           # Lower value = less strict = more topics
    'min_cluster_size': 15,     # Lower value = smaller clusters allowed = more topics
    'metric': 'euclidean',
    'prediction_data': True,
    'core_dist_n_jobs': 1
}

# Default BERTopic parameters
DEFAULT_BERTOPIC_PARAMS = {
    'calculate_probabilities': False,
    'verbose': True
}

# You can define different parameter sets for different scenarios
# For example, if you want to produce more topics:
MORE_TOPICS_UMAP_PARAMS = {
    'n_neighbors': 5,           # Very low value for more local structure
    'n_components': 75,         # Higher dimensionality
    'min_dist': 0.0,            # Tightest clusters
    'metric': 'cosine',
    'random_state': 42,
    'verbose': True
}

MORE_TOPICS_HDBSCAN_PARAMS = {
    'min_samples': 3,           # Very permissive
    'min_cluster_size': 10,     # Allow smaller clusters
    'metric': 'euclidean',
    'prediction_data': True,
    'core_dist_n_jobs': 1
}

# For fewer, more general topics:
FEWER_TOPICS_UMAP_PARAMS = {
    'n_neighbors': 30,          # Higher value for more global structure
    'n_components': 25,         # Lower dimensionality
    'min_dist': 0.3,            # More spread out clusters
    'metric': 'cosine',
    'random_state': 42,
    'verbose': True
}

FEWER_TOPICS_HDBSCAN_PARAMS = {
    'min_samples': 15,          # More strict
    'min_cluster_size': 50,     # Require larger clusters
    'metric': 'euclidean',
    'prediction_data': True,
    'core_dist_n_jobs': 1
} 