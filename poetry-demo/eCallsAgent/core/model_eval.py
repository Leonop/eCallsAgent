import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import silhouette_score
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
import logging
import sys
import os
import pandas as pd
from eCallsAgent.config import global_options as gl
import traceback
import gc
import torch
import json
import time
from typing import Tuple, Dict
from itertools import product
from joblib import Memory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('bertopic_processing.log'), logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation and parameter tuning."""
    def __init__(self):
        self.results = []
        # Create cache directory
        self.cache_dir = os.path.join(gl.temp_folder, 'hdbscan_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        # Initialize memory cache
        self.memory = Memory(self.cache_dir, verbose=0)
        
    def compute_coherence_score(self, topic_model, docs):
        """Compute topic coherence score."""
        try:
            # Calculate c_v coherence score
            coherence_scores = []
            topics = topic_model.get_topics()
            
            # Skip -1 topic (outliers)
            topic_words = {}
            for topic_id, topic_items in topics.items():
                if topic_id == -1:
                    continue
                    
                words = []
                for item in topic_items:
                    if isinstance(item, tuple) and len(item) > 0:
                        # Extract the first element (word) from the tuple, regardless of tuple length
                        words.append(item[0])
                    elif not isinstance(item, tuple):
                        # If it's not a tuple, use the item as is
                        words.append(item)
                topic_words[topic_id] = words
            
            # Calculate coherence for each topic
            for topic_id, words in topic_words.items():
                topic_docs = [doc for doc, topic in zip(docs, topic_model.topics_) if topic == topic_id]
                if topic_docs:
                    # Calculate word co-occurrence within the topic
                    word_pairs = [(w1, w2) for i, w1 in enumerate(words) for w2 in words[i+1:]]
                    pair_scores = []
                    
                    for w1, w2 in word_pairs:
                        # Count documents containing both words
                        both = sum(1 for doc in topic_docs if w1 in doc and w2 in doc)
                        w1_count = sum(1 for doc in topic_docs if w1 in doc)
                        w2_count = sum(1 for doc in topic_docs if w2 in doc)
                        
                        # Calculate PMI-based coherence
                        if both > 0:
                            score = np.log((both * len(topic_docs)) / (w1_count * w2_count))
                            pair_scores.append(score)
                            
                    if pair_scores:
                        coherence_scores.append(np.mean(pair_scores))
            
            return np.mean(coherence_scores) if coherence_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error computing coherence score: {e}")
            return 0.0

    def compute_silhouette_score(self, embeddings, labels):
        """Compute silhouette score for clustering quality."""
        try:
            # Convert inputs to numpy arrays and ensure correct types
            embeddings = np.array(embeddings, dtype=np.float32)
            labels = np.array(labels, dtype=np.int32)
            # Ensure embeddings and labels have matching dimensions
            if len(embeddings) != len(labels):
                logger.warning(f"Dimension mismatch: embeddings {len(embeddings)}, labels {len(labels)}")
                # Take the smaller length to match dimensions
                min_len = min(len(embeddings), len(labels))
                embeddings = embeddings[:min_len]
                labels = labels[:min_len]            
            # Debug info
            logger.info(f"Initial labels shape: {labels.shape}, unique labels: {np.unique(labels)}")
            logger.info(f"Initial embeddings shape: {embeddings.shape}")
            
            # Filter out noise points (label -1)
            mask = labels != -1
            n_valid = np.sum(mask)
            logger.info(f"Number of non-noise points: {n_valid}")
            
            if n_valid < 2:
                logger.warning("Not enough non-noise points for silhouette score")
                return 0.0
                
            filtered_embeddings = embeddings[mask]
            filtered_labels = labels[mask]
            
            # Check if we have at least 2 clusters
            unique_labels = np.unique(filtered_labels)
            n_clusters = len(unique_labels)
            logger.info(f"Number of unique clusters (excluding noise): {n_clusters}")
            
            if n_clusters < 2:
                logger.warning("Need at least 2 clusters for silhouette score")
                return 0.0
            
            # Use a sample if dataset is too large
            if len(filtered_embeddings) > 10000:
                indices = np.random.choice(len(filtered_embeddings), 10000, replace=False)
                filtered_embeddings = filtered_embeddings[indices]
                filtered_labels = filtered_labels[indices]
                logger.info(f"Sampled down to 10000 points")
            
            # Ensure we have enough samples per label
            label_counts = np.bincount(filtered_labels)
            min_samples = np.min(label_counts[label_counts > 0])
            logger.info(f"Samples per cluster - min: {min_samples}, max: {np.max(label_counts)}")
            
            logger.warning("Some clusters have less than 2 samples")
            # Remove clusters with less than 2 samples
            valid_labels = np.where(label_counts >= 2)[0]
            mask = np.isin(filtered_labels, valid_labels)
            filtered_embeddings = filtered_embeddings[mask]
            filtered_labels = filtered_labels[mask]
            # Relabel to ensure consecutive integers
            label_map = {old: new for new, old in enumerate(filtered_labels)}
            filtered_labels = np.array([label_map[label] for label in filtered_labels])
            if len(filtered_embeddings) < 2:
                logger.warning("Not enough samples after filtering small clusters")
                return 0.0
            logger.info(f"Final data for silhouette: {len(filtered_embeddings)} points, {len(np.unique(filtered_labels))} clusters")
            
            # Compute silhouette score
            score = silhouette_score(
                filtered_embeddings,
                filtered_labels,
                metric='cosine',
                sample_size=min(10000, len(filtered_embeddings))
            )
            logger.info(f"Computed silhouette score: {score:.4f}")
            return score
            
        except Exception as e:
            logger.error(f"Error computing silhouette score: {e}")
            logger.error(traceback.format_exc())
            return 0.0

    def grid_search(self, docs: list, embeddings: np.ndarray) -> Tuple[BERTopic, Dict]:
        """Perform grid search for optimal parameters."""
        try:
            # Ensure embeddings are float64
            embeddings = embeddings.astype(np.float64)
            
            # Create results directory
            results_dir = os.path.join(gl.models_folder, 'grid_search_results')
            os.makedirs(results_dir, exist_ok=True)
            
            logger.info(f"Starting grid search with {len(docs)} representative documents")
            
            best_score = float('-inf')
            best_model = None
            best_params = None
            results = []
            
            # Get parameter grid from global options
            param_grid = gl.GRID_SEARCH_PARAMETERS
            total_combinations = np.prod([len(values) for values in param_grid.values()])
            
            # Calculate target topics range - More flexible range
            # With 13618 docs, this would give min_topics = 68 (instead of 136)
            min_topics = max(10, len(docs) // 200)  # Reduced divisor to lower minimum topics
            max_topics = min(500, len(docs) // 20)  # Keep the same max
            
            logger.info(f"Target topic range: {min_topics} to {max_topics}")
            target_topics_range = (min_topics, max_topics)
            
            # Track skipped combinations for analysis
            skipped_combinations = 0
            skipped_reasons = {
                'too_few': 0,
                'too_many': 0
            }
            
            combination_count = 0
            for n_neighbors in param_grid['n_neighbors']:
                for n_components in param_grid['n_components']:
                    for min_dist in param_grid['min_dist']:
                        for min_samples in param_grid['min_samples']:
                            for min_cluster_size in param_grid['min_cluster_size']:
                                combination_count += 1
                                logger.info(f"Testing combination {combination_count}/{total_combinations}")
                                
                                try:
                                    # Configure model with current parameters
                                    umap_model = UMAP(
                                        n_neighbors=n_neighbors,
                                        n_components=n_components,
                                        min_dist=min_dist,
                                        metric='cosine',
                                        random_state=42,
                                        verbose=False,
                                        n_jobs=1 if torch.cuda.is_available() else -1,
                                        low_memory=True
                                    )
                                    
                                    hdbscan_model = HDBSCAN(
                                        min_samples=min_samples,
                                        min_cluster_size=min_cluster_size,
                                        metric='euclidean',
                                        prediction_data=True,
                                        core_dist_n_jobs=1,
                                        memory=self.memory,  # Use the initialized memory cache
                                        algorithm='best'
                                    )
                                    
                                    topic_model = BERTopic(
                                        umap_model=umap_model,
                                        hdbscan_model=hdbscan_model,
                                        calculate_probabilities=False,
                                        verbose=False,
                                        seed_topic_list=gl.SEED_TOPICS,
                                        min_topic_size=gl.OPTIMAL_DOCS_PER_TOPIC
                                    )
                                    
                                    # Train the model
                                    topic_model.fit(docs, embeddings)
                                    
                                    # Count topics (excluding -1 noise topic)
                                    topics = set(topic_model.topics_)
                                    if -1 in topics:
                                        topics.remove(-1)
                                    n_topics = len(topics)
                                    
                                    # Check if the number of topics is in the desired range
                                    if target_topics_range[0] <= n_topics <= target_topics_range[1]:
                                        logger.info(f"Valid combination - Topics: {n_topics}")
                                        
                                        # Evaluate the model
                                        coherence_score = self.compute_coherence_score(topic_model, docs)
                                        silhouette_score = self.compute_silhouette_score(embeddings, topic_model.topics_)
                                        
                                        # Calculate distance to target topics
                                        target_topics = (target_topics_range[0] + target_topics_range[1]) // 2
                                        topic_count_score = 1 - abs(n_topics - target_topics) / (target_topics_range[1] - target_topics_range[0])
                                        
                                        # Combined score (weighted)
                                        combined_score = (
                                            0.4 * coherence_score +
                                            0.4 * silhouette_score +
                                            0.2 * topic_count_score
                                        )
                                        
                                        # Save results
                                        result = {
                                            'n_neighbors': n_neighbors,
                                            'n_components': n_components,
                                            'min_dist': min_dist,
                                            'min_samples': min_samples,
                                            'min_cluster_size': min_cluster_size,
                                            'coherence_score': coherence_score,
                                            'silhouette_score': silhouette_score,
                                            'n_topics': n_topics,
                                            'topic_count_score': topic_count_score,
                                            'combined_score': combined_score
                                        }
                                        results.append(result)
                                        
                                        # Update best model if needed
                                        if combined_score > best_score:
                                            best_score = combined_score
                                            best_model = topic_model
                                            best_params = {
                                                'n_neighbors': n_neighbors,
                                                'n_components': n_components,
                                                'min_dist': min_dist,
                                                'min_samples': min_samples,
                                                'min_cluster_size': min_cluster_size
                                            }
                                            logger.info(f"New best score: {best_score:.4f} with {n_topics} topics")
                                    else:
                                        skipped_combinations += 1
                                        if n_topics < target_topics_range[0]:
                                            skipped_reasons['too_few'] += 1
                                        else:
                                            skipped_reasons['too_many'] += 1
                                        logger.info(f"Skipping combination - Topics {n_topics} outside target range {target_topics_range}")
                                        
                                        # Save minimal info about skipped combinations for analysis
                                        result = {
                                            'n_neighbors': n_neighbors,
                                            'n_components': n_components,
                                            'min_dist': min_dist,
                                            'min_samples': min_samples,
                                            'min_cluster_size': min_cluster_size,
                                            'n_topics': n_topics,
                                            'skipped': True
                                        }
                                        results.append(result)
                                    
                                    # Clean up
                                    if not best_model or topic_model != best_model:
                                        del topic_model
                                        gc.collect()
                                        if torch.cuda.is_available():
                                            torch.cuda.empty_cache()
                                
                                except Exception as e:
                                    logger.error(f"Error evaluating parameters: {e}")
                                    logger.error(traceback.format_exc())
                                    continue
            
            # Save results to CSV
            if results:
                results_df = pd.DataFrame(results)
                results_file = os.path.join(results_dir, f'grid_search_results_{time.strftime("%Y%m%d_%H%M%S")}.csv')
                results_df.to_csv(results_file, index=False)
                logger.info(f"Saved grid search results to {results_file}")
            
            # Report on skipped combinations
            logger.info(f"Grid search complete: {len(results) - skipped_combinations} valid combinations, {skipped_combinations} skipped")
            logger.info(f"Skipped reasons: {skipped_reasons}")
            
            # If no valid combinations were found, select the one closest to target range
            if not best_model and results:
                logger.warning("No valid parameter combinations found within target range. Selecting closest match.")
                # Find the combination with the most topics (closest to min_topics)
                closest_result = max([r for r in results if 'skipped' in r], key=lambda x: x['n_topics'])
                logger.info(f"Selected parameters with {closest_result['n_topics']} topics (below target minimum of {min_topics})")
                
                # Use these parameters to create a model
                umap_model = UMAP(
                    n_neighbors=closest_result['n_neighbors'],
                    n_components=closest_result['n_components'],
                    min_dist=closest_result['min_dist'],
                    metric='cosine',
                    random_state=42
                )
                hdbscan_model = HDBSCAN(
                    min_samples=closest_result['min_samples'],
                    min_cluster_size=closest_result['min_cluster_size'],
                    metric='euclidean',
                    prediction_data=True,
                    core_dist_n_jobs=1,
                    memory=self.memory
                )
                best_model = BERTopic(
                    umap_model=umap_model,
                    hdbscan_model=hdbscan_model,
                    calculate_probabilities=False,
                    verbose=True
                )
                best_model.fit(docs, embeddings)
                best_params = {
                    'n_neighbors': closest_result['n_neighbors'],
                    'n_components': closest_result['n_components'],
                    'min_dist': closest_result['min_dist'],
                    'min_samples': closest_result['min_samples'],
                    'min_cluster_size': closest_result['min_cluster_size']
                }
            
            return best_model, best_params
            
        except Exception as e:
            logger.error(f"Error in grid search: {e}")
            logger.error(traceback.format_exc())
            return None, None

    def plot_parameter_effects(self, results):
        """Plot the effects of different parameters on model performance."""
        try:
            df_results = pd.DataFrame(results)
            
            # Create visualization directory
            vis_dir = os.path.join(gl.output_folder, 'parameter_analysis')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Plot relationships between parameters and metrics
            parameters = ['n_neighbors', 'n_components', 'min_dist', 'min_samples', 'min_cluster_size']
            metrics = ['coherence_score', 'silhouette_score', 'n_topics']
            
            for param in parameters:
                fig = plt.figure(figsize=(15, 5))
                for i, metric in enumerate(metrics, 1):
                    plt.subplot(1, 3, i)
                    plt.scatter(df_results[param], df_results[metric], alpha=0.5)
                    plt.xlabel(param)
                    plt.ylabel(metric)
                    plt.title(f'{param} vs {metric}')
                
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, f'{param}_effects.pdf'))
                plt.close()
            
            # Create correlation heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(df_results[parameters + metrics].corr(), annot=True, cmap='coolwarm')
            plt.title('Parameter and Metric Correlations')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'correlation_heatmap.pdf'))
            plt.close()
            
            # Save detailed analysis report
            report = df_results.describe()
            report.to_csv(os.path.join(vis_dir, 'parameter_analysis_report.csv'))
            
            # Find best configurations
            best_coherence = df_results.nlargest(5, 'coherence_score')
            best_silhouette = df_results.nlargest(5, 'silhouette_score')
            
            with open(os.path.join(vis_dir, 'best_configurations.txt'), 'w') as f:
                f.write("Top 5 configurations by coherence score:\n")
                f.write(best_coherence.to_string())
                f.write("\n\nTop 5 configurations by silhouette score:\n")
                f.write(best_silhouette.to_string())
            
        except Exception as e:
            logger.error(f"Error in plotting parameter effects: {e}")