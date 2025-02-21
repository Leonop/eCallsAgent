"""
Visualization utilities for topic modeling results.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Optional, List, Tuple
import logging
from umap import UMAP

logger = logging.getLogger(__name__)

def create_topic_heatmap(topic_model, custom_labels: Optional[Dict[int, str]] = None, width: int = 1200, height: int = 800) -> go.Figure:
    """
    Create a heatmap visualization showing correlation between topics.
    
    Args:
        topic_model: Trained BERTopic model
        custom_labels: Dictionary mapping topic numbers to custom labels
        width: Width of the figure
        height: Height of the figure
        
    Returns:
        Plotly figure object
    """
    try:
        # Get topic similarity matrix
        topic_sim = topic_model.topic_similarities_
        
        # Get topic info and sort by size
        topic_info = topic_model.get_topic_info()
        topic_info = topic_info[topic_info['Topic'] != -1]  # Exclude outlier topic
        topics = topic_info['Topic'].tolist()
        
        # Create labels
        if custom_labels:
            labels = [custom_labels.get(topic, f"Topic {topic}") for topic in topics]
        else:
            labels = [f"Topic {topic}" for topic in topics]
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=topic_sim,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0,
            text=np.round(topic_sim, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False,
        ))
        
        # Update layout
        fig.update_layout(
            title="Topic Similarity Heatmap",
            width=width,
            height=height,
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'},
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating heatmap: {e}")
        raise

def create_topic_hierarchy(topic_model, custom_labels: Optional[Dict[int, str]] = None, width: int = 1200, height: int = 800) -> go.Figure:
    """
    Create a hierarchical visualization of topics using a treemap.
    
    Args:
        topic_model: Trained BERTopic model
        custom_labels: Dictionary mapping topic numbers to custom labels
        width: Width of the figure
        height: Height of the figure
        
    Returns:
        Plotly figure object
    """
    try:
        # Get topic info and hierarchical structure
        topic_info = topic_model.get_topic_info()
        topic_info = topic_info[topic_info['Topic'] != -1]  # Exclude outlier topic
        
        # Create hierarchical data
        data = []
        for _, row in topic_info.iterrows():
            topic_id = row['Topic']
            if custom_labels:
                label = custom_labels.get(topic_id, f"Topic {topic_id}")
            else:
                label = f"Topic {topic_id}"
            
            # Get top words for the topic
            words, weights = zip(*topic_model.get_topic(topic_id))
            top_words = ", ".join(words[:5])
            
            # Add main topic
            data.append(dict(
                ids=label,
                labels=label,
                parents="",
                values=row['Count'],
                text=top_words,
                hovertemplate="<b>%{label}</b><br>Documents: %{value}<br>Top words: %{text}<extra></extra>"
            ))
            
            # Add subtopics (words)
            for word, weight in zip(words[:5], weights[:5]):
                data.append(dict(
                    ids=f"{label}-{word}",
                    labels=word,
                    parents=label,
                    values=weight * row['Count'],
                    text=f"Weight: {weight:.3f}",
                    hovertemplate="<b>%{label}</b><br>Weight: %{text}<extra></extra>"
                ))
        
        # Create treemap
        fig = go.Figure(go.Treemap(
            ids=[item['ids'] for item in data],
            labels=[item['labels'] for item in data],
            parents=[item['parents'] for item in data],
            values=[item['values'] for item in data],
            text=[item['text'] for item in data],
            hovertemplate=[item['hovertemplate'] for item in data],
            textinfo="label",
            marker=dict(
                colors=[
                    '#1f77b4' if item['parents'] == "" else '#aec7e8'
                    for item in data
                ]
            ),
        ))
        
        # Update layout
        fig.update_layout(
            title="Topic Hierarchy",
            width=width,
            height=height,
            template='plotly_white',
            treemap=dict(
                textfont=dict(size=14),
                tiling=dict(pad=5)
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating hierarchy visualization: {e}")
        raise

def create_intertopic_distance_map(topic_model, custom_labels: Optional[Dict[int, str]] = None, width: int = 1200, height: int = 800) -> go.Figure:
    """
    Create a 2D visualization of topic distances using t-SNE or UMAP coordinates.
    
    Args:
        topic_model: Trained BERTopic model
        custom_labels: Dictionary mapping topic numbers to custom labels
        width: Width of the figure
        height: Height of the figure
        
    Returns:
        Plotly figure object
    """
    try:
        # Get topic info and coordinates
        topic_info = topic_model.get_topic_info()
        topic_info = topic_info[topic_info['Topic'] != -1]  # Exclude outlier topic
        
        # Get topic embeddings and reduce to 2D if needed
        if hasattr(topic_model, 'topic_embeddings_'):
            embeddings = topic_model.topic_embeddings_
            if embeddings.shape[1] > 2:
                from umap import UMAP
                reducer = UMAP(n_components=2, random_state=42)
                coords = reducer.fit_transform(embeddings)
            else:
                coords = embeddings
        else:
            logger.warning("No topic embeddings found. Using random coordinates.")
            coords = np.random.randn(len(topic_info), 2)
        
        # Create labels and sizes
        topics = topic_info['Topic'].tolist()
        if custom_labels:
            labels = [custom_labels.get(topic, f"Topic {topic}") for topic in topics]
        else:
            labels = [f"Topic {topic}" for topic in topics]
        
        sizes = topic_info['Count'].values
        sizes = 50 + (sizes - sizes.min()) / (sizes.max() - sizes.min()) * 150
        
        # Create scatter plot
        fig = go.Figure(data=go.Scatter(
            x=coords[:, 0],
            y=coords[:, 1],
            mode='markers+text',
            text=labels,
            marker=dict(
                size=sizes,
                color=topics,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Topic Number")
            ),
            textfont=dict(size=10),
            hovertemplate="<b>%{text}</b><br>Documents: %{marker.size}<extra></extra>"
        ))
        
        # Update layout
        fig.update_layout(
            title="Intertopic Distance Map",
            width=width,
            height=height,
            template='plotly_white',
            showlegend=False,
            xaxis=dict(
                showgrid=True,
                zeroline=False,
                showticklabels=False,
                title=""
            ),
            yaxis=dict(
                showgrid=True,
                zeroline=False,
                showticklabels=False,
                title=""
            )
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating intertopic distance map: {e}")
        raise

def create_umap_topic_clusters(embeddings: np.ndarray, topics: List[int], topic_labels: Dict[int, str] = None) -> go.Figure:
    """Create UMAP visualization of topic clusters.
    
    Args:
        embeddings: Document embeddings array
        topics: List of topic assignments
        topic_labels: Optional dictionary mapping topic IDs to labels
    
    Returns:
        Plotly figure object
    """
    try:
        # Reduce to 2D for visualization
        umap_model = UMAP(n_components=2, random_state=42)
        reduced_embeddings = umap_model.fit_transform(embeddings)
        
        # Create scatter plot
        fig = go.Figure()
        
        # Plot points for each topic
        unique_topics = sorted(list(set(topics)))
        for topic in unique_topics:
            if topic == -1:  # Skip outliers
                continue
                
            mask = np.array(topics) == topic
            topic_label = topic_labels.get(topic, f"Topic {topic}") if topic_labels else f"Topic {topic}"
            
            fig.add_trace(go.Scatter(
                x=reduced_embeddings[mask, 0],
                y=reduced_embeddings[mask, 1],
                mode='markers',
                name=topic_label,
                marker=dict(size=5),
                hovertemplate=f"{topic_label}<extra></extra>"
            ))
            
        # Update layout
        fig.update_layout(
            title="Topic Clusters Visualization",
            xaxis_title="UMAP Dimension 1",
            yaxis_title="UMAP Dimension 2",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating UMAP visualization: {e}")
        raise

def create_topic_similarity_heatmap(topic_similarities: np.ndarray, topic_labels: Dict[int, str] = None) -> go.Figure:
    """Create heatmap of topic similarities."""
    # Implementation for similarity heatmap
    pass

def create_topic_wordcloud(topic_words: Dict[str, float]) -> go.Figure:
    """Create word cloud visualization for a topic."""
    # Implementation for word cloud
    pass
