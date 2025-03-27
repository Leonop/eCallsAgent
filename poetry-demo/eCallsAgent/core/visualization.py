import os
import traceback
import logging
import networkx as nx
import plotly.graph_objects as go
import random
import numpy as np
import pandas as pd
import plotly.express as px
from scipy.cluster import hierarchy as sch
from sklearn.decomposition import PCA
import scipy.sparse as sp
from umap import UMAP
from bertopic import BERTopic
from typing import List, Dict, Optional, Any
import inspect
import pickle

# Global options – if available
try:
    from eCallsAgent.config import global_options as gl
except ImportError:
    class GlobalOptions:
        def __init__(self):
            self.VISUALIZATION_DIR = "visualizations"
            self.FIGURE_WIDTH = 1200
            self.FIGURE_HEIGHT = 800
            self.COLORSCALE = "Viridis"
            self.figures_folder = "figures"
            self.output_folder = "output"
            self.N_NEIGHBORS = [15]
            self.N_COMPONENTS = [5]
            self.MIN_CLUSTER_SIZE = [10]
            self.YEAR_START = 2011
            self.YEAR_END = 2014
    gl = GlobalOptions()

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class TopicVis:
    def __init__(self, topic_model: BERTopic, custom_labels: Optional[Dict] = None, n_topics: Optional[int] = None):
        """
        Initialize the TopicVis instance.
        Args:
            topic_model: A fitted BERTopic instance.
            custom_labels: (Optional) Mapping of topic IDs to custom labels.
            n_topics: (Optional) Total number of topics.
        """
        self.topic_model = topic_model
        self.custom_labels = custom_labels if custom_labels is not None else {}
        if n_topics is not None:
            self.n_topics = n_topics
        else:
            freq_df = self.topic_model.get_topic_freq()
            self.n_topics = len(freq_df.loc[freq_df.Topic != -1])
        self.figures_folder = getattr(gl, 'figures_folder', "figures")
        self.logger = logger

    def get_base_config(self, width=1600, height=900) -> Dict:
        """Return base configuration for Plotly figures."""
        return {
            'width': width,
            'height': height,
            'template': 'plotly_white',
            'font': dict(family="Arial, sans-serif", size=14),
            'title_font': dict(family="Arial, sans-serif", size=20),
            'plot_bgcolor': 'white',
            'paper_bgcolor': 'white',
            'margin': dict(l=80, r=80, t=100, b=80),
        }

    def save_figure(self, fig: go.Figure, filename: str, scale: int = 1) -> None:
        """Save a Plotly figure as HTML and PNG."""
        try:
            os.makedirs(self.figures_folder, exist_ok=True)
            formatted_name = f"{filename}_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}"
            if self.n_topics is not None:
                formatted_name += f"_{self.n_topics}"
            if hasattr(gl, 'YEAR_START') and hasattr(gl, 'YEAR_END'):
                formatted_name += f"_{gl.YEAR_START}_{gl.YEAR_END}"
            html_path = os.path.join(self.figures_folder, f"{formatted_name}.html")
            png_path = os.path.join(self.figures_folder, f"{formatted_name}.png")
            fig.write_html(html_path)
            self.logger.info(f"Figure saved to {html_path}")
            fig.write_image(png_path)
            self.logger.info(f"Figure saved to {png_path}")
        except Exception as e:
            self.logger.error(f"Error saving figure {filename}: {e}")
            self.logger.error(traceback.format_exc())

    def save_barchart(self, top_n_topics: int) -> None:
        """Save a barchart visualization."""
        base_config = self.get_base_config()
        fig = self.topic_model.visualize_barchart(top_n_topics=top_n_topics, custom_labels=self.custom_labels)
        fig.update_layout(**base_config)
        fig.update_traces(
            marker_color='rgb(55, 126, 184)',
            marker_line_color='rgb(25, 96, 154)',
            marker_line_width=1.5,
            opacity=0.8
        )
        self.save_figure(fig, 'barchart')

    def save_heatmap(self) -> None:
        """Save a heatmap visualization."""
        base_config = self.get_base_config()
        fig = self.topic_model.visualize_heatmap(custom_labels=self.custom_labels)
        fig.update_layout(**base_config)
        fig.update_layout(xaxis={'side': 'bottom'}, yaxis={'side': 'left'})
        self.save_figure(fig, 'heatmap')

    def save_distance_map(self) -> None:
        """Save an intertopic distance map."""
        base_config = self.get_base_config()
        try:
            fig = self.topic_model.visualize_topics(custom_labels=self.custom_labels)
        except Exception as e:
            self.logger.warning(f"Standard BERTopic visualization error: {e}")
            self.logger.info("Creating custom intertopic distance visualization")
            if hasattr(self.topic_model, 'topic_embeddings_') and isinstance(self.topic_model.topic_embeddings_, np.ndarray):
                if self.topic_model.topic_embeddings_.shape[1] > 2:
                    try:
                        topic_coords = self.reduce_embeddings(self.topic_model.topic_embeddings_, n_components=2)
                        self.logger.info("Reduced topic embeddings using UMAP")
                    except Exception as dim_e:
                        self.logger.error(f"Error reducing dimensions: {dim_e}")
                        topic_coords = self.topic_model.topic_embeddings_[:, :2]
                else:
                    topic_coords = self.topic_model.topic_embeddings_
            else:
                topic_coords = np.random.rand(len(self.custom_labels), 2)
                self.logger.warning("Using random coordinates (no topic_embeddings_ found)")
            fig = go.Figure()
            for topic_id, label in self.custom_labels.items():
                if topic_id >= 0 and topic_id < len(topic_coords):
                    fig.add_trace(go.Scatter(
                        x=[topic_coords[topic_id, 0]],
                        y=[topic_coords[topic_id, 1]],
                        mode='markers+text',
                        marker=dict(
                            size=15,
                            color=f'hsl({(topic_id * 50) % 360}, 80%, 50%)',
                            line=dict(width=1, color='white')
                        ),
                        text=[label],
                        textposition="top center",
                        name=label
                    ))
            fig.update_layout(
                title="Intertopic Distance Map (Custom)",
                xaxis_title="Dimension 1",
                yaxis_title="Dimension 2",
                template='plotly_white',
                showlegend=False
            )
        fig.update_layout(**base_config)
        fig.update_layout(xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray'),
                          yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray'))
        self.save_figure(fig, 'intertopic_distance_map')

    def reduce_embeddings(self, embeddings: np.ndarray, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
        """Reduce high-dimensional embeddings to 2D using UMAP."""
        try:
            sample_count = embeddings.shape[0]
            if n_components >= sample_count or n_neighbors >= sample_count:
                self.logger.warning(f"Not enough samples ({sample_count}) for UMAP. Adjusting parameters.")
                adjusted_n_components = min(n_components, max(1, sample_count - 1))
                adjusted_n_neighbors = min(n_neighbors, max(2, sample_count // 2))
                self.logger.info(f"Adjusted UMAP: n_components={adjusted_n_components}, n_neighbors={adjusted_n_neighbors}")
                umap_model = UMAP(n_components=adjusted_n_components, n_neighbors=adjusted_n_neighbors,
                                  min_dist=min_dist, metric='cosine', random_state=42)
            else:
                umap_model = UMAP(n_components=n_components, n_neighbors=n_neighbors,
                                  min_dist=min_dist, metric='cosine', random_state=42)
            return umap_model.fit_transform(embeddings)
        except Exception as e:
            self.logger.error(f"Error reducing embeddings: {e}")
            self.logger.error(traceback.format_exc())
            return np.random.rand(embeddings.shape[0], 2)

    def plot_document_embeddings(self, embeddings: Any, topics: List[int],
                                 sample_size: Optional[int] = 5000,
                                 title: str = "Document Embeddings") -> go.Figure:
        """Plot document embeddings in 2D, with fallback if embeddings are unsized."""
        try:
            if embeddings is None or not hasattr(embeddings, '__len__'):
                self.logger.error("Embeddings object is unsized. Returning fallback visualization.")
                fig = go.Figure()
                fig.add_annotation(
                    text="Embeddings object is unsized",
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False, font=dict(size=14, color="red")
                )
                fig.update_layout(title=title, width=1600, height=900, template='plotly_white')
                return fig
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            n_docs = embeddings.shape[0]
            if sample_size and sample_size < n_docs:
                indices = np.random.choice(range(n_docs), size=sample_size, replace=False)
                embeddings = embeddings[indices]
                topics = [topics[i] for i in indices] if topics is not None else [0] * sample_size
            if embeddings.shape[1] > 2:
                embeddings = self.reduce_embeddings(embeddings, n_components=2)
            df = pd.DataFrame({
                'x': embeddings[:, 0],
                'y': embeddings[:, 1],
                'topic': topics if topics is not None else [0] * len(embeddings)
            })
            self.custom_labels = self.custom_labels if isinstance(self.custom_labels, dict) else {}
            df['topic_label'] = df['topic'].apply(lambda x: self.custom_labels.get(x, f"Topic {x}"))
            df['hover_text'] = df.apply(lambda row: f"Topic: {row['topic_label']}<br>Topic ID: {row['topic']}", axis=1)
            colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
            fig = px.scatter(df, x="x", y="y", color="topic_label", hover_name="hover_text",
                             title=title, width=1600, height=900,
                             color_discrete_sequence=colors, opacity=0.7)
            fig.update_layout(xaxis_title="Dimension 1", yaxis_title="Dimension 2",
                              legend_title="Topics", template="plotly_white",
                              margin=dict(l=40, r=40, t=50, b=40))
            fig.update_traces(marker=dict(size=5, opacity=0.7), selector=dict(mode='markers'))
            return fig
        except Exception as e:
            self.logger.error(f"Error plotting document embeddings: {e}")
            self.logger.error(traceback.format_exc())
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error plotting document embeddings: {str(e)}",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False, font=dict(size=14, color="red")
            )
            fig.update_layout(title=title, width=1600, height=900, template='plotly_white')
            return fig

    def save_embedding_visualization(self, embeddings: Any, topics: List[int],
                                     sample_size: Optional[int] = 5000,
                                     title: str = "Document Embeddings") -> None:
        """Save the document embedding visualization."""
        try:
            fig = self.plot_document_embeddings(embeddings, topics, sample_size, title)
            self.save_figure(fig, 'document_embeddings', scale=2)
            self.logger.info("Saved document embedding visualization")
        except Exception as e:
            self.logger.error(f"Error saving embedding visualization: {e}")
            self.logger.error(traceback.format_exc())

    def save_embedding_document(self, docs_path=None):
        """
        Save the document embedding visualization of a BERTopic model to a file.
        
        Parameters:
            docs_path (str, optional): Path to separately stored documents if not in model.
            
        The function calls the built-in visualize_documents method to generate
        the interactive Plotly figure and then saves it using the save_figure method.
        """
        try:
             # Retrieve the documents from the model (use _docs if available)
            docs = self.topic_model._docs if hasattr(self.topic_model, "_docs") else []
            if not docs:
                raise ValueError("No documents found in the model (expected in _docs attribute).")
            
            # Generate the document embedding visualization
            self.logger.info(f"Generating document embedding visualization with {len(docs)} documents")
            fig = self.topic_model.visualize_documents(docs=docs)
            self.save_figure(fig, 'document_embeddings', scale=2)
            self.logger.info("Document embedding visualization saved successfully")
            
            # Return success status
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving document embedding visualization: {e}")
            self.logger.error(traceback.format_exc())
            
            # Create a fallback visualization
            try:
                self.logger.info("Creating fallback visualization without documents")
                fig = go.Figure()
                fig.add_annotation(
                    text="Document embedding visualization unavailable:<br>" + str(e),
                    xref="paper", yref="paper", x=0.5, y=0.5,
                    showarrow=False, font=dict(size=14, color="red")
                )
                fig.update_layout(title="Document Embeddings (Unavailable)", width=1600, height=900)
                self.save_figure(fig, 'document_embeddings_fallback', scale=1)
                self.logger.info("Fallback visualization saved")
            except Exception as fallback_error:
                self.logger.error(f"Error creating fallback visualization: {fallback_error}")
            
            # Return failure status
            return False


    def save_hierarchy(self, topic_model: BERTopic, custom_labels: dict, base_config: dict) -> None:
        """Save hierarchy visualization."""
        fig = topic_model.visualize_hierarchy(
            custom_labels=custom_labels,
            width=base_config['width'],
            height=base_config['height'],
            color_threshold=1.0,
            orientation='left'
        )
        
        # Update layout with specific hierarchy settings
        hierarchy_config = base_config.copy()
        hierarchy_config.update({
            'showlegend': False,
            'margin': dict(l=200, r=200, t=100, b=100)
        })
        fig.update_layout(**hierarchy_config)
        
        # Update trace properties
        fig.update_traces(
            textfont=dict(size=10, family='Arial')
        )
        
        self.save_figure(fig, 'hierarchy', scale=2)

        
    def save_all_visualizations(self, top_n_topics: int = 20, include_doc_vis: bool = False) -> None:
        """Generate and save all visualizations for the topic model.
        
        Args:
            top_n_topics: Maximum number of topics to include in visualizations
            include_doc_vis: Whether to try generating document embedding visualization
        """
        try:
            self.logger.info("Generating and saving all visualizations...")
            os.makedirs(self.figures_folder, exist_ok=True)
            
            # Core visualizations that don't require documents
            self.save_barchart(top_n_topics)
            self.save_hierarchy(self.topic_model, self.custom_labels, self.get_base_config())
            self.save_heatmap()
            self.save_distance_map()
            
            # Optionally try document visualizations
            if include_doc_vis:
                try:
                    self.logger.info("Attempting document embedding visualization...")
                    success = self.save_embedding_document()
                    if success:
                        self.logger.info("Document embedding visualization completed successfully")
                    else:
                        self.logger.warning("Document embedding visualization could not be generated")
                except Exception as e:
                    self.logger.warning(f"Skipping document visualization due to error: {e}")
            
            self.logger.info("All visualizations saved successfully.")
        except Exception as e:
            self.logger.error(f"Error saving visualizations: {e}")
            self.logger.error(traceback.format_exc())
