import os
import traceback
import logging
import networkx as nx
import plotly.graph_objects as go
import random
from bertopic import BERTopic
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.cluster import hierarchy as sch
from typing import List, Dict, Tuple, Union, Optional, Any
from sklearn.decomposition import PCA
import scipy.sparse as sp
from umap import UMAP

# Import global_options in a way that avoids circular imports
try:
    from eCallsAgent.config import global_options as gl
except ImportError:
    # Create a fallback for testing
    class GlobalOptions:
        def __init__(self):
            self.VISUALIZATION_DIR = "visualizations"
            self.FIGURE_WIDTH = 1200
            self.FIGURE_HEIGHT = 800
            self.COLORSCALE = "Viridis"
            self.figures_folder = "figures"
            self.output_folder = "output"
    gl = GlobalOptions()

# Set up logging
logger = logging.getLogger(__name__)

def save_figure(fig, filename: str, n_topics: int = None, scale: int = 1) -> None:
    """Save a plotly figure to a file with consistent naming format.
    
    Args:
        fig: Plotly figure to save
        filename: Base filename (e.g., 'barchart', 'heatmap')
        n_topics: Number of topics in the model
        scale: Scale factor for image (PNG) output
    """
    try:
        # Create figures directory if it doesn't exist
        os.makedirs(gl.figures_folder, exist_ok=True)
        
        # Construct the filename with parameters following topic_modeler.py convention
        if hasattr(gl, 'N_NEIGHBORS') and hasattr(gl, 'N_COMPONENTS') and hasattr(gl, 'MIN_CLUSTER_SIZE'):
            # Format filename with required parameters
            formatted_name = f"{filename}_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}"
            
            # Add n_topics if provided
            if n_topics is not None:
                formatted_name += f"_{n_topics}"
            
            # Add year range if available
            if hasattr(gl, 'YEAR_START') and hasattr(gl, 'YEAR_END'):
                formatted_name += f"_{gl.YEAR_START}_{gl.YEAR_END}"
        else:
            # Fallback if parameters aren't available
            formatted_name = filename
        
        # Full paths for HTML and PNG files
        html_path = os.path.join(gl.figures_folder, f"{formatted_name}.html")
        png_path = os.path.join(gl.figures_folder, f"{formatted_name}.png")
        
        # Save in both formats
        fig.write_html(html_path)
        logger.info(f"Figure saved to {html_path}")
        
        fig.write_image(png_path)
        logger.info(f"Figure saved to {png_path}")
        
    except Exception as e:
        logger.error(f"Error saving figure {filename}: {e}")
        logger.error(traceback.format_exc())

def get_base_config(width=1600, height=900):
    """Get base configuration for visualizations.
    
    Args:
        width: Width of the figure in pixels
        height: Height of the figure in pixels
        
    Returns:
        Dictionary with base configuration settings
    """
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

def save_barchart(topic_model: BERTopic, top_n_topics: int, custom_labels: dict, n_topics: int) -> None:
    """Save barchart visualization.
    
    Args:
        topic_model: Fitted BERTopic model
        top_n_topics: Number of top topics to display
        custom_labels: Dictionary mapping topic IDs to custom labels
        n_topics: Total number of topics in the model
    """
    base_config = get_base_config()
    fig = topic_model.visualize_barchart(top_n_topics=top_n_topics, custom_labels=custom_labels)
    fig.update_layout(**base_config)
    fig.update_traces(
        marker_color='rgb(55, 126, 184)',
        marker_line_color='rgb(25, 96, 154)',
        marker_line_width=1.5,
        opacity=0.8
    )
    save_figure(fig, 'barchart', n_topics=n_topics, scale=1)

def save_hierarchy(topic_model, output_dir=None, filename="hierarchy", orientation="left", topics=None, 
                  top_n_topics=None, custom_labels=False, width=1000, height=1000, n_topics=None):
    """Save hierarchical visualization of topics.
    
    Args:
        topic_model: A fitted BERTopic instance.
        output_dir: Directory to save the visualization. If None, uses gl.figures_folder.
        filename: Base name of the file to save.
        orientation: The orientation of the tree: 'left', 'right', 'top', or 'bottom'
        topics: A selection of topics to visualize
        top_n_topics: Only select the top n most frequent topics
        custom_labels: Whether to use custom topic labels
        width: The width of the figure
        height: The height of the figure
        n_topics: Total number of topics in the model
    
    Returns:
        The path to the saved file or None if no file was saved
    """
    # Use figures_folder if output_dir is not specified
    if output_dir is None:
        output_dir = gl.figures_folder
    
    logger.info(f"Saving hierarchical visualization...")
    
    # Check if hierarchical topics exist
    if not hasattr(topic_model, 'hierarchical_topics_') or \
       (hasattr(topic_model, 'hierarchical_topics_') and 
        ( (isinstance(topic_model.hierarchical_topics_, pd.DataFrame) and topic_model.hierarchical_topics_.empty) or
          (isinstance(topic_model.hierarchical_topics_, dict) and len(topic_model.hierarchical_topics_) == 0) )):
        logger.warning("No hierarchical topics found in the model. Attempting to generate them.")
        
        # Try to generate hierarchical topics
        try:
            # Get documents for each topic
            if hasattr(topic_model, 'get_representative_docs'):
                docs_per_topic = topic_model.get_representative_docs()
            elif hasattr(topic_model, 'topic_representations_'):
                docs_per_topic = pd.DataFrame({
                    'Topic': list(topic_model.topic_representations_.keys()),
                    'Doc_IDs': [docs for docs in topic_model.topic_representations_.values()]
                })
            else:
                logger.warning("Could not find document-topic associations")
                return
            
            # Check if we have enough documents
            if docs_per_topic.empty or len(docs_per_topic) < 2:
                logger.warning("Insufficient documents to generate hierarchical topics")
                
                # Create a placeholder figure with a message
                fig = go.Figure()
                fig.add_annotation(
                    text="No hierarchical topics could be generated due to insufficient data",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
                fig.update_layout(
                    width=width,
                    height=height
                )
                
                # Save the placeholder figure using save_figure for consistent naming
                save_figure(fig, filename, n_topics=n_topics)
                return
            
            # Get representative docs for each topic
            representative_docs = []
            for _, row in docs_per_topic.iterrows():
                topic = row['Topic']
                if topic != -1:  # Skip outlier topic
                    doc_ids = row['Doc_IDs']
                    if len(doc_ids) > 0:
                        representative_docs.append(topic_model._docs[doc_ids[0]])
            
            # Check if we have enough representative documents
            if len(representative_docs) < 2:
                logger.warning("Insufficient representative documents to generate hierarchical topics")
                
                # Create a placeholder figure with a message
                fig = go.Figure()
                fig.add_annotation(
                    text="No hierarchical topics could be generated due to insufficient representative documents",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    showarrow=False,
                    font=dict(size=20)
                )
                fig.update_layout(
                    width=width,
                    height=height
                )
                
                # Save the placeholder figure using save_figure for consistent naming
                save_figure(fig, filename, n_topics=n_topics)
                return
            
            # Ensure we're only using documents that have valid topic assignments
            if len(representative_docs) != len(topics):
                logger.warning(f"Document length ({len(representative_docs)}) doesn't match topics length ({len(topics)}). Adjusting...")
                # If lengths don't match, use only the documents that have corresponding topics
                min_length = min(len(representative_docs), len(topics))
                representative_docs = representative_docs[:min_length]
                topics = topics[:min_length]
            
            # Generate hierarchical topics
            logger.info("Generating hierarchical topics from representative documents")
            topic_model.hierarchical_topics(representative_docs)
            
        except Exception as e:
            logger.error(f"Error generating hierarchical topics: {e}")
            
            # Create a placeholder figure with the error message
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error generating hierarchical topics: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=20)
            )
            fig.update_layout(
                width=width,
                height=height
            )
            
            # Save the placeholder figure using save_figure for consistent naming
            save_figure(fig, filename, n_topics=n_topics)
            return
    
    # Visualize hierarchy
    fig = visualize_hierarchy(
        topic_model=topic_model,
        orientation=orientation,
        topics=topics,
        top_n_topics=top_n_topics,
        custom_labels=custom_labels,
        width=width,
        height=height
    )
    
    # If visualization failed, create a placeholder
    if fig is None:
        logger.warning("Failed to create hierarchical visualization")
        
        # Create a placeholder figure with a message
        fig = go.Figure()
        fig.add_annotation(
            text="No valid hierarchical structure could be created",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=20)
        )
        fig.update_layout(
            width=width,
            height=height
        )
    
    # Save the figure using save_figure for consistent naming
    save_figure(fig, filename, n_topics=n_topics)
    logger.info(f"Saved hierarchical visualization successfully")

def save_heatmap(topic_model: BERTopic, custom_labels: dict, n_topics: int) -> None:
    """Save heatmap visualization.
    
    Args:
        topic_model: Fitted BERTopic model
        custom_labels: Dictionary mapping topic IDs to custom labels
        n_topics: Total number of topics in the model
    """
    base_config = get_base_config()
    fig = topic_model.visualize_heatmap(custom_labels=custom_labels)
    fig.update_layout(**base_config)
    fig.update_layout(
        xaxis={'side': 'bottom'},
        yaxis={'side': 'left'}
    )
    save_figure(fig, 'heatmap', n_topics=n_topics, scale=1)

def save_distance_map(topic_model: BERTopic, custom_labels: dict, n_topics: int) -> None:
    """Save intertopic distance map.
    
    Args:
        topic_model: Fitted BERTopic model
        custom_labels: Dictionary mapping topic IDs to custom labels
        n_topics: Total number of topics in the model
    """
    base_config = get_base_config()
    
    try:
        # Try the standard BERTopic visualization
        fig = topic_model.visualize_topics(custom_labels=custom_labels)
    except Exception as e:
        logger.warning(f"Error using standard BERTopic visualization: {e}")
        
        # Handle the "k >= N" error from UMAP or any other error
        logger.warning("Creating custom intertopic distance visualization")

        # If topic_embeddings_ is high-dimensional, reduce it to 2D
        if hasattr(topic_model, 'topic_embeddings_') and isinstance(topic_model.topic_embeddings_, np.ndarray):
            # Check if we have enough samples
            n_topics_available = topic_model.topic_embeddings_.shape[0]
            
            if topic_model.topic_embeddings_.shape[1] > 2:
                try:
                    # Try UMAP first with adjusted parameters
                    
                    # Ensure parameters are valid for the number of samples
                    n_components = min(2, max(1, n_topics_available - 1))
                    n_neighbors = min(5, max(2, n_topics_available // 2))
                    
                    umap_model = UMAP(
                        n_components=n_components,
                        n_neighbors=n_neighbors,
                        min_dist=0.1,
                        metric='cosine',
                        random_state=42
                    )
                    
                    try:
                        topic_coords = umap_model.fit_transform(topic_model.topic_embeddings_)
                        logger.info("Successfully reduced topic embeddings using UMAP")
                    except Exception as umap_e:
                        logger.warning(f"UMAP reduction failed: {umap_e}. Trying alternative method.")
                        
                        # Try PCA with sklearn or numpy SVD fallback
                        try:
                            try:
                                pca = PCA(n_components=min(2, n_topics_available - 1))
                                topic_coords = pca.fit_transform(topic_model.topic_embeddings_)
                                logger.info("Successfully reduced topic embeddings using PCA")
                            except ImportError:
                                logger.warning("sklearn not available, using numpy SVD")
                                # Use numpy SVD as a simple alternative to PCA
                                data = topic_model.topic_embeddings_
                                # Center the data
                                centered_data = data - np.mean(data, axis=0)
                                # Perform SVD
                                U, s, Vt = np.linalg.svd(centered_data, full_matrices=False)
                                # Take the first n_components
                                n_comp = min(2, n_topics_available - 1)
                                topic_coords = U[:, :n_comp] * s[:n_comp]
                                logger.info("Successfully reduced topic embeddings using numpy SVD")
                        except Exception as dim_e:
                            logger.error(f"Error reducing dimensions: {dim_e}")
                            # Use the first two dimensions as a last resort
                            topic_coords = topic_model.topic_embeddings_[:, :2]
                except Exception as dim_e:
                    logger.error(f"Error reducing topic embeddings: {dim_e}")
                    # Use the first two dimensions as a last resort
                    topic_coords = topic_model.topic_embeddings_[:, :2]
            else:
                topic_coords = topic_model.topic_embeddings_
        else:
            # Create random 2D coordinates for topics
            topic_coords = np.random.rand(len(custom_labels), 2)
            logger.warning("Using random coordinates for topics (no topic_embeddings_ found)")
        
        # Create figure
        fig = go.Figure()
        
        # Add a scatter trace for each topic
        for topic_id, label in custom_labels.items():
            if topic_id >= 0:  # Skip outlier topic
                idx = topic_id
                if idx < len(topic_coords):
                    fig.add_trace(go.Scatter(
                        x=[topic_coords[idx, 0]],
                        y=[topic_coords[idx, 1]],
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
        
        # Update layout
        fig.update_layout(
            title="Intertopic Distance Map (Custom)",
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            template='plotly_white',
            showlegend=False
        )
    
    # Update layout with base config
    fig.update_layout(**base_config)
    fig.update_layout(
        xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
    )
    
    save_figure(fig, 'intertopic_distance_map', n_topics=n_topics, scale=1)

def visualize_hierarchy(topic_model, orientation="left", topics=None, top_n_topics=None, custom_labels=False, width=1000, height=1000, save=False, filename=None):
    """Visualize a hierarchical structure of the topics with improved spacing.
    
    Args:
        topic_model: A fitted BERTopic instance.
        orientation: The orientation of the tree: 'left', 'right', 'top', or 'bottom'
        topics: A selection of topics to visualize
        top_n_topics: Only select the top n most frequent topics
        custom_labels: Whether to use custom topic labels that were defined using 
                      `topic_model.set_topic_labels`.
        width: The width of the figure. Only applies to plotly figures.
        height: The height of the figure. Only applies to plotly figures.
        save: Whether to save the resulting graph
        filename: The name of the file to save. If None, a default name will be used.

    Returns:
        A plotly figure or None if no valid hierarchy could be created
    """
    # Check if hierarchical_topics_ exists and is not empty
    if not hasattr(topic_model, 'hierarchical_topics_'):
        logger.warning("No hierarchical topics found in the model")
        logger.info("To create hierarchical topics, use topic_model.hierarchical_topics(docs)")
        return None
    
    if not hasattr(topic_model, 'hierarchical_topics_') or \
       (hasattr(topic_model, 'hierarchical_topics_') and 
        ( (isinstance(topic_model.hierarchical_topics_, pd.DataFrame) and topic_model.hierarchical_topics_.empty) or
          (isinstance(topic_model.hierarchical_topics_, dict) and len(topic_model.hierarchical_topics_) == 0) )):
        logger.warning("Hierarchical topics dataframe is empty")
        return None
        
    # Select topics based on top_n and topics args
    freq_df = topic_model.get_topic_freq()
    freq_df = freq_df.loc[freq_df.Topic != -1, :]
    if topics is not None:
        topics = list(topics)
    elif top_n_topics is not None:
        topics = sorted(freq_df.Topic.to_list()[:top_n_topics])
    else:
        topics = sorted(freq_df.Topic.to_list())

    # Select topic labels
    if custom_labels:
        topic_labels = {topic: topic_model.custom_labels_[topic + topic_model._outliers]
                       for topic in topics}
    else:
        topic_labels = {topic: " | ".join([word for word, _ in topic_model.get_topic(topic)][:5])
                       for topic in topics}
    
    # Extract hierarchical topics
    hierarchical_topics = topic_model.hierarchical_topics_
    
    # Create graph
    G = nx.DiGraph()
    
    # Check if we have valid hierarchical relationships
    valid_relationships = False
    
    # Initialize variables that were missing
    topic_tree = {}
    all_topics = []
    
    # Add edges
    for row in hierarchical_topics.itertuples():
        parent_topic = row.Parent_ID
        child_topic = row.Child_ID
        
        # Only add topics that are in our selection
        if parent_topic in topics and child_topic in topics:
            G.add_edge(parent_topic, child_topic)
            topic_tree[parent_topic] = topic_tree.get(parent_topic, []) + [child_topic]
            all_topics.extend([parent_topic, child_topic])

    # Check if we have any edges
    if len(G.edges) == 0:
        logger.warning("No valid hierarchical relationships found between topics")
        return None
        
    # Create root node if it doesn't exist
    if len(G.nodes) > 0:
        all_topics = list(set(all_topics))
        
        # Fix: Find root nodes properly
        # A root node is a node that has no predecessors
        root_nodes = [node for node in G.nodes if len(list(G.predecessors(node))) == 0]
        
        if not root_nodes:
            logger.warning("No root nodes found in the graph")
            return None
            
        if len(root_nodes) > 1:
            G.add_node(-2)
            for node in root_nodes:
                G.add_edge(-2, node)
            root_node = -2
        else:
            root_node = root_nodes[0]

        # Create layout for the graph with improved spacing
        try:
            pos = hierarchy_pos(G, root_node, orientation=orientation, 
                               width=1, height=1, 
                               # Increase vertical spacing between nodes
                               vert_gap=0.6,  
                               # Increase horizontal spacing between nodes
                               vert_spacing=2.0,
                               # Increase spacing between branches
                               leaf_vs_root_factor=2.0)
        except Exception as e:
            logger.error(f"Error creating hierarchy layout: {e}")
            return None

        # Create edges
        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines')

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += tuple([x0, x1, None])
            edge_trace['y'] += tuple([y0, y1, None])

        # Create nodes
        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers+text',
            textposition='middle right' if orientation != "right" else 'middle left',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color='lightblue',
                size=10,
                line=dict(width=2)))

        # Add nodes
        for node in G.nodes():
            x, y = pos[node]
            node_trace['x'] += tuple([x])
            node_trace['y'] += tuple([y])
            
            # Add topic labels with increased text size
            if node != -2:
                node_trace['text'] += tuple([topic_labels.get(node, f"Topic {node}")])
            else:
                node_trace['text'] += tuple([""])

        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                      layout=go.Layout(
                          showlegend=False,
                          hovermode='closest',
                          margin=dict(b=20, l=5, r=5, t=40),
                          xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                          width=width,
                          height=height,
                          # Add more padding around the graph
                          plot_bgcolor='rgba(0,0,0,0)',
                          paper_bgcolor='rgba(0,0,0,0)',
                      ))
        
        # Adjust text size and font
        fig.update_traces(
            textfont=dict(size=12, family="Arial, sans-serif"),
            selector=dict(type='scatter', mode='markers+text')
        )
        
        # Increase spacing between nodes by adjusting the layout
        fig.update_layout(
            margin=dict(l=50, r=50, t=50, b=50),
        )

        if save:
            if filename is None:
                filename = "hierarchy.html"
            fig.write_html(filename)
            
        return fig
    
    return None

def hierarchy_pos(G, root=None, orientation="left", width=1., height=1., 
                 vert_gap=0.2, vert_spacing=1, leaf_vs_root_factor=0.5):
    """Position nodes in a hierarchical layout with improved spacing.

    Based on https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209
    
    If the graph is a tree this will return the positions to plot this in a 
    hierarchical layout.
    
    Args:
        G: The graph (must be a tree)
        root: The root node of the tree (will be computed if not set)
        orientation: Direction of the hierarchy. Options: "left", "right", "top", "bottom".
        width: Width of the layout
        height: Height of the layout
        vert_gap: Gap between nodes of the same depth
        vert_spacing: Vertical spacing between nodes
        leaf_vs_root_factor: Factor to increase the spacing between leaf nodes compared to root nodes
        
    Returns:
        A dictionary of positions keyed by node
    """
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., height=1., vert_gap=0.2, vert_spacing=1, 
                      xcenter=0.5, leaf_vs_root_factor=0.5, orientation="left", 
                      root_pos=None, min_dx=0.05, parent=None, parsed_nodes=None):
        """Recursive function to compute positions of nodes in a hierarchical layout."""
        if parsed_nodes is None:
            parsed_nodes = set()
        
        parsed_nodes.add(root)
        neighbors = list(G.neighbors(root))
        
        if parent is not None:
            # Fix: Check if parent is in neighbors before removing
            if parent in neighbors:
                neighbors.remove(parent)
        if len(neighbors) != 0:
            dx = max(min_dx, width / len(neighbors))
            
            # Increase spacing for leaf nodes
            if all(len(list(G.neighbors(n))) == 1 for n in neighbors):
                dx = dx * leaf_vs_root_factor
                
            nextx = xcenter - width / 2 - min_dx / 2
            for neighbor in neighbors:
                if neighbor not in parsed_nodes:
                    nextx += dx
                    root_pos = _hierarchy_pos(G, neighbor, width=dx, height=height, 
                                            vert_gap=vert_gap, vert_spacing=vert_spacing, 
                                            xcenter=nextx, leaf_vs_root_factor=leaf_vs_root_factor,
                                            orientation=orientation, root_pos=root_pos, 
                                            min_dx=min_dx, parent=root, parsed_nodes=parsed_nodes)
        else:
            if root_pos is None:
                root_pos = {root: (xcenter, 0)}
            else:
                root_pos[root] = (xcenter, 0)
                
        if root_pos is None:
            root_pos = {}
            
        # Ensure root is in root_pos
        if root not in root_pos:
            root_pos[root] = (xcenter, 0)
        
        if parent is None:
            level = 0
        else:
            # Fix: Ensure parent is in root_pos before accessing it
            if parent not in root_pos:
                # If parent is not in root_pos, add it with a default position
                root_pos[parent] = (0, 0)
            level = root_pos[parent][1] + vert_spacing
            
        # Update root position with level
        root_pos[root] = (xcenter, level)
            
        return root_pos

    pos = _hierarchy_pos(G, root, width, height, vert_gap, vert_spacing, 
                        leaf_vs_root_factor=leaf_vs_root_factor, orientation=orientation)

    # Adjust positions based on orientation
    if orientation == "left":
        return {u: (-v[1], v[0]) for u, v in pos.items()}
    elif orientation == "right":
        return {u: (v[1], v[0]) for u, v in pos.items()}
    elif orientation == "bottom":
        return {u: (v[0], -v[1]) for u, v in pos.items()}
    else:  # orientation == "top"
        return pos

def save_all_visualizations(topic_model: BERTopic, custom_labels: dict, n_topics: int, top_n_topics: int = 20):
    """Save all visualizations for a topic model.
    
    Args:
        topic_model: Fitted BERTopic model
        custom_labels: Dictionary mapping topic IDs to custom labels
        n_topics: Total number of topics in the model
        top_n_topics: Number of top topics to display in barchart
    """
    try:
        logger.info("Generating and saving visualizations...")
        
        # Create output directory if it doesn't exist
        output_dir = gl.figures_folder
        os.makedirs(output_dir, exist_ok=True)
        
        # Save barchart
        save_barchart(topic_model, top_n_topics, custom_labels, n_topics)
        
        # Save hierarchy with the new function signature
        save_hierarchy(
            topic_model=topic_model,
            output_dir=output_dir,
            filename="hierarchy",
            orientation="right",  # Horizontal orientation for better readability
            topics=None,  # Use all topics
            top_n_topics=top_n_topics,
            custom_labels=bool(custom_labels),  # Use custom labels if provided
            width=1200,
            height=800
        )
        
        # Save heatmap
        save_heatmap(topic_model, custom_labels, n_topics)
        
        # Save distance map
        save_distance_map(topic_model, custom_labels, n_topics)

        # Save document embeddings plot
        save_embedding_visualization(topic_model.topic_embeddings_, topic_model.topics_, custom_labels, n_topics)

        # Save topic clusters plot
        save_topic_clusters_plot(topic_model.topic_embeddings_, topic_model.topics_, custom_labels, n_topics)

        logger.info("All visualizations saved successfully.")
    except Exception as e:
        logger.error(f"Error saving visualizations: {e}")
        logger.error(traceback.format_exc())

def plot_document_embeddings(embeddings, topics, custom_labels=None, sample_size=None, width=1600, height=900, title="Document Embeddings by Topic"):
    """Plot 2D visualization of document embeddings colored by topic.
    
    Args:
        embeddings: 2D embeddings of documents (already dimensionality-reduced)
        topics: Topic assignments for each document
        custom_labels: Dictionary mapping topic IDs to custom labels
        sample_size: Number of documents to sample (for large datasets)
        width: Width of the figure in pixels
        height: Height of the figure in pixels
        title: Title of the plot
        
    Returns:
        A plotly figure object
    """
    try:
        import numpy as np
        import pandas as pd
        import plotly.express as px
        from sklearn.preprocessing import StandardScaler
        
        # Ensure we have numpy array for embeddings
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
        
        # Check if we need to sample
        if sample_size is not None and sample_size < len(embeddings):
            # Sample indices
            indices = np.random.choice(range(len(embeddings)), size=sample_size, replace=False)
            embeddings = embeddings[indices]
            if topics is not None:
                topics = [topics[i] for i in indices]
        
        # Ensure embeddings are 2D
        if embeddings.shape[1] > 2:
            # Reduce to 2D for visualization
            embeddings = reduce_embeddings_for_visualization(embeddings)
        
        # Create a dataframe for plotting
        df = pd.DataFrame({
            'x': embeddings[:, 0],
            'y': embeddings[:, 1],
            'topic': topics if topics is not None else [0] * len(embeddings)
        })
        
        # Ensure custom_labels is a dictionary
        if custom_labels is None:
            custom_labels = {}
        
        if not isinstance(custom_labels, dict):
            # Convert to dictionary if possible
            if isinstance(custom_labels, list):
                try:
                    custom_labels = {i: label for i, label in enumerate(custom_labels)}
                except:
                    logger.warning("Could not convert custom_labels list to dictionary")
                    custom_labels = {}
            else:
                logger.warning(f"custom_labels is not a dictionary (type: {type(custom_labels)})")
                custom_labels = {}
        
        # Create topic labels
        df['topic_label'] = df['topic'].apply(lambda x: custom_labels.get(x, f"Topic {x}"))
        
        # Create hover text
        df['hover_text'] = df.apply(
            lambda row: f"Topic: {row['topic_label']}<br>Topic ID: {row['topic']}", 
            axis=1
        )
        
        # Get colors for topics
        colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
        
        # Create the scatter plot
        fig = px.scatter(
            df, x="x", y="y", 
            color="topic_label", 
            hover_name="hover_text",
            title=title,
            width=width, 
            height=height,
            color_discrete_sequence=colors,
            opacity=0.7
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            legend_title="Topics",
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
        )
        
        # Update marker size and opacity
        fig.update_traces(
            marker=dict(size=5, opacity=0.7),
            selector=dict(mode='markers')
        )
        
        return fig
    
    except Exception as e:
        logger.error(f"Error plotting document embeddings: {e}")
        logger.error(traceback.format_exc())
        
        # Create a simple fallback figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error plotting document embeddings: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            template='plotly_white'
        )
        
        return fig

def reduce_embeddings_for_visualization(embeddings, n_components=3, n_neighbors=15, min_dist=0.1):
    """Reduce embeddings to 2D for visualization, handling potential UMAP errors.
    
    Args:
        embeddings: High-dimensional embeddings
        n_components: Number of dimensions to reduce to (default: 2)
        n_neighbors: Number of neighbors for UMAP (default: 15)
        min_dist: Minimum distance for UMAP (default: 0.1)
        
    Returns:
        2D embeddings for visualization
    """
    import numpy as np
    from umap import UMAP
    import traceback
    
    try:
        # Check if we have enough samples for the requested parameters
        sample_count = embeddings.shape[0]
        
        # If we don't have enough samples, adjust parameters
        if n_components >= sample_count or n_neighbors >= sample_count:
            logger.warning(f"Not enough samples ({sample_count}) for requested UMAP parameters (n_components={n_components}, n_neighbors={n_neighbors})")
            
            # Adjust parameters to avoid k >= N error
            adjusted_n_components = min(n_components, max(1, sample_count - 1))
            adjusted_n_neighbors = min(n_neighbors, max(2, sample_count // 2))
            
            logger.info(f"Adjusted UMAP parameters: n_components={adjusted_n_components}, n_neighbors={adjusted_n_neighbors}")
            
            # Create UMAP model with adjusted parameters
            umap_model = UMAP(
                n_components=adjusted_n_components,
                n_neighbors=adjusted_n_neighbors,
                min_dist=min_dist,
                metric='cosine',
                random_state=42
            )
        else:
            # Use original parameters
            umap_model = UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric='cosine',
                random_state=42
            )
        
        # Try to fit and transform the data
        try:
            reduced_embeddings = umap_model.fit_transform(embeddings)
            return reduced_embeddings
        except TypeError as e:
            # Handle the specific "k >= N" error
            if "k >= N" in str(e):
                logger.warning("UMAP error: k >= N. Converting sparse matrix to dense array.")
                
                # Try using scipy.linalg.eigh with dense array as suggested in the error
                try:
                    # If embeddings is a sparse matrix, convert to dense
                    if sp.issparse(embeddings):
                        dense_embeddings = embeddings.toarray()
                    else:
                        dense_embeddings = embeddings
                    
                    # Try to use PCA as a fallback
                    try:
                        # Try importing sklearn in a way that avoids the "could not be resolved" error
                        sklearn_available = False
                        try:
                            from sklearn.decomposition import PCA
                            sklearn_available = True
                        except ImportError:
                            logger.warning("sklearn.decomposition could not be imported")
                        
                        if sklearn_available:
                            pca = PCA(n_components=min(n_components, sample_count - 1))
                            reduced_embeddings = pca.fit_transform(dense_embeddings)
                            logger.info("Successfully reduced dimensions using sklearn PCA fallback")
                            return reduced_embeddings
                        else:
                            # Use numpy SVD as a simple alternative to PCA
                            logger.warning("sklearn not available, using numpy SVD as fallback")
                            # Center the data
                            centered_data = dense_embeddings - np.mean(dense_embeddings, axis=0)
                            # Perform SVD
                            U, s, Vt = np.linalg.svd(centered_data, full_matrices=False)
                            # Take the first n_components
                            n_comp = min(n_components, sample_count - 1)
                            reduced_embeddings = U[:, :n_comp] * s[:n_comp]
                            logger.info("Successfully reduced dimensions using numpy SVD fallback")
                            return reduced_embeddings
                    except Exception as inner_e:
                        logger.error(f"Error using dimensionality reduction fallback: {inner_e}")
                        logger.error(traceback.format_exc())
                except Exception as inner_e:
                    logger.error(f"Error using dense array approach: {inner_e}")
                    logger.error(traceback.format_exc())
            
            # For other TypeError errors, re-raise
            raise
            
    except Exception as e:
        logger.error(f"Error in dimensionality reduction: {e}")
        logger.error(traceback.format_exc())
        
        # Fall back to PCA or numpy SVD
        try:
            # Try importing sklearn in a way that avoids the "could not be resolved" error
            sklearn_available = False
            try:
                sklearn_available = True
            except ImportError:
                logger.warning("sklearn.decomposition could not be imported")
            
            if sklearn_available:
                logger.warning("Falling back to PCA for dimensionality reduction")
                
                # Ensure n_components is valid for PCA
                sample_count = embeddings.shape[0]
                valid_n_components = min(2, sample_count - 1)
                
                pca = PCA(n_components=valid_n_components)
                reduced_embeddings = pca.fit_transform(embeddings)
                return reduced_embeddings
            else:
                logger.warning("sklearn not available, using numpy SVD as fallback")
                # Use numpy SVD as a simple alternative to PCA
                sample_count = embeddings.shape[0]
                # Center the data
                centered_data = embeddings - np.mean(embeddings, axis=0)
                # Perform SVD
                U, s, Vt = np.linalg.svd(centered_data, full_matrices=False)
                # Take the first n_components
                n_comp = min(2, sample_count - 1)
                reduced_embeddings = U[:, :n_comp] * s[:n_comp]
                return reduced_embeddings
            
        except Exception as pca_e:
            logger.error(f"Dimensionality reduction failed: {pca_e}")
            logger.error(traceback.format_exc())
            
            # If all else fails, return random 2D points as placeholder
            logger.warning("Using random 2D points as fallback")
            return np.random.rand(embeddings.shape[0], 2)

def visualize_embeddings(embeddings, topics=None, custom_labels=None, sample_size=5000, title="Document Embeddings"):
    """Visualize document embeddings in 2D space.
    
    This function handles the entire process from high-dimensional embeddings to 2D visualization.
    
    Args:
        embeddings: Document embeddings (either reduced or original)
        topics: Topic assignments for documents (optional)
        custom_labels: Dictionary mapping of topic IDs to custom labels
        sample_size: Number of documents to sample for visualization
        title: Plot title
        
    Returns:
        A plotly figure
    """
    try:
        # Convert custom_labels to dictionary if it's not already
        if custom_labels is not None and not isinstance(custom_labels, dict):
            logger.warning("custom_labels is not a dictionary, creating a simple mapping")
            if isinstance(custom_labels, list):
                # Try to convert list to dict if possible
                try:
                    custom_labels = {i: label for i, label in enumerate(custom_labels)}
                except:
                    # Fallback to a simple mapping
                    if topics is not None:
                        unique_topics = set(topics)
                        custom_labels = {topic: f"Topic {topic}" for topic in unique_topics}
                    else:
                        custom_labels = {}
            else:
                # Create an empty dict if custom_labels is neither a dict nor a list
                custom_labels = {}
        
        # Create the plot
        fig = plot_document_embeddings(
            embeddings=embeddings,
            topics=topics,
            custom_labels=custom_labels if isinstance(custom_labels, dict) else {},
            sample_size=sample_size,
            title=title
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error visualizing embeddings: {e}")
        logger.error(traceback.format_exc())
        
        # Return a simple figure with error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error visualizing embeddings: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

def save_embedding_visualization(embeddings, topics=None, custom_labels=None, sample_size=5000, title="Document Embeddings"):
    """Save a visualization of document embeddings.
    
    Args:
        embeddings: Document embeddings (n_documents x n_dimensions)
        topics: Topic assignments for each document
        custom_labels: Dictionary mapping topic IDs to custom labels
        sample_size: Number of documents to sample
        title: Title of the visualization
    """
    try:
        # Ensure custom_labels is a dictionary
        if custom_labels is not None and not isinstance(custom_labels, dict):
            logger.warning("custom_labels is not a dictionary, creating a simple mapping")
            if isinstance(custom_labels, list):
                # Try to convert list to dict if possible
                try:
                    custom_labels = {i: label for i, label in enumerate(custom_labels)}
                except:
                    # Fallback to a simple mapping
                    custom_labels = {}
            else:
                # Create an empty dict
                custom_labels = {}
        
        # Reduce dimensionality if needed and create visualization
        fig = visualize_embeddings(
            embeddings=embeddings,
            topics=topics,
            custom_labels=custom_labels,
            sample_size=sample_size,
            title=title
        )
        
        # Save the figure
        save_figure(fig, 'document_embeddings', scale=2)
        
        logger.info(f"Saved document embedding visualization")
    except Exception as e:
        logger.error(f"Error saving embedding visualization: {e}")
        logger.error(traceback.format_exc())

def plot_topic_clusters(topic_model, embeddings=None, sample=5000, hide_document_hover=True, custom_labels=None, width=1600, height=900, title="Topic Clusters"):
    """Plot topic clusters with documents and topic centers.
    
    This function creates a more advanced visualization showing both documents and topic centers.
    
    Args:
        topic_model: Fitted BERTopic model
        embeddings: Document embeddings reduced to 2D (if None, will use topic_model.umap_model)
        sample: Number of documents to sample
        hide_document_hover: Whether to hide document details on hover
        custom_labels: Dictionary mapping topic IDs to custom labels
        width: Width of the figure in pixels
        height: Height of the figure in pixels
        title: Title of the plot
        
    Returns:
        A plotly figure object
    """
    try:
        # Get the figure from BERTopic
        fig = topic_model.visualize_documents(
            embeddings=embeddings,
            sample=sample,
            hide_document_hover=hide_document_hover,
            custom_labels=custom_labels,
            width=width,
            height=height,
            title=title
        )
        
        # Update layout
        fig.update_layout(
            template='plotly_white',
            font=dict(family="Arial, sans-serif", size=14),
            title_font=dict(family="Arial, sans-serif", size=20),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(l=80, r=80, t=100, b=80),
            legend=dict(
                font=dict(size=12),
                borderwidth=1,
                bordercolor='lightgray'
            )
        )
        
        # Update document markers
        fig.update_traces(
            marker=dict(
                size=6,
                opacity=0.7,
                line=dict(width=0.5, color='DarkSlateGrey')
            ),
            selector=dict(mode='markers')
        )
        
        # Update topic center markers
        fig.update_traces(
            marker=dict(
                size=16,
                opacity=1,
                line=dict(width=1.5, color='white')
            ),
            selector=dict(mode='markers+text')
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating topic clusters visualization: {e}")
        logger.error(traceback.format_exc())
        return None

def save_topic_clusters(topic_model, embeddings=None, sample=5000, custom_labels=None):
    """Save topic clusters visualization.
    
    Args:
        topic_model: Fitted BERTopic model
        embeddings: Document embeddings reduced to 2D (if None, will use topic_model.umap_model)
        sample: Number of documents to sample
        custom_labels: Dictionary mapping topic IDs to custom labels
    """
    try:
        # Create the plot
        fig = plot_topic_clusters(
            topic_model=topic_model,
            embeddings=embeddings,
            sample=sample,
            hide_document_hover=True,
            custom_labels=custom_labels,
            width=1600,
            height=900,
            title="Topic Clusters"
        )
        
        if fig is not None:
            # Save the figure
            save_figure(fig, 'topic_clusters', scale=2)
            logger.info("Topic clusters visualization saved successfully.")
    except Exception as e:
        logger.error(f"Error saving topic clusters visualization: {e}")
        logger.error(traceback.format_exc()) 