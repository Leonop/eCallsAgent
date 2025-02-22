"""
Module for topic modeling using BERTopic with optimizations for large datasets.
"""

import os
import time
import logging
import torch
import numpy as np
import traceback
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from hdbscan import HDBSCAN
from umap import UMAP
from bertopic import BERTopic
from openai import OpenAI
import json
import pickle
import gc
import pandas as pd
import csv
from eCallsAgent.core.embedding_generator import EmbeddingGenerator
from eCallsAgent.config import global_options as gl # global settings
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class TopicModeler:
    """Wraps topic modeling components and handles training and saving of results."""
    def __init__(self, device: str):
        self.device = device
        # Create necessary directories at initialization
        self.output_dirs = {
            'temp': os.path.join(gl.output_folder, 'temp'),
            'models': os.path.join(gl.output_folder, 'models'),
            'embeddings': os.path.join(gl.output_folder, 'temp', 'embeddings'),
            'figures': os.path.join(gl.output_folder, 'figures')
        }
        
        # Create all required directories
        for dir_path in self.output_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Use EmbeddingGenerator to set base batch size
        embedding_gen = EmbeddingGenerator(device)
        self.base_batch_size = embedding_gen.base_batch_size

        self.umap_model = UMAP(
            n_neighbors=gl.N_NEIGHBORS[0],
            n_components=gl.N_COMPONENTS[0],
            min_dist=0.0,
            metric='cosine',
            random_state=42,
            verbose=False,
            low_memory=True,
            n_jobs=-1
        )
        self.hdbscan_model = HDBSCAN(
            min_samples=gl.MIN_SAMPLES[0],
            min_cluster_size=gl.MIN_CLUSTER_SIZE[0],
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True,
            algorithm='best',
            core_dist_n_jobs=-1
        )

        self.topic_model = BERTopic(
            umap_model=self.umap_model,
            hdbscan_model=self.hdbscan_model,
            embedding_model=None,
            min_topic_size=gl.MIN_CLUSTER_SIZE[0],
            calculate_probabilities=False,
            seed_topic_list=gl.SEED_TOPICS,
            verbose=False,
            vectorizer_model=TfidfVectorizer(
                max_df=gl.MAX_DF[0],  # Use 95% as relative value
                min_df=gl.MIN_DF[0],  # Use 0.03% as relative value (will be ~3 docs in 10k dataset)
                stop_words='english',
                ngram_range=(1, 3),
                use_idf=True,
                smooth_idf=True
            ),
            top_n_words=25,  # Number of words per topic
            n_gram_range=(1, 3),  # n-gram range for topic representation
        )
        self.n_topics = None

    def _setup_openai(self):
        """Setup OpenAI client with error handling."""
        try:
            api_key_path = os.path.join(os.getcwd(), 'data', 'OPENAI_API_KEY.txt')
            with open(api_key_path, 'r') as f:
                self.client = OpenAI(api_key=f.read().strip())
        except Exception as e:
            logger.error(f"Error initializing OpenAI: {e}")
            raise

    def _create_chunk_model(self, chunk_size: int) -> BERTopic:
        """Create optimized model for chunk processing"""
        return BERTopic(
            umap_model=UMAP(
                n_neighbors=min(chunk_size-1, gl.N_NEIGHBORS[0]),
                n_components=gl.N_COMPONENTS[0],
                min_dist=0.0,
                metric='cosine',
                random_state=42,
                verbose=False,
                low_memory=True,
                n_jobs=-1
            ),
            hdbscan_model=HDBSCAN(
                min_samples=gl.MIN_SAMPLES[0],
                min_cluster_size=gl.MIN_CLUSTER_SIZE[0],
                metric='euclidean',
                cluster_selection_method='eom',
                prediction_data=True,
                core_dist_n_jobs=-1
            ),
            embedding_model=None,
            calculate_probabilities=False,
            verbose=False
        )

    def _store_representatives(self, chunk_model, chunk_docs, chunk_embeddings, topic_representatives):
        """Store representative documents for each topic in the chunk."""
        try:
            # Convert embeddings to numpy arrays to save memory
            topic_docs = chunk_model.get_representative_docs()
            topic_info = chunk_model.get_topic_info()

            # Precompute a mapping from document content to its index in chunk_docs
            doc_to_index = {doc: idx for idx, doc in enumerate(chunk_docs)}
                        
            # Store representatives for each topic
            for topic_id in topic_info['Topic']:
                if topic_id != -1:  # Skip outlier topic
                    docs_for_topic = topic_docs.get(topic_id, [])[:2]  # Only take top 2 directly
                    doc_indices = []
                    
                    # More efficient index finding
                    for doc in docs_for_topic:
                        idx = doc_to_index.get(doc)
                        if idx is not None:
                            doc_indices.append(idx)
                    
                    if doc_indices:
                        topic_key = f"topic_{len(topic_representatives)}"
                        # Convert numpy arrays to lists for JSON serialization
                        topic_representatives[topic_key] = {
                            'docs': [chunk_docs[idx] for idx in doc_indices],
                            'embeddings': [chunk_embeddings[idx].tolist() for idx in doc_indices]  # Convert to list
                        }
            
            # Save to JSON file with proper conversion
            temp_dir = os.path.join(gl.output_folder, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            json_path = os.path.join(temp_dir, f'{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{self.n_topics}_{gl.YEAR_START}_{gl.YEAR_END}_topic_representatives.json')
            with open(json_path, 'w') as f:
                json.dump(topic_representatives, f, ensure_ascii=False)
            
            logger.info(f"Stored representatives for {len(topic_representatives)} topics")
            
        except Exception as e:
            logger.error(f"Error storing representatives: {e}")
            raise

    def _transform_documents_gpu(self, docs: list, embeddings: np.ndarray, chunk_size: int = 5000) -> list:
        """Transform documents using GPU acceleration and chunking."""
        logger.info(f"Transforming {len(docs)} documents with chunk size {chunk_size}")
        all_topics = []
        
        # Process all documents in chunks
        for i in tqdm(range(0, len(docs), chunk_size), desc="Processing documents"):
            chunk_end = min(i + chunk_size, len(docs))
            chunk_embeddings = embeddings[i:chunk_end]
            try:
                # Transform embeddings using UMAP on CPU
                umap_embeddings = self.topic_model.umap_model.transform(chunk_embeddings)
                # Fit HDBSCAN on the chunk and get clusters
                self.topic_model.hdbscan_model.fit(umap_embeddings)
                labels = self.topic_model.hdbscan_model.labels_
                all_topics.extend(labels)
            except Exception as e:
                logger.error(f"Error processing chunk {i}-{chunk_end}: {str(e)}")
                # In case of error, assign -1 (noise) to all documents in the chunk
                all_topics.extend([-1] * (chunk_end - i))
        
        logger.info(f"Transformed {len(all_topics)} documents into topics")
        return all_topics


    def train_topic_model(self, docs: list, embeddings: np.ndarray) -> BERTopic:
        """Train BERTopic model on documents and embeddings.
        
        Args:
            docs: List of preprocessed documents
            embeddings: Document embeddings array
            
        Returns:
            Trained BERTopic model
        """
        try:
            logger.info("Starting topic model training")
            
            # Phase 1: Initial Processing
            logger.info(f"Processing Started at {time.time()}")
            docs, embeddings = self._preprocess_data(docs, embeddings)
            topic_representatives = self._process_chunks(docs, embeddings)
                
            # Phase 2: Topic Distillation
            all_rep_docs, all_rep_embeddings = self._distill_topics(topic_representatives, docs, embeddings)
            
            # Phase 3: Final Model Training
            self._train_final_model(all_rep_docs, all_rep_embeddings)
            
            # Phase 4: Document Mapping
            self._map_documents(docs, embeddings)
            
            # Store number of topics
            self.n_topics = len(set(self.topic_model.topics_)) - 1  # Exclude -1 (outlier topic)
            logger.info(f"Model trained successfully. Found {self.n_topics} topics")
            logger.info(f"Processing Completed at {time.time()}")
            
            # Clean up
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return self.topic_model
            
        except Exception as e:
            logger.error(f"Error in train_topic_model: {e}")
            logger.error(traceback.format_exc())
            raise

    
    def _preprocess_data(self, docs: list, embeddings: np.ndarray) -> tuple:
        """Preprocess input data with memory mapping."""
        docs = [str(doc) for doc in docs]
        
        # Create memory-mapped array for embeddings
        mmap_path = os.path.join(gl.output_folder, 'temp_embeddings.mmap')
        shape = embeddings.shape
        mmap_embeddings = np.memmap(mmap_path, dtype=np.float32, mode='w+', shape=shape)
        mmap_embeddings[:] = embeddings[:]
        
        return docs, mmap_embeddings

    def _process_chunks(self, docs: list, embeddings: np.ndarray) -> dict:
        """Process documents in chunks to identify representative examples."""
        try:
            topic_representatives = {}
            chunk_size = gl.DOCS_PER_RUN
            
            for i in range(0, len(docs), chunk_size):
                chunk_end = min(i + chunk_size, len(docs))
                chunk_docs = docs[i:chunk_end]
                chunk_embeddings = embeddings[i:chunk_end]
                
                # Process chunk
                chunk_model = self._create_chunk_model(len(chunk_docs))
                chunk_model.fit(chunk_docs, chunk_embeddings)
                
                # Store representatives
                self._store_representatives(chunk_model, chunk_docs, chunk_embeddings, topic_representatives)
                
                # Clear memory
                del chunk_model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Save final representatives
            representatives_path = os.path.join(self.output_dirs['models'], 'topic_representatives.json')
            with open(representatives_path, 'w', encoding='utf-8') as f:
                json.dump(topic_representatives, f, ensure_ascii=False, indent=2)
            
            return topic_representatives
            
        except Exception as e:
            logger.error(f"Error in _process_chunks: {e}")
            logger.error(traceback.format_exc())
            raise

    def _distill_topics(self, topic_representatives: dict, docs: list, embeddings: np.ndarray) -> tuple:
            """Distill topics from representatives."""
            logger.info("Phase 2: Topic distillation and refinement")
            
            # Calculate adaptive number of representatives
            total_docs = len(docs)
            max_representatives = self._calculate_max_representatives(total_docs)
            
            # Calculate target topics and docs per topic
            target_topics = min(gl.NR_TOPICS[0], len(topic_representatives))
            min_docs_per_topic = gl.MIN_DOCS_PER_TOPIC
            max_docs_per_topic = max(gl.MAX_DOCS_PER_TOPIC, max_representatives // target_topics)
            
            # Sort and collect representative documents
            sorted_topics = sorted(
                topic_representatives.items(),
                key=lambda x: len(x[1]['docs']),
                reverse=True
            )[:target_topics]
            
            all_rep_docs, all_rep_embeddings, topic_counts = self._collect_representatives(
                sorted_topics, max_representatives, min_docs_per_topic, max_docs_per_topic
            )
            
            # Fallback if insufficient representatives
            if len(all_rep_docs) < min_docs_per_topic * 10:
                logger.warning(f"Insufficient representative documents ({len(all_rep_docs)}). Using all documents.")
                all_rep_docs = docs[:max_representatives]
                all_rep_embeddings = embeddings[:max_representatives]
            
            return all_rep_docs, all_rep_embeddings

    def _train_final_model(self, all_rep_docs: list, all_rep_embeddings: list) -> None:
        """Train the final topic model."""
        # Configure model parameters
        self.topic_model.min_topic_size = gl.OPTIMAL_DOCS_PER_TOPIC
        target_topics = gl.NR_TOPICS[0] if gl.NR_TOPICS[0] > 0 else len(all_rep_docs) // 3
        self.topic_model.nr_topics = target_topics
        
        # Update HDBSCAN parameters
        self.topic_model.hdbscan_model.min_cluster_size = gl.OPTIMAL_DOCS_PER_TOPIC
        self.topic_model.hdbscan_model.min_samples = max(5, gl.OPTIMAL_DOCS_PER_TOPIC // 4)
        
        # Fit model and reduce topics if needed
        logger.info(f"Fitting final topic model with {len(all_rep_docs)} documents")
        self.topic_model.fit(all_rep_docs, all_rep_embeddings)
        self._reduce_topics(all_rep_docs)
        
        # Update topic count
        self.n_topics = len(self.topic_model.get_topics()) - 1

    def _map_documents(self, docs: list, embeddings: np.ndarray) -> None:
        """Map all documents to topics using GPU acceleration."""
        logger.info("Transforming all documents with GPU acceleration")
        self.final_topics = self._transform_documents_gpu(docs, embeddings, chunk_size=8000)

    def _calculate_max_representatives(self, total_docs: int) -> int:
        """Calculate maximum number of representative documents."""
        if total_docs < gl.MAX_ADAPTIVE_REPRESENTATIVES:
            return total_docs
        return min(
            max(2000, int(np.sqrt(total_docs) * 10)),
            gl.MAX_ADAPTIVE_REPRESENTATIVES
        )

    def _collect_representatives(self, sorted_topics: list, max_representatives: int,
                               min_docs_per_topic: int, max_docs_per_topic: int) -> tuple:
        """Collect representative documents from topics."""
        all_rep_docs = []
        all_rep_embeddings = []
        topic_counts = {}
        
        for topic_key, topic_data in sorted_topics:
            docs_to_take = min(
                max_docs_per_topic,
                len(topic_data['docs']),
                max_representatives - len(all_rep_docs)
            )
            
            if docs_to_take >= min_docs_per_topic:
                all_rep_docs.extend(topic_data['docs'][:docs_to_take])
                all_rep_embeddings.extend(topic_data['embeddings'][:docs_to_take])
                topic_counts[topic_key] = docs_to_take
            
            if len(all_rep_docs) >= max_representatives:
                break
        
        return all_rep_docs, all_rep_embeddings, topic_counts

    def _reduce_topics(self, all_rep_docs: list) -> None:
        """Reduce number of topics if needed."""
        target_topics = gl.NR_TOPICS[0]
        current_topics = len(self.topic_model.get_topics()) - 1
        
        if current_topics >= target_topics:
            logger.info(f"Reducing topics from {current_topics} to {target_topics}")
            try:
                self.topic_model = self.topic_model.reduce_topics(
                    docs=all_rep_docs,
                    nr_topics=target_topics
                )
            except Exception as e:
                logger.error(f"Error reducing topics: {e}")
                logger.warning("Keeping original topic distribution")
    
    def _cleanup_temp_files(self):
        """Clean up temporary memory-mapped files."""
        try:
            temp_files = [
                os.path.join(gl.output_folder, 'temp', 'embeddings', 'temp_embeddings.mmap'),
                os.path.join(gl.output_folder, 'temp', 'topic_keywords_checkpoint.pkl')
            ]
            for file_path in temp_files:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.info(f"Cleaned up temporary file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to remove temporary file {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")

    def _reduce_topics(self, all_rep_docs: list) -> None:
        """Reduce number of topics if needed."""
        target_topics = gl.NR_TOPICS[0]
        current_topics = len(self.topic_model.get_topics()) - 1
        
        if current_topics >= target_topics:
            logger.info(f"Reducing topics from {current_topics} to {target_topics}")
            try:
                self.topic_model = self.topic_model.reduce_topics(
                    docs=all_rep_docs,
                    nr_topics=target_topics
                )
            except Exception as e:
                logger.error(f"Error reducing topics: {e}")
                logger.warning("Keeping original topic distribution")

    def generate_topic_label(self, keywords: list, docs: list = None) -> str:
        """Generate a concise topic label via GPT."""
        try:
            # Add caching to avoid redundant API calls
            cache_key = '_'.join(keywords[:5])  # Create a unique key
            cache_file = os.path.join(gl.output_folder, 'temp', 'topic_label_cache.json')
            
            # Check cache first
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
                    if cache_key in cache:
                        return cache[cache_key]

            if not keywords:
                return "No Topic", ""
                
            top_keywords = [k.replace('_', ' ') for k in keywords[:5]]
            doc_context = ""
            if docs:
                doc_context = "\nExample discussions:\n" + "\n".join(docs[:2])
            prompt = f"""Generate a concise business topic label (1-2 words) and a subtopic label (2-3 words) based on the provided earnings call keywords.

                ### Examples:
                - **Topic**: "Business Strategy", **Subtopic**: "Market Expansion", "Mergers & Acquisitions", "Product Development", "Cost Optimization", "Others"
                - **Topic**: "Financial Position", **Subtopic**: "Debt Management", "Liquidity Risk", "Cash Flow", "Working Capital", "Others"
                - **Topic**: "Corporate Governance", **Subtopic**: "Board Structure", "Executive Compensation", "Regulatory Compliance", "Others"   
                - **Topic**: "Technology & Innovation", **Subtopic**: "Artificial Intelligence", "Digital Transformation", "R&D Investment", "Others"
                - **Topic**: "Risk Management", **Subtopic**: "Market Risk", "Operational Risk", "Regulatory Uncertainty", "Financial Stability", "Others"
                - **Topic**: "Market", **Subtopic**: "Market Expansion", "Mergers & Acquisitions", "Product Development", "Cost Optimization", "Others"
                - **Topic**: "Business Overview", **Subtopic**: "Business Strategy", "Company Description", "Geographic Presence", "Industry Trends", "Market Position", "Product Offerings", "Others"
                - **Topic**: "Contractual Obligations", **Subtopic**: "Revenue", "Earnings Per Share", "Gross Margin", "Net Income", "Others"
                - **Topic**: "Critical Accounting Policies", **Subtopic**: "Allowance for Doubtful Accounts", "Goodwill Impairment", "Income Taxes", "Inventory Valuation", "Revenue Recognition", "Share-Based Compensation", "Others"
                - **Topic**: "Financial Performance", **Subtopic**: "EBITDA", "Earnings Per Share", "Expenses", "Gross Profit", "Net Income", "Operating Income", "Revenues", "Others"
                - **Topic**: "Forward Looking Statements", **Subtopic**: "Assumptions", "Future Outlook", "Growth Strategy", "Market Opportunities", "Potential Risks", "Projections", "Others"
                - **Topic**: "Liquidity and Capital Resources", **Subtopic**: "Capital Expenditures", "Cash Flow", "Credit Facilities", "Debt Management", "Financing Activities", "Investing Activities", "Working Capital", "Others"
                - **Topic**: "Off Balance Sheet Arrangements", **Subtopic**: "Commitments", "Contingent Liabilities", "Guarantees", "Leases", "Variable Interest Entities", "Others"
                - **Topic**: "Recent Accounting Pronouncements", **Subtopic**: "Adoption Impact", "Impact Assessment", "Implementation Plans", "New Standards", "Others"
                - **Topic**: "Recent Developments", **Subtopic**: "Acquisitions", "Divestitures", "New Products", "Strategic Initiatives", "Others"
                - **Topic**: "Regulatory and Legal Matters", **Subtopic**: "Compliance", "Environmental Compliance", "Legal Proceedings", "Legislative Changes", "Regulatory Changes", "Others"
                - **Topic**: "Risk_Factors", **Subtopic**: "Competitive Risks", "Economic Conditions", "Financial Risks", "Market Risks", "Operational Risks", "Regulatory Risks", "Others"
                - **Topic**: "Segment Information", **Subtopic**: "Geographic Segments", "Product Segments", "Customer Segments", "Segment Performance", "Segment Profitability", "Segment Revenue", "Others"
                - **Topic**: "Sustainability_and_CSR", **Subtopic**: "Environmental Impact", "Social Responsibility", "Sustainability Initiatives", "Others"
                - **Topic**: "Accounting Policies", **Subtopic**: "Amortization", "Depreciation", "Revenue Recognition", "Income Taxes", "Leases", "Fair Value", "Goodwill"
                - **Topic**: "Auditor Report", **Subtopic**: "Audit Opinion", "Critical Audit Matters", "Internal Controls", "Basis for Opinion"
                - **Topic**: "Cash Flow", **Subtopic**: "Operating Activities", "Investing Activities", "Financing Activities"
                - **Topic**: "Corporate Governance", **Subtopic**: "Board Structure", "Executive Compensation", "Internal Controls", "Strategic Planning"
                - **Topic**: "Financial Performance", **Subtopic**: "Revenue", "Operating Income", "Net Income", "EPS", "Segment Results"
                - **Topic**: "Financial Position", **Subtopic**: "Assets", "Liabilities", "Equity", "Working Capital", "Investments"
                - **Topic**: "Business Overview", **Subtopic**: "Business Model", "Market Position", "Geographic Presence", "Industry Overview"
                - **Topic**: "Competition", **Subtopic**: "Market Share", "Competitive Advantages", "Industry Trends"
                - **Topic**: "Environmental Risks", **Subtopic**: "Climate Change", "Sustainability", "Resource Management"
                - **Topic**: "External Factors", **Subtopic**: "Economic Conditions", "Geopolitical Risks", "Market Conditions"
                - **Topic**: "Financial Risks", **Subtopic**: "Credit Risk", "Liquidity Risk", "Interest Rate Risk", "Market Risk"
                - **Topic**: "Regulatory Matters", **Subtopic**: "Compliance", "Legal Proceedings", "Regulatory Changes"
                - **Topic**: "Strategic Initiatives", **Subtopic**: "Growth Strategy", "Market Expansion", "Innovation"
                - **Topic**: "Operational Performance", **Subtopic**: "Efficiency", "Productivity", "Cost Management"
                - **Topic**: "Market Analysis", **Subtopic**: "Market Trends", "Consumer Behavior", "Competition"
                - **Topic**: "Industry Specific Information", **Subtopic**: "Industry Policy", "Industry Trends", "Regulatory Environment", "Competitive Landscape", "Others"
                ### Keywords:
                - **Primary Keywords**: {', '.join(top_keywords)}
                - **Secondary Keywords**: {', '.join([k.replace('_', ' ') for k in keywords[5:8]])}

                ### Context:
                {doc_context}

                ### Requirements:
                - Use standard financial and business terminology
                - Ensure topic labels are broad yet meaningful
                - Ensure subtopics are specific and relevant
                - Classify topics based on:
                  1. Financial Fundamentals (performance, position, cash flow)
                  2. Business Operations (strategy, market, competition)
                  3. Governance & Control (policies, audits, compliance)
                  4. Risk Factors (financial, operational, external)
                - Output format: "Topic: [Label], Subtopic: [Specific Area]"
                - Be concise and specific."""

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a financial analyst specializing in earnings call analysis and business intelligence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=50
            )

            result = response.choices[0].message.content.strip()
            # Parse the result assuming the output format is "Topic: [Label], Subtopic: [Specific Area]"
            topic_label, subtopic_label = "", ""
            try:
                parts = result.split(",")
                for p in parts:
                    if "Topic" in p:
                            topic_label = p.split(":")[1].strip()
                    elif "Subtopic" in p:
                        subtopic_label = p.split(":")[1].strip()
            except Exception as e:
                logger.error(f"Error parsing topic label: {e}")
                topic_label = ' '.join(top_keywords[:1])
                subtopic_label = ' '.join(top_keywords[1:3])

            # Cache the result
            cache = {}
            cache[cache_key] = f"{topic_label}, {subtopic_label}"
            with open(cache_file, 'w') as f:
                json.dump(cache, f)

            return topic_label.strip(), subtopic_label.strip()
        except Exception as e:
            logger.error(f"Error generating topic label: {e}")
            if keywords:
                topic_label = ' '.join(top_keywords[:1])
                subtopic_label = ' '.join(top_keywords[1:3])
            else:
                topic_label = "Unlabeled Topic"
                subtopic_label = "Unlabeled Subtopic"
            return topic_label.strip(), subtopic_label.strip()

    def save_topic_keywords(self, topic_model: BERTopic) -> pd.DataFrame:
        """Generate and save topic keywords with labels."""
        try:
            # Create temp directory if it doesn't exist
            temp_dir = os.path.join(gl.output_folder, 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            
            checkpoint_file = os.path.join(temp_dir, 'topic_keywords_checkpoint.pkl')
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'rb') as f:
                    return pickle.load(f)
                
            # Get basic topic info
            topic_info = topic_model.get_topic_info()
            
            # Add representative documents
            rep_docs = topic_model.representative_docs_ if hasattr(topic_model, "representative_docs_") else {}
            topic_info['Representative_Docs'] = topic_info['Topic'].map(lambda x: rep_docs.get(x, []))
            
            # Generate labels (split into topic and subtopic)
            main_topics = []
            subtopics = []
            for _, row in tqdm(topic_info.iterrows(), desc="Generating topic labels"):
                if row['Topic'] == -1:
                    main_topics.append("No Topic")
                    subtopics.append("")
                    continue
                keywords = [word for word, _ in topic_model.get_topics()[row['Topic']]]
                docs = row['Representative_Docs']
                topic_label, subtopic_label = self.generate_topic_label(keywords, docs)
                main_topics.append(topic_label)
                subtopics.append(subtopic_label)
            
            topic_info['Topic_Label'] = main_topics
            topic_info['Subtopic_Label'] = subtopics

            # Log the number of topics with labels
            logger.info(f"Generated labels for {len(main_topics)} topics")

            # Check if Topic_Label column is populated
            if 'Topic_Label' not in topic_info.columns or topic_info['Topic_Label'].isnull().all():
                logger.error("Topic_Label column is missing or not populated")
                raise ValueError("Topic_Label column is missing or not populated")

            output_path = os.path.join(
                gl.output_folder, 
                f"topic_keywords_{gl.N_NEIGHBORS[0]}_{gl.N_COMPONENTS[0]}_{gl.MIN_CLUSTER_SIZE[0]}_{self.n_topics}_{gl.YEAR_START}_{gl.YEAR_END}.csv"
            )
            topic_info.to_csv(output_path, index=False)
            logger.info(f"Saved topic keywords with labels to {output_path}")
            return topic_info
        except Exception as e:
            logger.error(f"Error in save_topic_keywords: {e}")
            raise

    def save_figures(self, topic_model: BERTopic) -> None:
        """Save visualization figures with improved quality and efficiency."""
        try:
            os.makedirs(gl.output_fig_folder, exist_ok=True)
            
            # Get topic information
            topic_info = topic_model.get_topic_info()
            sorted_topics = topic_info[topic_info['Topic'] != -1].sort_values('Count', ascending=False)
            n_topics = len(sorted_topics)
            
            if n_topics == 0:
                logger.warning("No topics to visualize")
                return
                
            # Determine number of topics to plot
            top_n_topics = max(1, min(gl.num_topic_to_plot if gl.num_topic_to_plot > 0 else n_topics, n_topics))
            logger.info(f"Visualizing top {top_n_topics} topics out of {n_topics} total topics")
            
            # Base configuration for all plots
            base_config = {
                'width': 1600,
                'height': 1000,
                'margin': dict(l=150, r=150, t=100, b=100),
                'showlegend': True,
                'template': 'plotly_white',
                'font': dict(size=12, family='Arial'),
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white'
            }
            
            # Create custom labels
            custom_labels = self._create_custom_labels(topic_info)
            
            # Generate and save each visualization type
            self._save_barchart(topic_model, top_n_topics, custom_labels, base_config)
            self._save_distance_map(topic_model, custom_labels, base_config)
            self._save_heatmap(topic_model, custom_labels, base_config)
            self._save_hierarchy(topic_model, custom_labels, base_config)
            
            logger.info(f"All visualizations saved to {gl.output_fig_folder}")
            
        except Exception as e:
            logger.error(f"Error in save_figures: {e}")
            logger.error(traceback.format_exc())

    def _save_distance_map(self, topic_model: BERTopic, custom_labels: dict, base_config: dict) -> None:
        """Save intertopic distance map."""
        fig = topic_model.visualize_topics(custom_labels=custom_labels)
        fig.update_layout(**base_config)
        fig.update_layout(
            xaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridwidth=0.5, gridcolor='lightgray')
        )
        self._save_figure(fig, 'intertopic_distance_map')

    def _save_heatmap(self, topic_model: BERTopic, custom_labels: dict, base_config: dict) -> None:
        """Save heatmap visualization."""
        fig = topic_model.visualize_heatmap(custom_labels=custom_labels)
        fig.update_layout(**base_config)
        fig.update_layout(
            xaxis={'side': 'bottom'},
            yaxis={'side': 'left'}
        )
        self._save_figure(fig, 'heatmap')

    def _save_hierarchy(self, topic_model: BERTopic, custom_labels: dict, base_config: dict) -> None:
        """Save hierarchy visualization."""
        fig = topic_model.visualize_hierarchy(
            custom_labels=custom_labels,
            width=base_config['width'],
            height=base_config['height'],
            color_threshold=1.0,
            orientation='left',
            leaf_font_size=10,
            leaf_rotation=0,
            link_colorscale='RdBu'
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
            textfont=dict(size=10, family='Arial'),
            leaf=dict(textfont=dict(size=8, color='darkslategray')),
            link=dict(colorscale='RdBu', width=0.5)
        )
        
        self._save_figure(fig, 'hierarchy', scale=2)

    def _create_custom_labels(self, topic_info: pd.DataFrame) -> dict:
        """Create custom labels for topics."""
        custom_labels = {}
        for _, row in topic_info.iterrows():
            if row['Topic'] != -1:
                topic_label = row.get('Topic_Label', '').strip()
                subtopic_label = row.get('Subtopic_Label', '').strip()
                if topic_label and subtopic_label:
                    custom_labels[row['Topic']] = f"{topic_label} - {subtopic_label}"
                else:
                    custom_labels[row['Topic']] = f"Topic {row['Topic']}"
        return custom_labels
    
    def _save_barchart(self, topic_model: BERTopic, top_n_topics: int, custom_labels: dict, base_config: dict) -> None:
        """Save barchart visualization."""
        fig = topic_model.visualize_barchart(top_n_topics=top_n_topics, custom_labels=custom_labels)
        fig.update_layout(**base_config)
        fig.update_traces(
            marker_color='rgb(55, 126, 184)',
            marker_line_color='rgb(25, 96, 154)',
            marker_line_width=1.5,
            opacity=0.8
        )
        self._save_figure(fig, 'barchart')

    def update_topic_labels(self, topic_info: pd.DataFrame, topic_model: BERTopic) -> BERTopic:
        """Update topic model with topic and subtopic labels from topic_info.
        
        Args:
            topic_info: DataFrame containing topic information with Topic_Label and Subtopic_Label columns
            topic_model: BERTopic model to update
            
        Returns:
            Updated BERTopic model
        """
        try:
            logger.info("Updating topic model with topic and subtopic labels")
            
            # Create dictionaries for topic and subtopic labels
            topic_labels = {}
            subtopic_labels = {}
            for _, row in topic_info.iterrows():
                topic_id = row['Topic']
                if topic_id != -1:  # Skip outlier topic
                    # Remove any quotes from the labels when reading from DataFrame
                    topic_labels[topic_id] = row['Topic_Label'].strip().replace('"', '')
                    subtopic_labels[topic_id] = row['Subtopic_Label'].strip().replace('"', '')
                else:
                    topic_labels[topic_id] = "No Topic"  # Remove quotes
                    subtopic_labels[topic_id] = ''  # Empty string without quotes
            
            # Add labels to model as custom attributes
            topic_model.topic_labels_ = topic_labels
            topic_model.subtopic_labels_ = subtopic_labels
            
            # Update the topic names in the model
            custom_labels = {topic_id: f"{label} - {subtopic_labels[topic_id]}".strip() 
                           for topic_id, label in topic_labels.items()}
            topic_model.set_topic_labels(custom_labels)
            
            logger.info(f"Updated {len(topic_labels)} topics with labels")
            return topic_model
            
        except Exception as e:
            logger.error(f"Error in update_topic_labels: {e}")
            logger.error(traceback.format_exc())
            raise
