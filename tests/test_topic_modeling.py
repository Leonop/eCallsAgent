import numpy as np
from eCallsAgent.core.model_eval import ModelEvaluator
from eCallsAgent.core.topic_modeler import TopicModeler
from eCallsAgent.utils.cuda_setup import setup_cuda
import logging
import random

logger = logging.getLogger(__name__)

# Define topic templates with more realistic business language
TOPIC_TEMPLATES = {
    "financial": [
        "In Q{} {}, we reported revenue of ${} million, with EBITDA margin of {}%. Operating cash flow improved to ${} million.",
        "Our {} quarter financial results showed ${} million in net income, representing a {}% increase year-over-year.",
        "The company achieved ${} billion in annual revenue, with gross margins expanding to {}% driven by operational efficiency.",
        "Return on equity reached {}%, while maintaining a healthy debt-to-equity ratio of {}.",
        "Capital expenditure was ${} million, focused on expanding our {} facilities."
    ],
    "operations": [
        "Manufacturing efficiency improved {}% through implementation of {} automation systems.",
        "Supply chain optimization resulted in {} day reduction in inventory turnover.",
        "Operational costs decreased by {}% following the implementation of {} initiative.",
        "Factory utilization rate increased to {}% with the new {} system deployment.",
        "We completed the integration of {} technology, improving throughput by {}%."
    ],
    "market": [
        "Market share in {} segment grew to {}%, outperforming industry average by {} basis points.",
        "New product launches in {} contributed {}% to quarterly revenue growth.",
        "Customer acquisition costs decreased {}% while retention rate improved to {}%.",
        "Expansion into {} market is progressing with {}% penetration rate.",
        "Digital transformation initiatives led to {}% growth in online sales."
    ],
    "strategy": [
        "Strategic partnership with {} expected to generate ${} million in synergies.",
        "R&D investment increased to {}% of revenue, focusing on {} technology.",
        "Completed acquisition of {} for ${} million, expanding our {} capabilities.",
        "Innovation pipeline includes {} new products scheduled for {} launch.",
        "Cost reduction program on track to deliver ${} million in savings by {}."
    ],
    "risk": [
        "Implemented enhanced {} risk management framework across {} operations.",
        "Regulatory compliance costs increased {}% due to new {} requirements.",
        "Cybersecurity investments of ${} million strengthened our {} infrastructure.",
        "Insurance coverage expanded to include {} risks with {} coverage limit.",
        "Risk assessment identified {} key areas for {} mitigation strategies."
    ]
}

def generate_synthetic_doc():
    """Generate a synthetic earnings call transcript."""
    topic = random.choice(list(TOPIC_TEMPLATES.keys()))
    template = random.choice(TOPIC_TEMPLATES[topic])
    
    # Generate random values for placeholders
    values = {
        "Q": random.choice(['Q1', 'Q2', 'Q3', 'Q4']),
        "{}": lambda: random.choice([
            str(random.randint(1, 999)),
            str(round(random.uniform(1, 99), 1)),
            "cloud",
            "AI",
            "blockchain",
            "IoT",
            "digital",
            "global",
            "regional",
            "enterprise",
            "consumer",
            "2024",
            "2025",
            "next-gen",
            "automated",
            "integrated"
        ])
    }
    
    # Fill template with random values
    doc = template
    while '{}' in doc:
        doc = doc.replace('{}', values['{}'](), 1)
    
    # Replace Q{} with actual quarter if present
    if 'Q{}' in doc:
        doc = doc.replace('Q{}', values['Q'])
        
    return doc

def test_pipeline():
    try:
        # Initialize components
        device = setup_cuda()
        model_evaluator = ModelEvaluator()
        topic_modeler = TopicModeler(device)
        
        # Generate synthetic documents
        sample_docs = [generate_synthetic_doc() for _ in range(5000)]
        
        # Validate documents
        if not all(isinstance(doc, str) for doc in sample_docs):
            raise ValueError("Some documents are not strings")
        if not all(doc for doc in sample_docs):
            raise ValueError("Some documents are empty")
            
        logger.info(f"First document example: {sample_docs[0]}")

        # Generate embeddings
        n_docs = len(sample_docs)
        sample_embeddings = np.random.rand(n_docs, 768).astype(np.float32)
        sample_embeddings = sample_embeddings / np.linalg.norm(sample_embeddings, axis=1)[:, np.newaxis]
        
        logger.info(f"Number of documents: {len(sample_docs)}")
        logger.info(f"Embeddings shape: {sample_embeddings.shape}")
        logger.info(f"Sample document length: {len(sample_docs[0].split())}")
        
        # Train model
        model = topic_modeler.train_topic_model(sample_docs, sample_embeddings)
        
        # Test evaluation metrics
        coherence = model_evaluator.compute_coherence_score(model, sample_docs)
        silhouette = model_evaluator.compute_silhouette_score(sample_embeddings, model.topics_)
        
        print(f"Coherence score: {coherence:.4f}")
        print(f"Silhouette score: {silhouette:.4f}")
        print(f"Number of topics: {len(set(model.topics_)) - 1}")
        
    except Exception as e:
        logger.error(f"Error in test: {str(e)}")
        logger.error(f"Embeddings shape: {sample_embeddings.shape}")
        logger.error(f"Number of docs: {len(sample_docs)}")
        raise

if __name__ == "__main__":
    test_pipeline() 