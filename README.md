# eCallsAgent is an advanced natural language processing (NLP) tool leveraging BERTopic modeling, and fine-tuned by ChatGPT 4-o, to extract fundamental narratives from earnings call transcripts. Designed for financial analysts, researchers, and data scientists, it identifies key business topics and tracks their evolution over time, offering insights into corporate performance and strategy. With future potential for integration into large language models (LLMs) using chain-of-thought reasoning, eCallsAgent is a cutting-edge solution for deep reseach in financial fundamental narratives and context analysis.

@todo:
- [ ] Add more tests
- [ ] Add more documentation
- [ ] Add more examples
- [ ] Add more features
- [ ] Add more evaluation metrics
- [ ] Add more visualization
- [ ] Add more analysis
- [ ] Add more research
- [ ] Add more application
- [ ] Add more integration


# Developer:
- Name: Dr. Zicheng Xiao
- Title: Quantitative Researcher / Assistant Professor in Finance|FinTech
- Afflication: University of Arkansas, Fayetteville
- Paper: Measuring Information Quality by Using Topic Attention Divergence: Evidence from Earnings Calls (Tentative)
- Date: 2025-02-21
- Version: 0.1.0
- License: No license is granted for use, reproduction, modification, or distribution, including for research, commercial, or personal purposes, without explicit written permission from [Dr. Zicheng Xiao]. Contact [zicheng.xiao@gmail.com] for inquiries
- Email: zicheng.xiao@gmail.com
- GitHub: [https://github.com/zichengxiao]
- LinkedIn: [https://www.linkedin.com/in/zichengxiao/]

Copyright (c) [Zicheng Xiao] 2025. All rights reserved.
This project is publicly visible for viewing only. No permission is granted to use, copy, modify, or distribute this code for any purpose, including research, without explicit written consent from the author.

## Overview
Earnings calls are a goldmine of unstructured data, revealing corporate strategies, risks, and market sentiments. eCallsAgent transforms this raw information into actionable insights by:

Processing large-scale earnings call transcripts efficiently.
Generating GPU-accelerated embeddings for deep text understanding.
Applying BERTopic modeling to uncover coherent business topics.
Providing automated topic labeling and visualization tools.
Developed by Dr.Zicheng Xiao,  Quantitative Researcher, this tool is shared publicly for transparency and demonstration, though usage is restricted under a strict copyright policy.

## Features
- Advanced topic modeling using BERTopic
- Efficient processing of large-scale earnings call transcripts
- GPU-accelerated embedding generation
- Automated topic labeling and visualization
- Comprehensive evaluation metrics for model performance
- Pre-defined business-specific seed topics for enhanced topic coherence

eCallsAgent/
├── README.md                  # Project documentation
├── setup.sh                   # Setup script for environment configuration
├── settings.py                # Configuration settings
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files to ignore in version control
├── input_data/                # Data directory
│   ├── raw/                   # Raw input data
│   │   └── eCallsAgent.csv    # Original earnings call data
│   ├── processed/             # Processed data
│   │   └── preprocessed_docs.txt  # Preprocessed documents
│   ├── interim/               # Intermediate data
│   │   └── embeddings/        # Generated embeddings
│   └── external/              # External data
│       └── stoplist.csv       # Stop words list
├── eCallsAgent/               # Core package
│   ├── __init__.py            # Package initialization
│   ├── config/                # Configuration modules
│   │   ├── __init__.py
│   │   └── global_options.py  # Global settings
│   ├── core/                  # Core functionality
│   │   ├── __init__.py
│   │   ├── data_handler.py    # Data loading and preprocessing
│   │   ├── embedding_generator.py  # Embedding creation
│   │   └── topic_modeler.py   # Topic modeling logic
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── cuda_setup.py      # GPU setup utilities
│   │   ├── logging_utils.py   # Logging support
│   │   └── visualization.py   # Visualization tools
│   └── main.py                # Entry point for the tool
└── tests/                     # Unit tests
    ├── __init__.py
    ├── test_data_handler.py    # Tests for data handling
    ├── test_embedding_generator.py  # Tests for embeddings
    └── test_topic_modeler.py   # Tests for topic modeling


Getting Started
Currently, eCallsAgent is a demonstration project and not available for direct use due to copyright restrictions. However, here’s how it’s structured for potential future access:

Prerequisites
Python 3.8+
CUDA-enabled GPU (optional, for accelerated embedding generation)
Dependencies listed in requirements.txt
Setup (Hypothetical)
Clone the repository:
bash
Wrap
Copy
git clone https://github.com/zichengxiao/eCallsAgent.git
cd eCallsAgent
Run the setup script:
bash
Wrap
Copy
bash setup.sh
Install dependencies:
bash
Wrap
Copy
pip install -r requirements.txt
Configure settings in settings.py as needed.
Example Usage (Conceptual)
python
Wrap
Copy
from eCallsAgent.main import run_analysis
run_analysis(input_file="input_data/raw/eCallsAgent.csv", output_dir="output")
Note: This is a placeholder; actual use requires permission from the author.

Roadmap
- [ ] Enhance test coverage
- [ ] Expand documentation with detailed guides
- [ ] Add usage examples and sample outputs
- [ ] Implement additional features (e.g., sentiment analysis)
- [ ] Include more evaluation metrics (e.g., coherence scores)
- [ ] Improve visualization options (e.g., interactive plots)
- [ ] Deepen analysis capabilities (e.g., temporal trends)
- [ ] Conduct further research for optimization
- [ ] Explore real-world applications
- [ ] Enable integration with LLMs and chain-of-thought reasoning

Copyright © 2025 Zicheng Xiao. All rights reserved.

This project is publicly visible on GitHub for viewing purposes only. No permission is granted to use, copy, modify, or distribute this code, including for research, commercial, or personal purposes, without explicit written consent from the author. For inquiries, contact zicheng.xiao@gmail.com.

Why eCallsAgent?
Earnings calls are critical for understanding corporate narratives, yet their unstructured nature makes analysis challenging. eCallsAgent bridges this gap with state-of-the-art NLP, offering a glimpse into how AI can revolutionize financial research—if permitted for use in the future.

Contributing
Contributions are not accepted at this stage due to the project’s restrictive copyright. However, feedback or inquiries are welcome via email.

Acknowledgments
Built with BERTopic for topic modeling.
Inspired by the need for deeper insights into financial narratives.