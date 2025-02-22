# eCallsAgent

## Description
eCallsAgent is an advanced natural language processing (NLP) tool leveraging BERTopic modeling and fine-tuned by ChatGPT 4.0 to extract fundamental narratives from earnings call transcripts. Designed for financial analysts, researchers, and data scientists, it identifies key business topics and tracks their evolution over time, offering insights into corporate performance and strategy. With future potential for integration into large language models (LLMs) using chain-of-thought reasoning, eCallsAgent is a cutting-edge solution for deep research in financial fundamental narratives and context analysis.

## Developer Information
- **Name:** Dr. Zicheng Xiao
- **Title:** Quantitative Researcher / Assistant Professor in Finance|FinTech
- **Affiliation:** University of Arkansas, Fayetteville
- **Research:** Measuring Information Quality by Using Topic Attention Divergence: Evidence from Earnings Calls (Tentative)
- **Contact:** zicheng.xiao@gmail.com
- **Links:** 
  - [GitHub](https://github.com/zichengxiao)
  - [LinkedIn](https://www.linkedin.com/in/zichengxiao/)

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

## Project Structure

Getting Started
Currently, eCallsAgent is a demonstration project and not available for direct use due to copyright restrictions. However, here's how it's structured for potential future access:

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
Configure settings in eCallsAgent/config/global_options.py as needed.
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
Contributions are not accepted at this stage due to the project's restrictive copyright. However, feedback or inquiries are welcome via email.

Acknowledgments
Built with BERTopic for topic modeling.
Inspired by the need for deeper insights into financial narratives.

AIPHABIZ/
├── eCallsAgent/               # Core package
    ├── input_data/               # Data directory
    ├── output/                   # Output directory
    ├── temp/                     # Temporary directory
    ├── models/                   # Model directory
    ├── figures/                  # Figure directory
├── tests/                    # Unit tests
├── README.md                 # Documentation
├── setup.sh                  # Setup script
└── requirements.txt          # Dependencies