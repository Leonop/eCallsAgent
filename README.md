# eCallsAgent

A Python package for analyzing earnings call transcripts using BERTopic and advanced NLP techniques.

## Features

- Document preprocessing and cleaning
- BERT-based embedding generation
- Topic modeling using BERTopic
- Model evaluation and visualization
- GPU acceleration support

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd eCallsAgent
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your earnings call data in `input_data/raw/`
2. Configure settings in `config/global_options.py`
3. Run the main script:
```bash
python -m eCallsAgent.main
```

## Project Structure

```
eCallsAgent/
├── core/                 # Core functionality
│   ├── data_handler.py   # Data processing
│   ├── embedding_generator.py  # Embedding generation
│   ├── model_eval.py    # Model evaluation
│   ├── topic_modeler.py # Topic modeling
│   └── preprocess_earningscall.py
├── config/              # Configuration files
├── utils/               # Utility functions
├── input_data/          # Input data storage
├── output/              # Output storage
└── tests/               # Test files
```

## Testing

Run tests using:
```bash
python -m unittest eCallsAgent/tests/test_pipeline.py
```

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, for acceleration)
- See requirements.txt for package dependencies

## License

[Your License]

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
├── setup.py                  # Setup eCallsAgent Package
└── requirements.txt          # Dependencies
```

## Skipping Grid Search in Topic Modeling

The topic modeling process includes a grid search step that can be time-consuming. There are several ways to skip this step:

### Option 1: Using Configuration File

Edit the `eCallsAgent/config/global_options.py` file and set:

```python
# Flag to skip grid search and use default parameters instead
SKIP_GRID_SEARCH = True
```

### Option 2: Using Command Line Arguments

When running the main script, use the `--skip-grid-search` flag:

```bash
python eCallsAgent/main.py --skip-grid-search
```

### Option 3: Selecting Parameter Sets

You can choose different parameter sets optimized for different numbers of topics:

```bash
# For default parameters
python eCallsAgent/main.py --skip-grid-search --parameter-set default

# For more topics
python eCallsAgent/main.py --skip-grid-search --parameter-set more_topics

# For fewer topics
python eCallsAgent/main.py --skip-grid-search --parameter-set fewer_topics
```

### Customizing Default Parameters

You can customize the default parameters by editing the `eCallsAgent/config/default_model_params.py` file. This file contains parameter sets for:

- `DEFAULT_UMAP_PARAMS` and `DEFAULT_HDBSCAN_PARAMS`: Balanced parameters
- `MORE_TOPICS_UMAP_PARAMS` and `MORE_TOPICS_HDBSCAN_PARAMS`: Parameters optimized for generating more topics
- `FEWER_TOPICS_UMAP_PARAMS` and `FEWER_TOPICS_HDBSCAN_PARAMS`: Parameters optimized for generating fewer, more general topics

### Performance Impact

Skipping the grid search will significantly reduce the processing time but may result in a less optimal topic model. The default parameters are designed to work well for most cases, but they may not be the best for your specific dataset.


# Earnings Call Transcript Themes

Below is an aggregated table that organizes the raw keyword pairs into higher‐level “Topic” categories with their common “Subtopics.” This table not only makes the data structural but also captures the latent themes that often surface in earnings call transcripts—such as operational efficiency, financial performance, strategic guidance, risk factors, and innovation.

| **Topic**                  | **Subtopics**                                                                                                                                                                          |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Business Operations        | Vertical Integration; Employee & Client Retention; Market Expansion; Manufacturing; Supply Chain & Logistics; Advertising; Packaging; Retail & Franchise Strategy; General    |
| Financial Performance      | Adjusted EBITDA; Revenue Growth; Gross Margin; Operating Income; Bonus Accrual; Free Cash Flow; Payout & Earnings Per Share; Non‐GAAP Measures; Other Performance Metrics         |
| Financial Position         | Currency Impact; Capital Expenditures; Debt & Cash Management; Equity Structure; Contingent Liabilities; Loan Portfolio                                                               |
| Forward Looking Statements | Future Outlook; Potential Risks; Guidance; Strategic Initiatives                                                                                                                       |
| Risk Management / Factors  | Underwriting Risk; Cybersecurity Threats; Political Turbulence; Environmental & Legal Risks; Inflation & Commodity Price Pressures                                                              |
| Governance & Controls      | Audit Participation; Board Structure; Disclosure & Regulatory Compliance; Tax & Legal Proceedings                                                                                      |
| Business Strategy          | Conservative & Pricing Approaches; Digital Transformation; Mergers & Acquisitions; Strategic Partnerships; Market Trends; Resource Allocation                                               |
| Product Development        | New Product Launches; Product Innovation & Strategy; R&D; Technology Deployment; Digital Platforms; Product Sales Performance                                                             |
| Regulatory Matters         | Orphan Designation; Legal & Regulatory Compliance; Tax Compliance; Disclosure Practices                                                                                                 |
| Market Analysis            | Commodity Prices; Market Trends; Competitive Landscape; Pricing Strategy; Supply Constraints; Channel & Distribution Strategies                                                              |
| Technology & Innovation    | Digital Media & Software Development; Network & Data Storage Strategies; Technology Integration; Disruptive Innovation                                                                     |

**Note:** The table reflects how earnings call language often intermingles operational execution with financial metrics and strategic outlook—revealing a dual focus on short‑term performance and long‑term positioning.
