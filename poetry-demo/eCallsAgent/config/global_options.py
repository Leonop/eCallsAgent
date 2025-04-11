import os
from typing import Dict, List
import pandas as pd

# Directory locations
UNIQUE_KEYS = ['companyid', 'keydevid', 'transcriptid', 'transcriptcomponentid'] # composite key in earnings call data
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ECALLS_DIR = os.path.join(PROJECT_DIR, "eCallsAgent")

# File names and data settings
data_filename = 'ecc_transcripts_2006_2020.csv'
data_filename_prefix = 'Attn'  # Base filename for the data
figure_base_name = f'bertopic_{data_filename}'  # Base name for figure files

# Main directories
output_folder = os.path.join(ECALLS_DIR, "output")
temp_folder = os.path.join(output_folder, "temp")
models_folder = os.path.join(output_folder, "models")
input_folder = os.path.join(ECALLS_DIR, "input_data")
log_folder = os.path.join(output_folder, "log_files")
embeddings_folder = os.path.join(temp_folder, "embeddings")
figures_folder = os.path.join(output_folder, "figures")

# Create all required directories
required_dirs = [
    output_folder,
    temp_folder,
    models_folder,
    input_folder,
    log_folder,
    embeddings_folder,
    figures_folder
]

for directory in required_dirs:
    os.makedirs(directory, exist_ok=True)

# File paths
stop_list = pd.read_csv(os.path.join(ECALLS_DIR, "input_data", "external", "stoplist.csv"))['stopwords'].tolist()
MODEL_SCORES = os.path.join(output_folder, "models", "model_scores.txt")
DATE_COLUMN = "mostimportantdateutc"
TOPIC_SCATTER_PLOT = os.path.join(figures_folder, "topic_scatter_plot.pdf")
num_topic_to_plot = 20 # top_N topics to plot
MODEL_SELECTION_RESULTS = os.path.join(output_folder, "models", "model_selection_results.csv")
TEXT_COLUMN = "componenttext" # the column in the main earnings call data that contains the earnings transcript
START_ROWS = 0 # start row to read from the csv file
NROWS = 15000000 # number of rows to read from the csv file
CHUNK_SIZE = 1000 # number of rows to read at a time
YEAR_END = 2013 # train the model on data from start year to this year
YEAR_START = 2011 # start year of the data
GPU_MEMORY = 40 # in GB
MIN_COUNT = 5 # minimum number of occurrences for a bigram/trigram to be considered
THRESHOLD = 10 # controls the tendency to form phrases, higher means fewer phrases

# Batch Size for Bert Topic Model Training in BERTopic_big_data_hpc.py
GPU_BATCH_SIZE = 512 # For V100, For Generating Embeddings with DeBERTa-v3-large
CPU_BATCH_SIZE = 2000 # For CPU, For Training Topic Model in Phase 2
DOCS_PER_RUN = 150_000 # For V100, For Training Topic Model
OPTIMAL_DOCS_PER_TOPIC = 5 # For V100, For Training Topic Model
# UMAP and embedding reduction parameters
N_NEIGHBORS = [30]      # Increased from 15 for capturing a more global structure in the data
N_COMPONENTS = [50]     # Increased from 5 for a richer low-dimensional embedding space
MIN_DIST = [0.1]        # Controls how tightly points can be packed; 0.1 encourages good separation
base_batch_size = 1024
GPU_CHUNK_SIZE = 5000
EMBEDDING_DIM = 1024

final_parameters = {
    "n_neighbors": 10,                    # max(12, min(10, 1_000_000 // 8000)) = max(12, 10) = 12
    "n_components": 300,                  # adjusted for large embedding + not A100 (see below)
    "min_dist": 0.025,                     # assuming average doc length < 200
    "min_cluster_size": 5,              # Reduces over-fragmentation; encourages merging noisy small clusters.
    "min_samples": 3,                    # relaxed for more clusters
    "cluster_selection_epsilon": 0.015,   # increased slightly for broader clusters
}

# HDBSCAN clustering parameters
MIN_SAMPLES = [10]      # Remains at 10 to still capture smaller clusters reliably
MIN_CLUSTER_SIZE = [30] # Decreased from 40 to ensure clusters have enough documents for robust statistics

# Topic modeling target parameters
NR_TOPICS = [300]        # Typically, a reduced, manageable number of final topics

# Topic representations and feature extraction
TOP_N_WORDS = [20]      # Increased from 15 to capture a broader context in topic keywords
METRIC = ['cosine']

# Enhanced embedding models - from general purpose to finance-specific
EMBEDDING_MODELS = [
    'sentence-transformers/all-mpnet-base-v2',   # Original model (good general purpose)
    'BAAI/bge-large-en-v1.5',                    # Better for retrieval and similarity (1024 dimensions)
    'openai/text-embedding-ada-002', 
    'meta-llama/Llama-2-7b-chat-hf',
    'thenlper/gte-large',                        # Large model with excellent semantic understanding (1024 dimensions) 
    'intfloat/e5-large-v2',                      # Large model with strong performance (768 dimensions)
    'ProsusAI/finbert',                          # Finance-specific BERT model
    'yiyanghkust/finbert-tone',                  # Finance-specific model with tone analysis
    'nbroad/ESG-BERT'                            # Environmental, Social, Governance reports model
]

# Default model index to use (0 = original model, 1-6 = enhanced models)
# Change this value to use a different model
DEFAULT_MODEL_INDEX = 4  # Updated to use thenlper/gte-large

# Parameters for Phase 2: Topic Modeling Distillation
MAX_ADAPTIVE_REPRESENTATIVES = 600_000 # Maximum number of adaptive representative documents to use for distillation in phase 2. 
MIN_DOCS_PER_TOPIC = 10 # Minimum number of documents per topic to use for distillation
MAX_DOCS_PER_TOPIC = 5000 # Maximum number of documents per topic to use for distillation

# Vectorizer parameters for c-TF-IDF computation
MAX_DF = [0.95]         # Retains filtering for terms appearing in over 95% of documents
MIN_DF = [0.01]        # Terms must appear in at least 5% of documents (optionally test [0.01, 0.05])

# Updated grid search parameters to produce more topics
GRID_SEARCH_PARAMETERS = {
    'n_neighbors': [5, 8, 10, 12],        # Lower values (5-15) tend to produce more topics
    'n_components': [50, 75, 100, 200, 400],     # Higher dimensionality can help preserve more structure
    'min_dist': [0.0, 0.05, 0.1],       # Lower values create tighter clusters
    'min_samples': [5, 8, 10, 15],         # Lower values are less strict on what forms a cluster
    'min_cluster_size': [15, 30, 50, 100],       # Lower values allow smaller topics
    'cluster_selection_epsilon': [0.01, 0.05, 0.1]
}

# Flag to skip grid search and use default parameters instead
SKIP_GRID_SEARCH = True  # Set to True to skip grid search, False to perform grid search

# Parameter set to use when skipping grid search
# Options: 'default', 'more_topics', 'fewer_topics'
PARAMETER_SET = 'default'

# File naming patterns
data_filename_prefix = 'Attn'  # Base filename for the data
figure_base_name = f'bertopic_{data_filename}'  # Base name for figure files

# Temporary file paths
TEMP_EMBEDDINGS = os.path.join(embeddings_folder, f'{data_filename_prefix}_embeddings_{EMBEDDING_MODELS[DEFAULT_MODEL_INDEX].replace("/", "-").replace(" ", "_")}.mmap') # good for large emebeddings   
TEMP_TOPIC_KEYWORDS = os.path.join(temp_folder, f'{data_filename_prefix}_topic_keywords.pkl')
TEMP_TOPIC_LABELS = os.path.join(temp_folder, f'{data_filename_prefix}_topic_labels.json')

# SAVE RESULTS 
SAVE_RESULTS_COLS = ["params", "score", "probability"]


SEED_TOPICS  = [
  ["Financial Performance (Revenue & Sales)", "revenue", "income statement", "sales", "top-line", "total revenue"],
  ["Financial Performance (Growth & YOY Trends)", "growth", "expansion", "increase", "rise", "escalation", "year over year", "quarter over quarter", "yoy", "qoq"],
  ["Financial Performance (Profit & Earnings)", "profit", "net income", "bottom-line", "earnings", "net profit"],
  ["Financial Performance (Margins)", "gross margin", "profit ratio", "markup percentage", "gross profit rate", "sales margin", "operating profit margin", "ebit margin", "operating income margin", "profit margin"],
  ["Financial Performance (Return & Investment Efficiency)", "operational efficiency", "return on equity", "equity returns", "shareholder return", "net income to equity", "equity performance", "profitability ratio", "return on assets", "asset returns", "asset performance", "asset profitability", "return on investment", "investment returns", "investment performance", "net income to investment", "investment profitability"],
  ["Costs & Expenses (Cost Structure & Overhead)", "cost", "overhead", "resource costs"],
  ["Costs & Expenses (Expense Items)", "expenses", "expenditure", "cost of goods sold", "cogs", "production costs", "manufacturing expenses", "selling, general, and administrative expenses", "sg&a", "non-production costs", "administrative burden"],
  ["Costs & Expenses (Cost Reduction & Efficiency)", "cost-cutting initiatives", "expense reduction", "efficiency programs", "budget trimming", "cost optimization", "supply chain optimization", "overhead reduction", "restructuring", "streamlining"],
  ["Costs & Expenses (Raw Material & Commodity Costs)", "raw material costs", "input costs", "commodity prices"],
  ["Costs & Expenses (Labor & Payroll)", "labor costs", "workforce expenses", "employee costs", "payroll expenses"],
  ["Costs & Expenses (Operating Margin)", "operating margin"],
  ["Cash & Liquidity (Cash & Liquid Assets)", "cash", "cash flow", "liquidity", "cash position", "cash balance", "liquid assets"],
  ["Cash & Liquidity (Debt & Liabilities)", "debt", "liabilities", "borrowing", "indebtedness", "debt burden"],
  ["Cash & Liquidity (Financial Health)", "financial position", "balance sheet", "financial health", "financial stability", "financial standing"],
  ["Cash & Liquidity (Working Capital Indicators)", "current assets", "quick ratio", "current ratio"],
  ["Equity & Capital Management (Equity & Ownership)", "equity", "shareholders", "stockholders", "ownership", "equity holders"],
  ["Equity & Capital Management (Dividends & Payouts)", "dividend", "dividend payment", "dividend yield", "dividend payout", "dividend policy", "payout policy", "shareholder distributions", "income distribution plan", "yield policy"],
  ["Equity & Capital Management (Structure & Leverage)", "capital structure", "financial leverage", "loan balances", "bond rating", "debt-to-equity ratio", "equity financing", "credit rating"],
  ["Equity & Capital Management (Buybacks & M&A)", "share buyback", "merge and acquisition", "strategic investment", "share buyback plans", "stock repurchase program", "share repurchases", "buyback initiative", "stock retirement", "equity reduction"],
  ["Investments & Expenditures (Investment Activities)", "investment", "investing", "investment spending"],
  ["Investments & Expenditures (Capital Expenditures)", "capital expenditure", "capex", "capital expenditure plans", "capex projections", "asset acquisition strategy", "infrastructure spending", "capital outlays"],
  ["Investments & Expenditures (Working Capital & Liquidity)", "working capital management", "cash flow management", "operational liquidity", "short-term asset management"],
  ["Forecast & Outlook (Short-Term & Quarterly)", "short-term forecast", "upcoming quarter outlook", "near-term projections", "quarterly expectations", "forward guidance"],
  ["Forecast & Outlook (Long-Term & Annual)", "full-year outlook", "annual forecast", "yearly projection", "fiscal year outlook", "long-term", "multi-year goals", "strategic financial objectives", "extended financial outlook", "future financial aims", "investment horizon"],
  ["Forecast & Outlook (Strategic & Growth)", "growth strategy", "strategic horizon", "strategic initiatives", "sustainable growth", "forward-looking", "future potential", "market leadership", "value creation", "future roadmap", "comprehensive view", "down the road", "going forward", "upcoming quarters", "long run", "in future", "enduring value"],
  ["Forecast & Outlook (Industry & Economic)", "industry forecast", "sector outlook", "market projections", "industry trends", "vertical predictions", "sector expectations", "economic forecast", "macroeconomic outlook", "economic projections", "economic trends", "economic expectations", "consumer spending", "economic crisis", "economic expansion", "economic recession", "fiscal policy", "monetary policy", "inflation", "interest rate", "jobs data", "market volatility", "pandemic", "recovery", "recession", "unemployment rate", "consumer confidence", "federal reserve", "fed"],
  ["Risks (Currency & Interest Rate)", "foreign exchange impact", "currency effects", "forex exposure", "exchange rate influence", "currency risk", "interest rate sensitivity", "rate exposure", "interest risk", "borrowing cost sensitivity"],
  ["Risks (Liquidity & Credit)", "liquidity risk", "financial tightness", "credit crunch", "cash reserve", "bankruptcy", "credit risk", "default risk", "credit exposure", "credit quality", "credit portfolio", "counterparty risk"],
  ["Risks (Operational & Legal)", "operational risk", "business continuity", "health and safety", "logistics disruption", "incident management", "internal control", "legal risk", "compliance", "lawsuit", "litigation", "anticorruption", "anti-monopoly", "regulator", "settlement"],
  ["Risks (Political & Regulatory)", "political risk", "political uncertainty", "regulatory reform", "policy goals", "public opinion", "government policy", "sanction", "trade policy", "trade war", "tariff", "election", "geopolitical uncertainty"],
  ["Risks (Sovereign & International)", "sovereign risk", "sovereign debt", "sovereign default", "country downgrade", "sovereign credit rating", "international policy"],
  ["Risks (Climate & Natural Disaster)", "climate risk", "extreme weather", "drought", "flood", "wildfire", "storm", "tsunami", "hurricane", "natural disaster", "global warming"],
  ["Risks (Regulatory Challenges & Disputes)", "regulatory challenges", "compliance issues", "legal hurdles", "regulatory environment", "policy challenges", "litigation updates", "legal proceedings", "lawsuit status", "court case developments", "legal disputes"],
  ["Operations & Productivity (Automation & Digital)", "automation", "robotic process automation", "rpa", "digital transformation", "efficiency"],
  ["Operations & Productivity (Capacity & Resources)", "capacity utilization", "resource usage", "production capacity", "facility utilization", "asset efficiency"],
  ["Operations & Productivity (Cost Efficiency & Layoffs)", "cost cutting", "cost efficiency", "labor cost", "labor efficiency", "labor productivity", "layoff"],
  ["Operations & Productivity (Process & Productivity)", "operational improvements", "process enhancements", "efficiency gains", "operational streamlining", "productivity boosts", "performance upgrades", "productivity metrics", "efficiency measures", "output indicators", "performance ratios", "productivity kpis", "operational effectiveness"],
  ["Operations & Productivity (Supply Chain & Inventory)", "supply chain efficiency", "logistics performance", "procurement effectiveness", "distribution efficiency", "supply chain optimization", "inventory turnover", "stock rotation", "inventory efficiency", "stock velocity", "goods turnover rate"],
  ["Workforce & Human Capital (Headcount & Turnover)", "employee headcount", "workforce size", "staff numbers", "personnel count", "headcount management", "employee turnover rate", "staff attrition", "workforce stability", "retention challenges", "employee departures"],
  ["Workforce & Human Capital (Talent & Retention)", "talent acquisition and retention strategies", "hiring initiatives", "employee retention programs", "workforce planning", "talent management"],
  ["Workforce & Human Capital (Diversity & Engagement)", "workforce diversity and inclusion", "diversity metrics", "inclusivity efforts", "equal opportunity initiatives", "employee engagement metrics", "staff satisfaction", "workforce morale", "employee loyalty", "job satisfaction", "team engagement"],
  ["Market & Competition (Market Share & Rivalry)", "market share", "market dominance", "market leadership", "market position", "business footprint", "competitive landscape", "competitive risk", "competitive environment", "market competition", "competitor analysis", "competitive advantage", "industry rivalry"],
  ["Market & Competition (Brand & Pricing)", "brand loyalty", "price competition", "price war", "brand strength", "brand power", "brand health", "brand recognition", "brand equity", "brand value"],
  ["Market & Competition (Acquisition & Loyalty)", "new customer growth", "client onboarding", "customer wins", "new business generation", "expanding customer base", "client loyalty", "churn rate", "customer stickiness", "repeat business", "customer longevity"],
  ["Market & Competition (Satisfaction & Service Quality)", "customer satisfaction index", "loyalty metric", "referral likelihood", "customer advocacy", "satisfaction score", "service quality", "customer satisfaction measures", "service performance indicators", "quality assurance metrics", "service level achievements", "customer experience scores"],
  ["Market & Competition (Product Launches & Mix)", "new product launches", "product releases", "new offerings", "product introductions", "market debuts", "new solutions", "product mix changes", "product portfolio shifts", "offering diversification", "product line adjustments", "sales mix"],
  ["Market & Competition (Sales Pipeline & Marketing)", "sales pipeline", "sales funnel", "prospect pipeline", "revenue pipeline", "deal flow", "backlog or order book status", "unfilled orders", "work in progress", "future revenue", "committed sales", "customer acquisition costs", "cac", "cost per customer", "marketing efficiency", "acquisition spend"],
  ["Market & Competition (Lifetime Value & Productivity)", "lifetime value of customers", "ltv", "customer worth", "long-term customer value", "client profitability", "marketing effectiveness", "roi on marketing", "campaign performance", "promotional impact", "advertising effectiveness", "sales force productivity", "sales efficiency", "rep performance", "sales team effectiveness", "selling productivity"],
  ["Market & Competition (Benchmarking & Positioning)", "industry benchmarking", "peer comparison", "competitive benchmarking", "market positioning", "sector performance ranking"],
  ["Segment Reporting (Business & Divisions)", "business unit breakdowns", "divisional performance", "segment analysis", "unit-level results", "departmental breakdown"],
  ["Segment Reporting (Geographic)", "geographic segment performance", "regional results", "territorial analysis", "location-based performance"],
  ["Segment Reporting (Product & Offerings)", "product category performance", "product line results", "offering performance", "product mix analysis"],
  ["Segment Reporting (Customer & Demographics)", "customer segment analysis", "client group performance", "demographic performance", "target market results"],
  ["Technology & Innovation (R&D & Breakthroughs)", "research_and_development", "r&d spending", "innovation funding", "product development costs", "research expenditure", "technology investments", "innovation pipeline", "development roadmap", "future products", "corporate innovation", "innovation", "r&d", "breakthrough technologies"],
  ["Technology & Innovation (Roadmap & Upgrades)", "product_roadmap", "development timeline", "product strategy", "future releases", "product evolution plan", "feature roadmap", "digital transformation initiatives", "digitalization efforts", "tech modernization", "it transformation", "technology upgrade", "it infrastructure investments", "tech spending", "system upgrades", "it capex", "technology infrastructure"],
  ["Technology & Innovation (E-commerce & Data)", "e-commerce performance", "online sales", "digital revenue", "internet retail performance", "web store results", "data analytics capabilities", "business intelligence", "data-driven insights", "analytics infrastructure", "predictive modeling"],
  ["Technology & Innovation (AI & Machine Learning)", "artificial intelligence and machine learning applications", "ai integration", "ml implementation", "cognitive computing", "smart algorithms", "transformer", "chatgpt", "large language model", "neural networks", "reinforcement learning"],
  ["Technology & Innovation (Intellectual Property)", "patent portfolio", "ip assets", "patent holdings", "invention rights", "proprietary technology", "trademark developments", "brand protection", "trademark portfolio", "intellectual property rights", "licensing agreements", "ip licensing", "technology transfer", "patent licensing", "trademark licensing", "ip litigation", "patent disputes", "trademark infringement", "copyright cases", "intellectual property lawsuits"],
  ["ESG (Extreme Weather & Climate Change)", "extreme weather", "climate change", "weather events", "weather conditions", "weather patterns", "weather forecasts", "weather impact", "weather risk", "weather disasters", "weather emergencies", "weather alerts", "weather warnings", "weather emergencies", "extreme weather", "drought", "flood", "wildfire", "storm", "tsunami", "hurricane", "natural disaster", "global warming"],
  ["ESG (Environmental: Initiatives & Carbon)", "environmental initiatives", "eco-friendly programs", "green initiatives", "sustainability efforts", "carbon emissions", "renewable energy", "carbon footprint", "carbon footprint reduction efforts", "emissions reduction", "climate impact mitigation", "greenhouse gas reduction"],
  ["ESG (Social Responsibility)", "social responsibility programs", "community initiatives", "social impact", "philanthropic efforts", "corporate citizenship"],
  ["ESG (Governance & Ethics)", "governance practices", "corporate governance", "board practices", "ethical leadership", "shareholder rights", "management oversight", "sustainable sourcing", "ethical procurement", "responsible sourcing", "supply chain sustainability", "eco-friendly suppliers"],
  ["M&A & Corporate Strategy (M&A & Consolidation)", "merger and acquisition activities", "m&a strategy", "corporate takeovers", "business combinations", "acquisition plans", "consolidation efforts"],
  ["M&A & Corporate Strategy (Expansion & Diversification)", "diversification efforts", "business expansion", "new venture development", "portfolio diversification", "risk spreading"],
  ["M&A & Corporate Strategy (Market Entry & Partnerships)", "new market entry", "geographic expansion", "market entry", "territorial growth", "global reach expansion", "regional diversification", "partnerships and collaborations", "strategic alliances", "joint ventures", "cooperative agreements", "business partnerships", "collaborative initiatives"],
  ["Other (Tax & Credits)", "corporate tax", "effective tax rate", "tax liabilities", "tax planning", "tax credits", "deferred taxes"],
  ["Other (Impairments & Write-offs)", "allowance", "write-off", "impairment charge", "asset impairment", "goodwill impairment"],
  ["Other (Challenges & Headwinds)", "challenges", "headwind", "conservative outlook", "lack of visibility", "mixed results", "expense growth", "falloff", "dead horse", "downtime", "nightmare"]
]

