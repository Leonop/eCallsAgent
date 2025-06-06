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
YEAR_END = 2014 # train the model on data from start year to this year
YEAR_START = 2013 # start year of the data
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

# Set embedding dimension based on model
# The DEFAULT_MODEL_INDEX is defined below but we need to pre-define embedding dimensions
MODEL_DIMENSIONS = {
    'sentence-transformers/all-mpnet-base-v2': 768,
    'BAAI/bge-large-en-v1.5': 1024,
    'openai/text-embedding-ada-002': 1536,
    'meta-llama/Llama-2-7b-chat-hf': 4096,
    'thenlper/gte-large': 1024,
    'intfloat/e5-large-v2': 768,
    'ProsusAI/finbert': 768,
    'yiyanghkust/finbert-tone': 768,
    'nbroad/ESG-BERT': 768,
    'hkunlp/instructor-xl': 1024  # Corrected: instructor-xl uses 1024 dimensions
}

final_parameters = {
    "n_neighbors": 10,                     # Lower value for more local structure = more topics
    "n_components": 50,                  # Keep high dimensionality for preserving structure
    "min_dist": 0.0,                      # Lower value for tighter clusters = more topics
    "min_cluster_size": 20,               # Reduced to allow smaller topics to form
    "min_samples": 10,                     # Lowered to be less strict on what forms a cluster
    "cluster_selection_epsilon": 0.1,    # Keep existing value
}

# HDBSCAN clustering parameters
MIN_SAMPLES = [5]       # Lowered to be less strict on what forms a cluster
MIN_CLUSTER_SIZE = [10] # Reduced to allow smaller topics to form

# Topic modeling target parameters
NR_TOPICS = [100]        # Typically, a reduced, manageable number of final topics

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
    'nbroad/ESG-BERT',                           # Environmental, Social, Governance reports model
    'hkunlp/instructor-xl'                       # High quality instruction-tuned embedding model (768 dimensions)
]

# Default model index to use (0 = original model, 1-9 = enhanced models)
# Change this value to use a different model
DEFAULT_MODEL_INDEX = 9  # Updated to use hkunlp/instructor-xl

# Parameters for Phase 2: Topic Modeling Distillation
MAX_ADAPTIVE_REPRESENTATIVES = 750_000 # Increased from 500k to ensure more representation
MIN_DOCS_PER_TOPIC = 5 # Reduced minimum threshold to allow more diverse topics
MAX_DOCS_PER_TOPIC = 100_000 # More balanced distribution across topics

# Vectorizer parameters for c-TF-IDF computation
MAX_DF = [0.9]         # Retains filtering for terms appearing in over 95% of documents
MIN_DF = [0.02]        # Reduced to include less frequent but potentially important terms

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
  #Financial Performance 
  ["Revenue & Sales", "revenue", "income statement", "sales", "top-line", "total revenue"],
  ["Growth & YOY Trends", "growth", "expansion", "increase", "rise", "escalation", "year over year", "quarter over quarter", "yoy", "qoq"],
  ["Profit & Earnings", "profit", "net income", "bottom-line", "earnings", "net profit"],
  ["Margins", "gross margin", "profit ratio", "markup percentage", "gross profit rate", "sales margin", "operating profit margin", "ebit margin", "operating income margin", "profit margin"],
  ["Return & Investment Efficiency", "operational efficiency", "return on equity", "equity returns", "shareholder return", "net income to equity", "equity performance", "profitability ratio", "return on assets", "asset returns", "asset performance", "asset profitability", "return on investment", "investment returns", "investment performance", "net income to investment", "investment profitability"],
  # Cost Structure & Overhead
  ["Cost Structure & Overhead", "cost", "overhead", "resource costs"],
  # Bottom-Line Items
  ["Expense Items", "expenses", "expenditure", "cost of goods sold", "cogs", "production costs", "manufacturing expenses", "selling, general, and administrative expenses", "sg&a", "non-production costs", "administrative burden"],
  ["Cost Reduction & Efficiency", "cost-cutting initiatives", "expense reduction", "efficiency programs", "budget trimming", "cost optimization", "supply chain optimization", "overhead reduction", "restructuring", "streamlining"],
  ["Raw Material & Commodity Costs", "raw material costs", "input costs", "commodity prices"],
  ["Labor & Payroll", "labor costs", "workforce expenses", "employee costs", "payroll expenses"],
# Operations Cost & Productivity & Efficiency
  ["Automation & Digital", "automation", "robotic process automation", "rpa", "digital transformation", "efficiency"],
  ["Capacity & Resources", "capacity utilization", "resource usage", "production capacity", "facility utilization", "asset efficiency"],
  ["Cost Efficiency & Layoffs", "cost cutting", "cost efficiency", "labor cost", "labor efficiency", "labor productivity", "layoff"],
  ["Process & Productivity", "operational improvements", "process enhancements", "efficiency gains", "operational streamlining", "productivity boosts", "performance upgrades", "productivity metrics", "efficiency measures", "output indicators", "performance ratios", "productivity kpis", "operational effectiveness"],
  ["Supply Chain & Inventory", "supply chain efficiency", "logistics performance", "procurement effectiveness", "distribution efficiency", "supply chain optimization", "inventory turnover", "stock rotation", "inventory efficiency", "stock velocity", "goods turnover rate"],
# Workforce & Human Capital
  ["Headcount & Turnover", "employee headcount", "workforce size", "staff numbers", "personnel count", "headcount management", "employee turnover rate", "staff attrition", "workforce stability", "retention challenges", "employee departures"],
  ["Talent & Retention", "talent acquisition and retention strategies", "hiring initiatives", "employee retention programs", "workforce planning", "talent management"],
  ["Diversity & Engagement", "workforce diversity and inclusion", "diversity metrics", "inclusivity efforts", "equal opportunity initiatives", "employee engagement metrics", "staff satisfaction", "workforce morale", "employee loyalty", "job satisfaction", "team engagement"],
  # Cash & Liquidity
  ["Cash & Liquidity", "cash", "cash flow", "liquidity", "cash position", "cash balance", "liquid assets", "Debt & Liabilities", "debt", "liabilities", "borrowing", "indebtedness", "debt burden", "financial position", "balance sheet", "financial health", "financial stability", "financial standing", "current assets", "quick ratio", "current ratio"],
  ["Equity & Capital Management", "equity", "shareholders", "stockholders", "ownership", "equity holders"],
  ["Dividends & Payouts", "dividend", "dividend payment", "dividend yield", "dividend payout", "dividend policy", "payout policy", "shareholder distributions", "income distribution plan", "yield policy"],
  ["Structure & Leverage", "capital structure", "financial leverage", "loan balances", "bond rating", "debt-to-equity ratio", "equity financing", "credit rating"],
  ["Equity & Capital Management (Buybacks & M&A)", "share buyback", "merge and acquisition", "strategic investment", "share buyback plans", "stock repurchase program", "share repurchases", "buyback initiative", "stock retirement", "equity reduction"],
  # Investments & Expenditures
  ["Investment Activities", "investment", "investing", "investment spending", "Capital Expenditures", "capital expenditure", "capex", "capital expenditure plans", "capex projections", "asset acquisition strategy", "infrastructure spending", "capital outlays"],
  ["Working Capital & Liquidity", "working capital management", "cash flow management", "operational liquidity", "short-term asset management"],
  # Forecast & Outlook
  ["Short-Term & Quarterly", "short-term forecast", "upcoming quarter outlook", "near-term projections", "quarterly expectations", "forward guidance", "Priority & Urgency", "priority", "immediate", "urgent", "short term", "short-term", "short run", "quickly", "quick", "fast", "fastest", "speed", "speed up", "sped up", "speeding up", "rapid", "rapidly", "accelerate", "swift", "prompt", "promptly", "expedite", "expeditiously"],
  ["Long-Term & Annual", "full-year outlook", "annual forecast", "yearly projection", "fiscal year outlook", "long-term", "multi-year goals", "strategic financial objectives", "extended financial outlook", "future financial aims", "investment horizon"],
  # Strategic & Growth
  ["Strategic & Growth", "growth strategy", "strategic horizon", "strategic initiatives", "sustainable growth", "forward-looking", "future potential", "market leadership", "value creation", "future roadmap", "comprehensive view", "down the road", "going forward", "upcoming quarters", "long run", "in future", "enduring value"],
  ["Industry & Economic", "industry forecast", "sector outlook", "market projections", "industry trends", "vertical predictions", "sector expectations", "economic forecast", "macroeconomic outlook", "economic projections", "economic trends", "economic expectations", "consumer spending", "economic crisis", "economic expansion", "economic recession", "fiscal policy", "monetary policy", "inflation", "interest rate", "jobs data", "market volatility", "pandemic", "recovery", "recession", "unemployment rate", "consumer confidence", "federal reserve", "fed"],
  # Risks
  ["Risks", "uncertainty", "possible", "probability", "riskiness", "chance", "likelihood", "volatile", "volatility", "insecure", "dangerous", "unpredictability", "unpredictable", "jeopardy", "hazardous", "precarious", "riskier", "riskiest", "exposure", "variable", "danger"],
  ["Currency & Interest Rate", "foreign exchange impact", "currency effects", "forex exposure", "exchange rate influence", "currency risk", "interest rate sensitivity", "rate exposure", "interest risk", "borrowing cost sensitivity"],
  ["Liquidity & Credit", "liquidity risk", "financial tightness", "credit crunch", "cash reserve", "bankruptcy", "credit risk", "default risk", "credit exposure", "credit quality", "credit portfolio", "counterparty risk"],
  ["Operational risk", "operational risk", "business continuity", "health and safety", "logistics disruption", "incident management", "internal control", "legal risk", "compliance", "lawsuit", "litigation", "anticorruption", "anti-monopoly", "regulator", "settlement"],
  ["Political & Regulatory", "political risk", "political uncertainty", "regulatory reform", "policy goals", "public opinion", "government policy", "sanction", "trade policy", "trade war", "tariff", "election", "geopolitical uncertainty"],
  ["Sovereign & International", "geopolitical risk","war", "russia", "China", "Ukraine", "Tax Tariffs", "sovereign risk", "sovereign debt", "sovereign default", "country downgrade", "sovereign credit rating", "international policy"],
  ["Climate & Natural Disaster", "climate risk", "extreme weather", "drought", "flood", "wildfire", "storm", "tsunami", "hurricane", "natural disaster", "global warming"],
  ["Regulatory Challenges & Disputes", "regulatory challenges", "compliance issues", "legal hurdles", "regulatory environment", "policy challenges", "litigation updates", "legal proceedings", "lawsuit status", "court case developments", "legal disputes"],
  # Market Share & Rivalry
  ["Market Share & Rivalry", "market share", "market dominance", "market leadership", "market position", "business footprint", "competitive landscape", "competitive risk", "competitive environment", "market competition", "competitor analysis", "competitive advantage", "industry rivalry"],
  ["Challenges & Headwinds", "challenges", "headwind", "conservative outlook", "lack of visibility", "mixed results", "expense growth", "falloff", "dead horse", "downtime", "nightmare"],
  ["Brand & Pricing", "brand loyalty", "price competition", "price war", "brand strength", "brand power", "brand health", "brand recognition", "brand equity", "brand value"],
  ["Acquisition & Loyalty", "new customer growth", "client onboarding", "customer wins", "new business generation", "expanding customer base", "client loyalty", "churn rate", "customer stickiness", "repeat business", "customer longevity"],
  ["Satisfaction & Service Quality", "customer satisfaction index", "loyalty metric", "referral likelihood", "customer advocacy", "satisfaction score", "service quality", "customer satisfaction measures", "service performance indicators", "quality assurance metrics", "service level achievements", "customer experience scores"],
  ["Product Launches & Mix", "new product launches", "product releases", "new offerings", "product introductions", "market debuts", "new solutions", "product mix changes", "product portfolio shifts", "offering diversification", "product line adjustments", "sales mix"],
  ["Sales Pipeline & Marketing", "sales pipeline", "sales funnel", "prospect pipeline", "revenue pipeline", "deal flow", "backlog or order book status", "unfilled orders", "work in progress", "future revenue", "committed sales", "customer acquisition costs", "cac", "cost per customer", "marketing efficiency", "acquisition spend"],
  ["Lifetime Value & Productivity", "lifetime value of customers", "ltv", "customer worth", "long-term customer value", "client profitability", "marketing effectiveness", "roi on marketing", "campaign performance", "promotional impact", "advertising effectiveness", "sales force productivity", "sales efficiency", "rep performance", "sales team effectiveness", "selling productivity"],
  ["Benchmarking & Positioning", "industry benchmarking", "peer comparison", "competitive benchmarking", "market positioning", "sector performance ranking"],
  # Segment Reporting
  ["Segment Reporting", "business unit breakdowns", "divisional performance", "segment analysis", "unit-level results", "departmental breakdown"],
  ["Geographic Segment Reporting", "geographic segment performance", "regional results", "territorial analysis", "location-based performance"],
  ["Product & Offerings Segment Reporting", "product category performance", "product line results", "offering performance", "product mix analysis"],
  ["Customer & Demographics Segment Reporting", "customer segment analysis", "client group performance", "demographic performance", "target market results"],
  # Technology & Innovation
  ["Technology & Innovation", "research_and_development", "r&d spending", "innovation funding", "product development costs", "research expenditure", "technology investments", "innovation pipeline", "development roadmap", "future products", "corporate innovation", "innovation", "r&d", "breakthrough technologies"],
  ["Roadmap & Upgrades", "product_roadmap", "development timeline", "product strategy", "future releases", "product evolution plan", "feature roadmap", "digital transformation initiatives", "digitalization efforts", "tech modernization", "it transformation", "technology upgrade", "it infrastructure investments", "tech spending", "system upgrades", "it capex", "technology infrastructure"],
  ["E-commerce & Data", "e-commerce performance", "online sales", "digital revenue", "internet retail performance", "web store results", "data analytics capabilities", "business intelligence", "data-driven insights", "analytics infrastructure", "predictive modeling"],
  ["AI & Machine Learning", "artificial intelligence and machine learning applications", "ai integration", "ml implementation", "cognitive computing", "smart algorithms", "transformer", "chatgpt", "large language model", "neural networks", "reinforcement learning"],
  ["Intellectual Property", "patent portfolio", "ip assets", "patent holdings", "invention rights", "proprietary technology", "trademark developments", "brand protection", "trademark portfolio", "intellectual property rights", "licensing agreements", "ip licensing", "technology transfer", "patent licensing", "trademark licensing", "ip litigation", "patent disputes", "trademark infringement", "copyright cases", "intellectual property lawsuits"],
  ["Autonomous Driving", "autonomous driving", "self-driving cars", "autonomous vehicles", "3D printing", "automation, robotics", "RPA", "robotic process automation", "artificial intelligence", "machine learning", "process automation", "digital transformation", "digitalization", "digital process automation", "autonomous drive", "autonomous vehicle", "autonomous system"],
  # ESG
  ["Extreme Weather & Climate Change", "extreme weather", "climate change", "weather events", "weather conditions", "weather patterns", "weather forecasts", "weather impact", "weather risk", "weather disasters", "weather emergencies", "weather alerts", "weather warnings", "weather emergencies", "extreme weather", "drought", "flood", "wildfire", "storm", "tsunami", "hurricane", "natural disaster", "global warming"],
  ["Environmental: Initiatives & Carbon", "environmental initiatives", "eco-friendly programs", "green initiatives", "sustainability efforts", "carbon emissions", "renewable energy", "carbon footprint", "carbon footprint reduction efforts", "emissions reduction", "climate impact mitigation", "greenhouse gas reduction"],
  ["Social Responsibility", "social responsibility programs", "community initiatives", "social impact", "philanthropic efforts", "corporate citizenship"],
  ["Governance & Ethics", "governance practices", "corporate governance", "board practices", "ethical leadership", "shareholder rights", "management oversight", "sustainable sourcing", "ethical procurement", "responsible sourcing", "supply chain sustainability", "eco-friendly suppliers"],
  # M&A & Corporate Strategy
  ["M&A & Corporate Strategy", "M&A & Consolidation", "merger and acquisition activities", "m&a strategy", "corporate takeovers", "business combinations", "acquisition plans", "consolidation efforts"],
  ["Expansion & Diversification", "diversification efforts", "business expansion", "new venture development", "portfolio diversification", "risk spreading"],
  ["Market Entry & Partnerships", "new market entry", "geographic expansion", "market entry", "territorial growth", "global reach expansion", "regional diversification", "partnerships and collaborations", "strategic alliances", "joint ventures", "cooperative agreements", "business partnerships", "collaborative initiatives"],
  # Other
  ["Tax & Credits", "corporate tax", "effective tax rate", "tax liabilities", "tax planning", "tax credits", "deferred taxes"],
  ["Impairments & Write-offs", "allowance", "write-off", "impairment charge", "asset impairment", "goodwill impairment"],
]

