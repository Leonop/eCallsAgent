import os

# Directory locations
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
data_folder = os.path.join(PROJECT_DIR, "input_data", "external")
models_folder = os.path.join(PROJECT_DIR, "eCallsAgent", "output", "models")
output_folder = os.path.join(PROJECT_DIR, "eCallsAgent", "output")
output_fig_folder = os.path.join(output_folder, "figures")
temp_folder = os.path.join(output_folder, "temp")
data_filename = 'ecc_transcripts_2006_2020.csv'
input_folder = os.path.join(PROJECT_DIR, "eCallsAgent", "input_data")

# Create required directories
for directory in [data_folder, models_folder, output_folder, output_fig_folder, temp_folder]:
    os.makedirs(directory, exist_ok=True)

# File paths
data_filename = 'eCallsAgent'  # Base filename for the data
figure_base_name = f'bertopic_{data_filename}'  # Base name for figure files

# Temporary file paths
TEMP_EMBEDDINGS = os.path.join(temp_folder, f'{data_filename}_embeddings.mmap')
TEMP_TOPIC_KEYWORDS = os.path.join(temp_folder, f'{data_filename}_topic_keywords.pkl')
TEMP_TOPIC_LABELS = os.path.join(temp_folder, f'{data_filename}_topic_labels.json')
PREPROCESSED_DOCS = os.path.join(output_folder, f'preprocessed_docs_{data_filename}.txt')

# Model output paths
MODEL_SCORES = os.path.join(output_folder, "model_scores.txt")
MODEL_SELECTION_RESULTS = os.path.join(output_folder, "model_selection_results.csv")
TOPIC_SCATTER_PLOT = os.path.join(output_fig_folder, "topic_scatter_plot.pdf") 