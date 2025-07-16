# Data Alchemist

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

Data Alchemist is an interactive data science tool built with Streamlit, LangGraph, LangChain, and Model Context Protocol (MCP). It enables users to upload CSV or Excel files, perform data preprocessing, analysis, and simple machine learning tasks through a conversational AI agent. The tool leverages Polars for efficient data handling, scikit-learn for modeling, and MCP servers for modular tool integration, providing a seamless workflow for data exploration and modeling.

## Features

- **Data Loading and Inspection**: Upload CSV or Excel files, inspect data (head, schema, missing values, duplicates, unique counts).
- **Preprocessing Tools**: Impute missing values (mean/median/mode), convert columns to numeric, encode categorical features (one-hot/ordinal), detect/handle outliers (IQR/Z-score), scale features (standard/minmax/robust), drop columns.
- **Data Splitting and Modeling**: Split data into train/test sets (simple train-test-split or K-Fold Cross Validation), train linear regression, logistic regression, or random forest (regression or classification) models, evaluate performance (R², MAE, accuracy, etc.).
- **Session Management**: Persistent sessions with reset functionality; save processed data to new CSV files.
- **Interactive UI**: Chat-based interface for natural language queries; separate data viewer page for visualizing DataFrames.
- **Modular Architecture**: MCP servers for tool extensibility; LangGraph for agent orchestration.

## Develop Roadmap
- **Immediate Plans:** Robust data preprocessing suite, more model options (e.g., SVM, K-Means Clustering, Decision Trees), ensemble methods, more data visualization tools, Retrieval Augmented Generated, Knowledge Graphs
- **Intermediate Plans:** Full analysis and model report generation, tools for neural networks (TensorFlow/Pytorch), ability to run simulations on data, database implementation with SQL query tools
- **Future Plans:** User-interface overhaul with interactive features, multi-agent system for different stages of workflow

## Installation

### Prerequisites
- Python 3.12+
- Git

### Steps
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/DataAlchemist-MCP.git
   cd DataAlchemist-MCP
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

   *Note*: The project uses libraries like `streamlit`, `polars`, `scikit-learn`, `langgraph`, `langchain`, `joblib`, and `dotenv`. Ensure `MCP_FILESYSTEM_DIR` is set in `.env` for data storage (defaults to `./data`).

4. Set up environment variables:
   Create a `.env` file in the root directory:
   ```
   MCP_FILESYSTEM_DIR=./data
   GOOGLE_API_KEY=your_google_api_key  # For Gemini LLM in graph.py
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. **Main Page (Data Alchemist)**:
   - Upload a CSV file via the file uploader.
   - Interact with the AI agent via chat input (e.g., "Inspect the data" or "Impute missing values in column X with mean").
   - Use commands like "Reset session" to clear state.
   - Save processed data with "Save to processed_data.csv".

3. **Data Viewer Page**:
   - View current DataFrames (full data, train/test splits) with options to show head or full views.

4. **Example Workflow**:
   - Upload `example.csv`.
   - Chat: "Load data from example.csv".
   - Chat: "Inspect data with 10 rows".
   - Preprocess: "Encode categorical features: {'column1': 'one-hot'}".
   - Model: "Split data with target 'label', then train linear regression".
   - Save: "Save to output.csv".

The MCP server (`data_alchemist.py`) runs in the background for tool execution. Sessions persist via Parquet files and joblib.

## Project Structure

```
DataAlchemist-MCP/
├── app.py                  # Streamlit entry point
├── ui_pages/
│   ├── main_page.py        # Chat interface and agent interaction
│   └── data_viewer.py      # DataFrame visualization page
│   └── visualization_page.py # Visualization (e.g., distribution plots) page
├── src/
│   ├── client.py           # Graph response streaming
│   ├── graph.py            # LangGraph agent graph builder
│   └── mcp_servers/
│       ├── config.py       # MCP config loader
│       ├── data_alchemist.py  # MCP server with data tools
│       └── mcp_config.json # MCP server configuration
├── .env.example            # Sample env file
├── requirements.txt        # Dependencies
├── README.md               # This file
└── LICENSE                 # MIT License
```

## Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

Ensure code adheres to PEP8 standards and includes tests where applicable.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io/), [LangGraph](https://langchain-ai.github.io/langgraph/), [Polars](https://pola.rs/), and [MCP](https://modelcontextprotocol.io/introduction).
- Inspired by interactive data science workflows for AI-assisted analysis.
