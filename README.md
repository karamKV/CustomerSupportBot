# Customer Support System

A comprehensive search system that combines web crawling, semantic processing, and hybrid retrieval methods for efficient enterprise-scale search operation that ensure **effective answer with Citations** , **Guardrails** and **ChitchatAnswers**

## 🚀 Features

- Advanced web crawling with Crawlbase
- Hierarchical content structuring
- Semantic processing and embeddings
- Hybrid retrieval system (Dense + Sparse)
- Real-time performance monitoring
- Interactive UI with Streamlit
- Comprehensive logging and benchmarking

## 🏗️ System Architecture

### Data Collection & Processing
- Web crawling using Crawlbase
- Hierarchical content organization
- Automated metadata extraction
- Contextualized chunking and structuring

### Search & Retrieval
- Dense retrieval using embeddings
- Sparse retrieval for keyword matching
- FAISS vector database integration
- Hybrid retrieval optimization

## 🛠️ Installation

```bash
# Clone the repository
https://github.com/karamKV/CustomerSupportBot.git

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

## 🚦 Usage

### Starting the Application

```bash
# Start the Streamlit interface
streamlit run startMLFlow.py

# Start the API server
python main.py

# Start the Relevance server
python relevanceGroundedness.py

```

## 📊 User Interface

### Tab 1: Analytics Dashboard
- System performance metrics
- Usage statistics
- Real-time monitoring

### Tab 2: Search Interface
- Query input
- Results display
- Metadata visualization

### Tab 3: Query Logs
- Historical queries
- Accuracy metrics
- Performance analysis

## 📈 Monitoring & Evaluation

- MLflow integration for experiment tracking
- Groundedness validation
- Relevance scoring
- Performance benchmarkin

## 📝 Logging

All system activities are logged using MLflow, including:
- Query performance
- Retrieval accuracy
- System metrics


## 📄 License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details


## 🙏 Acknowledgments

- Crawlbase for web crawling capabilities
- Facebook AI for FAISS
- Streamlit team for the UI framework
- MLflow contributors

---
Built with ❤️ by [Karamvir Singh]