# AskMyDocs – LLM-Powered RAG Document QA System

AskMyDocs is a high-performance Retrieval-Augmented Generation (RAG) system designed for private document question-answering. It allows users to query their PDF collections without the need for LLM fine-tuning, ensuring data privacy and rapid deployment.

## 🚀 Key Features

- **Document Processing Pipeline**: End-to-end flow from PDF ingestion to text chunking and vector embeddings.
- **Advanced Chunking Strategies**: Supported fixed chunking (Recursive) and direct token splitting (500/1000/1500 tokens) to optimize between context quality and query speed.
- **Hallucination Control**: Optimized prompt engineering and temperature adjustment (0.3 vs 0.7) resulting in a significantly reduced hallucination rate (~40%).
- **Semantic Retrieval**: Uses ChromaDB and OpenAI Embeddings for precise document retrieval based on semantic meaning.
- **Premium UI**: Built with Streamlit for a seamless, dark-themed user experience.

## 🛠 Tech Stack

- **Framework**: LangChain
- **LLM**: GPT-4 (OpenAI)
- **Vector Store**: ChromaDB
- **Embeddings**: OpenAI Embeddings
- **Frontend**: Streamlit
- **PDF Processing**: PyPDF

## 📦 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/AskMyDocs.git
cd AskMyDocs
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
streamlit run app.py
```

## 🧪 Experiments and Findings

One of the core focuses of this project was balancing context quality with speed. 

- **Chunking strategies**: Testing fixed sizes (500, 1000, 1500 tokens) revealed that 1000 tokens often provided the best balance for technical documents, whereas 500 tokens were faster but occasionally fragmented dense context.
- **Temperature control**: During testing, reducing the LLM temperature from 0.7 to 0.3 reduced hallucination by ~40%, making the system more reliable for factual Q&A.

## 📄 License
MIT License
