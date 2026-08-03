# Cricketers Self-Query RAG

A Retrieval-Augmented Generation (RAG) application built on the **World_Cricketers.xlsx** dataset.

Unlike a normal RAG system that only performs semantic search, this project uses **Self-Query Retrieval**, allowing it to answer structured questions such as:

- Show all Australian batsmen
- How many Pakistani all-rounders are there?
- Players active in the 2000s
- Left-arm fast bowlers

The system combines **metadata filtering** with **semantic search** to provide more accurate list and count answers.


# Submission Details

**Team Name [number]:** Sheesh[14]

**PS ID:** 9 

**Demo Video (Google Drive):**
`________________`


# Technologies Used

- Python
- Streamlit
- LangChain
- Groq (Llama 3)
- Qdrant (Local Vector Database)
- HuggingFace Embeddings (all-MiniLM-L6-v2)

## Project Structure

```text
BuildWithRAG/
│
├── data/
│   └── World_Cricketers.xlsx
│
├── src/
│   ├── app.py
│   ├── ingestion.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── pipeline.py
│   └── monitoring.py
│
├── qdrant_data/
├── requirements.txt
├── .env.example
└── README.md
```

# Setup

## 1. Create a virtual environment

```bash
py -3.11 -m venv .venv
```

Activate it.

Install dependencies.

```bash
pip install -r requirements.txt
```

---

## 2. Configure API Key

Copy

```text
.env.example
```

to

```text
.env
```

and add Groq API key.

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

---

## 3. Build the Vector Database

Run

```bash
python -m src.ingestion
```

This creates the local Qdrant database from the provided dataset.

---

## 4. Start the Application

```bash
streamlit run src/app.py
```

Open

```
http://localhost:8501
```

---

# Local Vector Database

This project uses **Qdrant in Local Embedded Mode**.

The repository includes:

- Source dataset (`data/World_Cricketers.xlsx`)
- Script to regenerate the vector database (`python -m src.ingestion`)

**Choose ONE option before submitting:**

### Option A 

☐ The generated `qdrant_data/` folder is included in this repository.

The application can be run directly.

### Option B

☐ `qdrant_data/` is **not** included.

it should be run

```bash
python -m src.ingestion
```

to regenerate the vector database.

---

# API Key

This project requires a Groq API key.

```
GROQ_API_KEY= (1gsk_BKGzgwG2GVrUBGVZcha2WGdyb3FYyvvcNSHFHYyRmRw7sNrKvbSh01)(please remove the 1's before and after the key while using it)
```

# Features

- Self-Query Retrieval
- Metadata Filtering
- Semantic Search
- Hybrid Retrieval
- Local Vector Database
- Out-of-Scope Detection
- Naive RAG Comparison

# Sample Questions

- Show all Australian batsmen
- List left-arm fast bowlers
- How many Pakistani all-rounders are there?
- Players active in 2024
- Players active before 1950
- Who is known for aggressive batting?


# Notes

This project was built using only the provided **World_Cricketers.xlsx** dataset as required in the problem statement. Any information not available in it is considered out of scope.