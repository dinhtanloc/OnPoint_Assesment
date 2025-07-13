# 📘 Mini Hackathon: OnPoint Knowlede Portal - Search Engine based on dataset


## 1. Problem
The Knowledge Portal:
* Background:
    - OnPoint is serving up to 300 brands, and each of them has a wide range of documents. Most of them are available in PDF, DOCX, or XLSX format.
    - A customer service agent spends a lot of time looking for the correct information about a product (e.g., ingredients, instructions, warnings) when supporting a customer.
* Problem Statement:
- Can we build a tool to help users quickly, efficiently, accurately, and securely look up the information?
- Input: a wide range of files, but can start with an Excel file
- Output: specific product information
**Evaluation criteria**:
- System design: a system that enables end-users (Customer Service team) to manage, use, and update their knowledge base
- User experience (UI, UX)
**Technical aspects**:
    - Search accuracy for top 3 results (Precision@3): 100%
    - Relevancy (e.g., Mean Reciprocal Rank - MRR): > 50%
    - Response time: < 200 ms
    - Load capacity: > 10,000 products without memory and performance issues.
## 2. Introduction

This project is a **Streamlit-based web application** designed for interactive data interaction and visualization. It uses modern Python tooling and ensures a consistent environment setup via `uv`, a high-performance Python package manager. The system is optimized for rapid local development and testing.

### Requirements
- Python 3.10 (strictly required)
- Git
- `uv` (for dependency syncing)
- Streamlit

## 3. Quick Install

Follow these steps to get the project up and running:

### Step 1: Clone the repository
```
git clone https://github.com/dinhtanloc/OnPoint_Assesment.git
```

### Step 2: Change into the project directory
```
cd OnPoint_Assesment
```
### Step 3: Create a virtual environment (Python 3.10 required)
```
python3.10 -m venv .venv
```

# On Windows:
```
py -3.10 -m venv .venv
```
### Step 4: Activate the virtual environment

# On macOS/Linux:
```
source .venv/bin/activate
```
# On Windows:
```
.venv\Scripts\activate
```
### Step 5: Sync dependencies using `uv`
```
pip install uv
uv pip sync uv.lock
```
### Step 6: Run the Streamlit 
```
streamlit run main.py
```
## 4. Demo

# Note:
# Please wait a few minutes for the database to initialize on first run.

---
✅ You're now ready to use the application!
