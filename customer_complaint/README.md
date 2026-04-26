# Automated Customer Complaint Analyzer

> End-to-end NLP pipeline using OpenAI fine-tuning to transform unstructured customer complaints into structured, actionable insights.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenAI](https://img.shields.io/badge/OpenAI-Fine--Tuning-412991)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

Businesses receive large volumes of customer complaints in unstructured text. This project builds a domain-specific LLM that automatically extracts three structured fields from free-form complaint text:

| Field | Description | Example |
|---|---|---|
| `Topic` | Area of issue | `Billing`, `Internet`, `TV` |
| `Problem` | Short description | `"Missing channels"` |
| `Customer_Dissatisfaction_Index` | Severity score (0–100) | `85` |

---

## Example

**Input:**
```
TV channels keep disappearing from my subscription! Extremely annoyed!
```

**Output:**
```json
{
  "Topic": "TV",
  "Problem": "Missing channels",
  "Customer_Dissatisfaction_Index": 85
}
```

---

## Pipeline

```
Raw CSV Dataset
      ↓
Data Preprocessing
      ↓
JSONL Conversion (Chat Format)
      ↓
Upload to OpenAI API
      ↓
Fine-Tuning Job
      ↓
Loss Curve Evaluation
      ↓
Inference & Prediction
```

---

## Tech Stack

- **Language:** Python 3.8+
- **ML / API:** OpenAI API (fine-tuning + inference)
- **Data:** Pandas
- **Visualization:** Matplotlib
- **Config:** python-dotenv
- **Environment:** Google Colab / Jupyter Notebook

---

## Project Structure

```
project/
├── Customer Complaints.csv     # Source dataset
├── training_data.jsonl         # Processed training file (chat format)
├── notebook.ipynb              # Main implementation notebook
├── README.md                   # Project documentation
└── .env                        # API key — never commit this
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/project-name.git
cd project-name
```

### 2. Install dependencies

```bash
pip install openai python-dotenv pandas matplotlib
```

### 3. Configure your API key

Create a `.env` file in the project root:

```
MY_API_KEY=your_openai_api_key_here
```

> **Note:** OpenAI billing must be enabled on your account to use fine-tuning.

---

## How It Works

### 1. Data Preparation
Each row from the CSV is converted into OpenAI's chat-based JSONL format — a system prompt defining the extraction task, paired with a user message (the complaint) and an assistant response (the structured JSON output).

### 2. Fine-Tuning
The formatted dataset is uploaded to OpenAI and a fine-tuning job is initiated via the API. Training runs on OpenAI's cloud infrastructure — no local GPU required.

### 3. Evaluation
Training and validation loss are extracted from the fine-tuning job and plotted to assess learning progress and detect overfitting.

### 4. Inference
The fine-tuned model is called via the standard completions API to produce structured JSON from new complaint inputs.

---

## Results & Limitations

### Strengths
- Produces consistent, structured JSON output
- Good performance on complaint patterns well-represented in training data

### Limitations
- Prone to overfitting on small datasets (< 100 samples)
- Generalizes poorly to complaint types not seen during training
- Requires sufficient, high-quality labeled data for best results

---

## Fine-Tuning vs. Prompt Engineering

| Criteria | Fine-Tuning | Prompt Engineering |
|---|---|---|
| Task-specific accuracy | High | Moderate |
| Flexibility | Low | High |
| Cost | Higher | Lower |
| Iteration speed | Slow | Fast |
| Recommended when | You have labeled data and a stable, well-defined task | You're exploring or need fast iteration |

---

## Key Takeaways

- Fine-tuning improves domain specialization — it does not make the model more intelligent
- Data quality matters more than dataset size; noisy labels produce noisy outputs
- Prompt engineering is worth exhausting before committing to fine-tuning
- Always evaluate on held-out data — training loss alone is not sufficient evidence of generalization

---

## Future Work

- Expand dataset to 200+ diverse complaint samples
- Experiment with data augmentation to improve coverage
- Evaluate combined prompt engineering + fine-tuning approaches
- Add automated evaluation metrics (e.g., JSON parse rate, field-level accuracy)

---

## Important Notes

- Fine-tuning jobs are billed through your OpenAI account
- Model training runs entirely on OpenAI infrastructure — not locally
- Never commit your `.env` file or API key to version control
