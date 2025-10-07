# 🧠 User Narrative Generator
---

## 📘 Overview

Traditional conversational recommender systems (CRS) often fail to capture the subtle and evolving preferences of users because they rely heavily on structured data (ratings, item features).  
This project introduces a **narrative-enriched CRS** that jointly trains **Collaborative Filtering (CF)** and **Semantic Encoders** to generate **context-aware user narratives** (e.g., purchase reasons, explanations, and summaries) using the **REGEN dataset**.

Our model integrates:
- Structured purchase history  
- User reviews and profiles  
- Semantic embeddings  
- Large Language Model fine-tuning (Flan-T5 + LoRA)  

This hybrid framework produces personalized, semantically rich narratives to support next-generation recommendation systems.

---


This fusion framework enables **joint training** of structured and unstructured information, ensuring the generated narratives reflect both **quantitative** (interaction patterns) and **qualitative** (user intent) aspects.

---

## 📊 Dataset

We use the **REGEN dataset** ([Sayana et al., 2024](https://arxiv.org/abs/2410.16780)), derived from the 2018 Amazon Reviews dataset.

| Dataset | Reviewers | Features Used |
|----------|------------|----------------|
| Clothing, Shoes & Jewelry | 1,117,912 | Titles, categories, price, reviews, user summaries |
| Office Products | 100,613 | Titles, price, explanations, purchase reasons |

Each user has purchase history, metadata, and generated summaries.  
This repository uses a **subset of 10,000 users** per domain for computational feasibility.

---

## ⚙️ Installation

```bash
# 1️⃣ Clone this repository
git clone https://github.com/Shreya-0309/User-Narrative-Generator.git
cd User-Narrative-Generator

# 2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate    # (Linux/Mac)
venv\Scripts\activate       # (Windows)

# 3️⃣ Install dependencies
pip install -r requirements.txt
