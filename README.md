# CareerCompass – NLP-Based Job Recommendation System

CareerCompass is an NLP-based job recommendation system that matches resumes with relevant job postings using keyword-based and semantic similarity techniques.

The system combines TF-IDF, a fine-tuned Sentence-BERT model (Career-SBERT), K-Means clustering, hybrid ranking, and skill-gap analysis. An interactive Streamlit application allows users to upload a resume and receive personalized job recommendations.

## Project Overview

The project uses the LinkedIn Job Postings dataset containing **123,849 job postings**.

The workflow includes:

- Data preprocessing and cleaning
- TF-IDF-based keyword similarity
- Sentence-BERT semantic embeddings
- Domain-adaptive fine-tuning of Sentence-BERT on recruitment-related text
- K-Means clustering of job postings
- Hybrid recommendation using TF-IDF and Career-SBERT
- Skill-gap analysis
- Interactive Streamlit deployment

The modeling pipeline uses a **10,000-job sample** selected from the cleaned dataset for experimentation and recommendation modeling.

## Methodology

### 1. Data Preprocessing

The job posting dataset is cleaned and prepared for NLP processing.

The preprocessing workflow includes:

- Missing-value handling
- Text cleaning
- Removal of irrelevant text
- Tokenization and normalization
- Stop-word removal
- Lemmatization
- Preparation of job-related text for NLP modeling

### 2. TF-IDF Baseline

TF-IDF is used as a keyword-based baseline for measuring similarity between resume text and job postings.

The implementation uses:

- Maximum 5,000 features
- Unigrams and bigrams
- English stop-word removal
- Cosine similarity for ranking

### 3. Career-SBERT

Sentence-BERT is used to capture semantic similarity between resumes and job postings.

The base model used is:
all-MiniLM-L6-v2

The model produces 384-dimensional embeddings.
A domain-adaptive fine-tuning step is applied to recruitment-related text to create the Career-SBERT model.
### 4. K-Means Clustering
K-Means clustering is applied to the job embeddings to group job postings based on semantic similarity.
The predicted career cluster of the uploaded resume is used to narrow the recommendation space before ranking the jobs.
### 5. Hybrid Recommendation
The final recommendation score combines:
40% TF-IDF similarity
60% Career-SBERT semantic similarity

This combines keyword-level matching with semantic understanding.
### 6. Skill-Gap Analysis
CareerCompass compares skills identified in the resume with skills detected in recommended job descriptions.
For each recommendation, the application displays:
- Skills already detected in the resume
- Skills that may be worth adding or developing
### 7. Streamlit Application
The complete recommendation workflow is deployed through Streamlit.
Users can:
1. Upload a PDF or TXT resume
2. Extract the resume text
3. Generate job recommendations
4. View hybrid match scores
5. Compare TF-IDF, Career-SBERT, and hybrid recommendations
6. Review detected skill gaps
7. Open the original job posting

The dataset, processed datasets, trained models, embeddings, and NLTK resources are excluded from version control because of their size and are generated or stored locally.
### Dataset
The project uses the LinkedIn Job Postings dataset.
Dataset source:
https://www.kaggle.com/datasets/arshkon/linkedin-job-postings
Place the downloaded dataset at:
data/postings.csv

The dataset is not included in this repository.