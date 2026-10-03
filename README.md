## Setup Instructions

1. Clone this repository.

2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Download the LinkedIn Job Postings dataset (link below).

4. Place the dataset in the project folder.

5. Run:

```bash
jupyter notebook experimentation.ipynb
```

This notebook preprocesses the data, fine-tunes the SBERT model, creates embeddings, and generates the files required by the Streamlit application.

6. Start the application:

```bash
streamlit run streamlit_app.py
```