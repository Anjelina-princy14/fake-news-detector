

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange?logo=scikitlearn)
![Status](https://img.shields.io/badge/Status-Deployed-success)
![Repo Size](https://img.shields.io/github/repo-size/Anjelina-princy14/fake-news-detector)
![Last Commit](https://img.shields.io/github/last-commit/Anjelina-princy14/fake-news-detector)
[![Streamlit App](https://img.shields.io/badge/Live_App-Streamlit-success?logo=streamlit)](https://anjelina-fake-news-detector.streamlit.app/)

 
 
📰 Fake News Detector

A simple machine learning web app that can classify news headlines or short articles as **REAL** or **FAKE**, with confidence scores and a small explanation of *why* it thinks so.

🔗 **Live Demo:** https://anjelina-fake-news-detector.streamlit.app/

---

## 🚀 Features

- Type any news headline or short article
- Classifies it as **REAL** or **FAKE**
- Shows **confidence scores** for both classes
- Explains the prediction using important words that pushed it towards REAL or FAKE
- Clean **dark UI** built with Streamlit and custom CSS

---

🧠 Tech Stack

- **Language:** Python  
- **Web Framework:** Streamlit  
- **ML / NLP:**
  - scikit-learn
  - Logistic Regression
  - TF-IDF Vectorizer
- **Others:** NumPy, pandas, joblib

---

📂 Project Structure

```text
fake-news-detector/
├── app.py                   # Streamlit UI (main web app)
├── train_model.py           # Script to train the model
├── prepare_dataset.py       # Script to merge & label datasets
├── requirements.txt         # Project dependencies
├── data/
│   ├── Fake.csv             # Fake news dataset
│   ├── True.csv             # Real news dataset
│   └── news.csv             # Final combined + cleaned dataset
└── models/
    ├── fake_news_model.pkl  # Trained Logistic Regression model
    └── tfidf_vectorizer.pkl # Trained TF-IDF vectorizer
````

---

⚙️ How It Works

1. **Dataset Preparation (`prepare_dataset.py`)**

   * Reads `Fake.csv` and `True.csv`
   * Adds a `label` column:

     * `"FAKE"` → for fake news rows
     * `"REAL"` → for true news rows
   * Combines them into a single file `data/news.csv`
   * Optionally balances the dataset to avoid bias

2. **Model Training (`train_model.py`)**

   * Loads `data/news.csv`
   * Uses the **`text`** column as input and **`label`** as output
   * Converts text into numerical features using **TF-IDF**
   * Trains a **Logistic Regression** classifier
   * Evaluates accuracy on a test split
   * Saves the model and vectorizer as:

     * `models/fake_news_model.pkl`
     * `models/tfidf_vectorizer.pkl`

3. **Web App (`app.py`)**

   * Loads the saved model and vectorizer
   * User types a headline/article into a text box
   * Text → TF-IDF → model → prediction (`REAL` or `FAKE`)
   * Displays:

     * Final prediction
     * Confidence REAL vs FAKE
   * Uses model coefficients to highlight words that pushed the prediction towards REAL or FAKE

---

🖥️ Running Locally

 1. Clone the repository

```bash
git clone https://github.com/Anjelina-princy14/fake-news-detector.git
cd fake-news-detector
```

2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux / macOS
```

 3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Prepare the dataset (only needed first time)

Make sure `data/Fake.csv` and `data/True.csv` exist, then run:

```bash
python prepare_dataset.py
```

This will create `data/news.csv`.

 5. Train the model (only needed if you retrain)

```bash
python train_model.py
```

This will create:

* `models/fake_news_model.pkl`
* `models/tfidf_vectorizer.pkl`

6. Run the Streamlit app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

---

🌐 Deployment (Streamlit Cloud)

This app is deployed using **Streamlit Community Cloud**:

1. Push the project to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Click **New app**
4. Select:

   * **Repo:** `Anjelina-princy14/fake-news-detector`
   * **Branch:** `main`
   * **Main file path:** `app.py`
5. Click **Deploy**

Streamlit will read `requirements.txt`, install the libraries, and start the app online.

---

 🔍 Example 

**Example 1 – Real-style headline**

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/e60c61c6-b71a-4859-80c4-5f320d610e81" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/22316538-496f-4fa3-90ed-b2309939bdfc" />

**Example 2 – Fake-style headline**

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/deaabb99-c102-4cd0-b110-183995b9570e" />
<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3d7ac397-4b57-423c-aa09-0def8c8f44f1" />


---
⚠️ Limitations

* The model is trained on a fixed dataset and **can make mistakes**.
* Some real headlines may be classified as FAKE and vice versa.
* It does **not** verify facts on the internet; it only learns patterns from the training data.
* Best suited for **educational / demo** use, not for real-world journalism decisions.

---

🚧 Future Improvements

* Use a larger and more diverse dataset
* Try advanced models (e.g., LSTM, BERT)
* Add support for full news articles, not just headlines
* Show more detailed explanations and highlight words in color
* Add language selection / multi-language support

---

🧑‍💻 Author

**Anjelina Princy A**
B.E CSE Student | Aspiring Data Analyst & Python Developer
