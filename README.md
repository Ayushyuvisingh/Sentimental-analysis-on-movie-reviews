# Sentiment Analysis on Movie Reviews

## 🚀 Project Overview

[cite_start]This project is a machine learning model that performs sentiment analysis on movie reviews, automatically classifying them as either **Positive** or **Negative**[cite: 5]. [cite_start]The goal is to understand audience sentiment from text, which is valuable for filmmakers, studios, and review platforms[cite: 10].

> [cite_start]This project aims to develop a machine learning model for sentiment analysis on movie reviews, automatically classifying them as positive or negative based on the expressed sentiment. [cite: 5]

[cite_start]The end-to-end pipeline involves collecting data, preprocessing text to remove noise, extracting numerical features using TF-IDF, training a Logistic Regression model, and finally evaluating its performance[cite: 6, 8, 10].

## ✨ Key Features

* **Sentiment Classification**: Classifies text into 'Positive' or 'Negative' categories.
* **Text Preprocessing**: A robust pipeline that cleans raw text by removing HTML tags, punctuation, and stopwords.
* **Feature Extraction**: Uses `TfidfVectorizer` to convert text data into a meaningful numerical format for the model.
* **High Performance**: Achieves approximately **89% accuracy** on the test dataset.

## 🛠️ Technologies & Libraries Used

* **Python 3**
* **Pandas**: For data manipulation and loading the CSV file.
* **NLTK (Natural Language Toolkit)**: For text cleaning, tokenization, and stopword removal.
* **Scikit-learn**: For feature extraction (TF-IDF), model training (Logistic Regression), and performance evaluation.

## 💾 Dataset

The project uses the **IMDB Dataset of 50K Movie Reviews**, which is a balanced dataset containing 25,000 positive and 25,000 negative reviews.

You can download the dataset from Kaggle: [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## ⚙️ Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Install the required libraries:**
    ```bash
    pip install pandas scikit-learn nltk
    ```

3.  **Download NLTK data:**
    Run the following command in your terminal or execute it within a Python script to download the necessary packages for tokenization and stopwords.
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    ```

4.  **Place the dataset** in the root directory of the project and ensure it is named `IMDB Dataset.csv`.

## ▶️ How to Run

Execute the main script from your terminal:

```bash
python your_script_name.py
```

The script will load the data, train the model, and print the performance evaluation results to the console.

## 📊 Model Performance

[cite_start]The Logistic Regression model was evaluated using **Accuracy** and **F1-Score**, achieving strong results on the unseen test data[cite: 10].

* **Accuracy**: `0.8924` (89.2%)
* **F1-Score**: `0.8947` (89.5%)

### Classification Report

| Class    | Precision | Recall | F1-Score | Support |
| :------- | :-------- | :----- | :------- | :------ |
| Negative | 0.90      | 0.88   | 0.89     | 4961    |
| Positive | 0.88      | 0.91   | 0.89     | 5039    |
| **Total**| **0.89** | **0.89** | **0.89** | **10000**|


## 🎬 Example Predictions

The model can accurately predict the sentiment of new, unseen reviews:

```python
# Positive Review Example
new_review_1 = "This was an absolutely fantastic movie. The acting was superb and the plot was thrilling!"
# Predicted Sentiment: Positive

# Negative Review Example
new_review_2 = "I was really disappointed. The story was boring and it felt way too long."
# Predicted Sentiment: Negative
```
