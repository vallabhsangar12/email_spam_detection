# email_spam_detection
Email Spam detection project using Naive Bayes Algorithm
1.EMAIL SPAM DETECTION

![Screenshot 1](images/ss1.png)  
![Screenshot 2](images/ss2.png)

This is a simple spam detection machine learning model using the Multinomial Naive Bayes algorithm. It classifies SMS messages as Spam or Not Spam. Built as part of the Oasis Infobyte Internship Program.

## Technologies Used
- Python
- scikit-learn
- pandas
- CountVectorizer
- VS Code
- Streamlit - UI

## Concept

This project focuses on detecting **spam emails or SMS messages** using **Machine Learning**. The key idea is to train a model that can differentiate between *spam* and *not spam* (ham) messages based on the text content.

The steps involved in the project are:

1. **Data Loading**: The dataset (`spam.csv`) contains labeled SMS messages as 'spam' or 'ham'.
2. **Text Preprocessing**: Cleaning and vectorizing text using `CountVectorizer` to convert it into a format the machine can understand.
3. **Train-Test Split**: The data is split into training and testing sets to evaluate performance.
4. **Model Training**: A **Multinomial Naive Bayes** classifier is trained on the training data.
5. **Prediction**: The model predicts whether new/unseen messages are spam or not.
6. **Evaluation**: We evaluate the model using metrics like accuracy, precision, recall, and confusion matrix.

This is an example of a **Supervised Learning** classification task, where:
- **Input (X)**: SMS message text
- **Output (Y)**: Spam or Not Spam label

The `Naive Bayes` algorithm is particularly well-suited for text classification tasks because of its simplicity, speed, and surprisingly good performance on spam filtering problems.

---

**Why Naive Bayes?**  
- It works well with **word frequency data**
- It’s **fast and efficient**
- It handles **large text data** with ease

  
## How to Run
1. Clone this repo
2. Install required libraries: scikit-learn, pandas, streamlit
3. Run the script

## Deployment using Streamlit

This project includes a **Streamlit interface** for deploying the spam detection model as a simple and interactive web app.

### What is Streamlit?

[Streamlit](https://streamlit.io) is an open-source Python library that lets you create **beautiful web apps** for data science and machine learning projects — with **just a few lines of Python code**.

It’s perfect for deploying ML models without needing complex front-end development.



### How to Run the App Locally

1. Make sure you have Python installed.
2. Install Streamlit (if not already installed): pip install streamlit
3. In the terminal, navigate to your project folder and run: streamlit run spam.py
4. This will open the app in your browser at: http://localhost:8501

## Sample Output (without streamlit)
Accuracy : 0.98
['Spam']

