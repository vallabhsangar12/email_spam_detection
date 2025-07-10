import pandas as pd  
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st  

# Load data
data = pd.read_csv('D:\Vallabh Sangar\OIB-SIP-main\emailspam\dataset-mail\spam.csv', encoding='latin-1')

# OPTIONAL: Print original shape
print("Original Shape:", data.shape)
print(data.head())

# Drop unnecessary unnamed columns 
data = data[['v1', 'v2']]

# Clean the data
data.drop_duplicates(inplace=True)
data.dropna(inplace=True)
print("\nCleaned Shape:", data.shape)


# Rename and relabel
data['v1'] = data['v1'].replace(['ham', 'spam'], ['Not Spam', 'Spam'])
data.rename(columns={'v1': 'Category', 'v2': 'Message'}, inplace=True)

# Check class distribution
print("\nLabel Distribution:\n", data['Category'].value_counts())

# Prepare data
mess = data['Message']
cat = data['Category']

# Train-test split
mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2, random_state=0, stratify=cat)

# Vectorize text
cv = CountVectorizer(stop_words='english')
features_train = cv.fit_transform(mess_train)
features_test = cv.transform(mess_test)

# Train model
model = MultinomialNB()
model.fit(features_train, cat_train)

# Evaluate
print("Accuracy:", model.score(features_test, cat_test))

# Predict
def predict(message):
  message = cv.transform([message]).toarray()
  result = model.predict(message)
  return result

st.header("Email Spam Detector")

# Test
output = predict("WINNER!! This is the secret code to unlock the money: C3421.")
print(output)

#building application with streamlit

input_message=st.text_input("Enter a message")

if st.button('Go'):
    output=predict(input_message)
    st.text(output)
else:
    pass
    





