import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the saved model and preprocessor
try:
    model = joblib.load('model.pkl')
    preprocessor = joblib.load('preprocessor.pkl')
    # Assuming label_encoder was also saved, load it here
    # If not saved, we need to recreate the mapping from encoded to original labels
    # For now, let's assume we know the mapping or can recreate it.
    # Based on previous output:
    # 0: Banh Mi, 1: Coffee, 2: Espresso Based Coffee, 3: Food, 4: Freeze, 5: Other, 6: Tea
    product_type_mapping = {
        0: 'Banh Mi',
        1: 'Coffee',
        2: 'Espresso Based Coffee',
        3: 'Food',
        4: 'Freeze',
        5: 'Other',
        6: 'Tea'
    }

except FileNotFoundError:
    st.error("Model or preprocessor not found. Please ensure 'model.pkl' and 'preprocessor.pkl' are in the same directory.")
    st.stop()


st.title('Product Type Prediction App')

st.write("""
Enter the features below to predict the product type.
""")

# Define input fields based on the features used for training (excluding target and IDs)
# Referencing the categorical and numerical features used in the preprocessor:
# categorical_features = ['store_location', 'product_category', 'product_detail', 'Size', 'Gender', 'Occupation', 'Season', 'product_name']
# numerical_features = ['transaction_qty', 'unit_price', 'Total_Bill', 'Age', 'Income']

# Create input widgets
store_location = st.selectbox('Store Location', ['Nha Trang - City Center', 'Da Nang - Thanh Khe', 'Hanoi - Hoan Kiem', 'Hanoi - Ba Dinh', 'Hue - City Center', 'Ho Chi Minh City - District 7', 'Ho Chi Minh City - Tan Binh', 'Da Nang - Hai Chau', 'Can Tho - Ninh Kieu', 'Hanoi - Cau Giay', 'Hanoi - Dong Da', 'Hanoi - Tay Ho', 'Ho Chi Minh City - District 1', 'Ho Chi Minh City - District 3', 'Ho Chi Minh City - District 5']) # Add all unique store locations from your data
product_category = st.selectbox('Product Category', ['Tea', 'Traditional Coffee', 'Food', 'Freeze', 'Espresso Based Coffee', 'Other']) # Add all unique product categories
product_detail = st.text_input('Product Detail', 'Unknown') # Allow text input, fillna('Unknown') was used
Size = st.selectbox('Size', ['M', 'Regular', 'L', 'Small']) # Add all unique sizes
Gender = st.selectbox('Gender', ['Male', 'Female', 'Other']) # Add all unique genders
Occupation = st.selectbox('Occupation', ['Nghề tự do', 'Sinh viên, học sinh', 'Kinh doanh', 'Trưởng phòng', 'Giám đốc', 'Nhà đầu tư', 'Nhân viên văn phòng', 'Other']) # Add all unique occupations
Season = st.selectbox('Season', ['Xuân', 'Hè', 'Thu', 'Đông']) # Add all unique seasons
product_name = st.text_input('Product Name', 'Unknown') # Allow text input, fillna('Unknown') was used

transaction_qty = st.number_input('Transaction Quantity', min_value=1, value=1)
unit_price = st.number_input('Unit Price', min_value=0, value=30000)
Total_Bill = st.number_input('Total Bill', min_value=0, value=30000)
Age = st.number_input('Age', min_value=0, value=25)
Income = st.number_input('Income', min_value=0.0, value=7000000.0)


if st.button('Predict Product Type'):
    # Create a DataFrame with the user's input
    input_data = pd.DataFrame([[
        store_location, product_category, product_detail, Size, Gender,
        Occupation, Season, product_name, transaction_qty, unit_price,
        Total_Bill, Age, Income
    ]], columns=[
        'store_location', 'product_category', 'product_detail', 'Size', 'Gender',
        'Occupation', 'Season', 'product_name', 'transaction_qty', 'unit_price',
        'Total_Bill', 'Age', 'Income'
    ])

    # Ensure the order of columns matches the training data (this might need adjustment
    # if the original training dataframe had other columns not used in preprocessing.
    # Based on the preprocessor definition, these are the columns it expects in order
    # for the numerical and then categorical features).
    # Let's re-create a dummy dataframe with all columns to ensure correct order
    # before preprocessing.
    dummy_data = pd.DataFrame(columns=df_combined.drop(columns=['transaction_id', 'transaction_date', 'transaction_time', 'product_id', 'customer_id', 'customer_name', 'product_type', 'product_type_encoded']).columns)
    input_data = pd.concat([dummy_data, input_data], ignore_index=True)


    # Preprocess the input data
    input_data_processed = preprocessor.transform(input_data)

    # Make prediction
    prediction_encoded = model.predict(input_data_processed)[0]

    # Convert encoded prediction back to original label
    predicted_product_type = product_type_mapping.get(prediction_encoded, 'Unknown')


    st.success(f'Predicted Product Type: {predicted_product_type}')
