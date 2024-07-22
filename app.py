import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

st.write("# Credit Card Fraud Detection")

st.sidebar.header('Input Credit Card Details')

# Upload CSV file
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=['csv'])

# Example CSV file download
example_csv = pd.DataFrame({
    'V1': [0], 'V2': [0], 'V3': [0], 'V4': [0], 'V5': [0], 'V6': [0], 'V7': [0], 'V8': [0], 'V9': [0], 'V10': [0],
    'V11': [0], 'V12': [0], 'V13': [0], 'V14': [0], 'V15': [0], 'V16': [0], 'V17': [0], 'V18': [0], 'V19': [0], 
    'V20': [0], 'V21': [0], 'V22': [0], 'V23': [0], 'V24': [0], 'V25': [0], 'V26': [0], 'V27': [0], 'V28': [0], 
    'Amount': [0]
})
example_csv_data = example_csv.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    label="Download Example CSV",
    data=example_csv_data,
    file_name='example_input.csv',
    mime='text/csv',
)

# Function to create slider for 'Amount' only
def user_input():
    amount = st.sidebar.number_input('Amount')
    return pd.DataFrame({'Amount': [amount]})

# Load the pre-trained model
try:
    load_clf = joblib.load('SavedModel/credit_fraud_model')
except Exception as e:
    st.write("Error loading the model:", e)
    load_clf = None

# Main app logic
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.subheader('Uploaded CSV File')
    st.write(input_df)
    
    # Filter dataframe based on Amount
    min_amount = st.sidebar.slider('Minimum Amount', float(input_df['Amount'].min()), float(input_df['Amount'].max()), float(input_df['Amount'].min()))
    max_amount = st.sidebar.slider('Maximum Amount', float(input_df['Amount'].min()), float(input_df['Amount'].max()), float(input_df['Amount'].max()))

    filtered_df = input_df[(input_df['Amount'] >= min_amount) & (input_df['Amount'] <= max_amount)]
    st.subheader('Filtered Data')
    st.write(filtered_df)

    if st.sidebar.button("Predict") and load_clf:
        try:
            # Data Preprocessing
            # Remove rows with NaN or infinity
            filtered_df = filtered_df.replace([np.inf, -np.inf], np.nan).dropna()

            # Ensure all values are finite
            if filtered_df.empty:
                st.write("No valid data available for prediction.")
            else:
                # Make predictions
                predictions = load_clf.predict(filtered_df)
                prediction_probabilities = load_clf.predict_proba(filtered_df)

                # Display detailed results
                st.subheader('Detailed Results')
                results_df = filtered_df.copy()
                results_df['Prediction'] = predictions
                results_df['Prediction Probability'] = prediction_probabilities[:, 1]  # Probability of being fraudulent
                results_df['Is_Fraudulent'] = results_df['Prediction'].apply(lambda x: 'Fraudulent' if x == 1 else 'Normal')
                st.write(results_df)

                # Visualize prediction results
                prediction_counts = results_df['Prediction'].value_counts()
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie(prediction_counts, labels=prediction_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette('viridis'))
                ax.set_title('Predicted Transactions')
                st.pyplot(fig)

                # Display predictions
                st.subheader('Prediction Summary')
                st.write(f'Normal Transactions: {np.sum(predictions == 0)}')
                st.write(f'Fraudulent Transactions: {np.sum(predictions == 1)}')

                # Highlight fraudulent transactions in the CSV
                def highlight_fraudulent(row):
                    return ['background-color: yellow' if val == 'Fraudulent' else '' for val in row]

                styled_df = results_df.style.apply(highlight_fraudulent, subset=['Is_Fraudulent'])

                # Convert filtered DataFrame to CSV
                csv = results_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download Detailed Results as CSV",
                    data=csv,
                    file_name='detailed_results.csv',
                    mime='text/csv',
                )

        except Exception as e:
            st.write("Error making predictions:", e)

else:
    st.write('Awaiting CSV file to be uploaded.')
    # Manually create an input dataframe with only 'Amount'
    filtered_df = user_input()

# Additional instructions and notes
st.sidebar.subheader('Instructions')
st.sidebar.markdown("""
1. Upload your CSV(it should have 29 features) file using the sidebar.
2. Adjust the sliders to filter transactions based on amount.
3. Click on the **Predict** button to see the fraud prediction results.
4. Download the detailed results using the provided buttons.
""")

st.sidebar.subheader('Notes')
st.sidebar.markdown("""
- This app demonstrates how to use a pre-trained RandomForestClassifier for fraud detection.
- Visualizations and data summaries help understand the dataset and prediction outcomes.
- Adjust filters and explore results interactively.
""")

# Display any errors related to model loading
if load_clf is None:
    st.sidebar.error("Model not loaded, predictions cannot be made.")
