import streamlit as st
import pickle
import pandas as pd
from feature_extraction import extract_features # Make sure feature_extraction.py is in the same folder

# --- Load the saved model and scaler ---
try:
    # Load the best-performing model
    with open('best_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    
    # Load the scaler
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Model or scaler not found. Please run the training notebook first to generate 'best_model.pkl' and 'scaler.pkl'.")
    st.stop()


# --- Streamlit App Interface ---
st.title("URL Phishing Detector 🎣")
st.write(
    "Enter a URL to check if it's a phishing site or a legitimate one. "
    "This app uses a machine learning model to predict the result."
)

url_input = st.text_input("Enter the URL to analyze:", "https://www.google.com")

if st.button("Check URL"):
    if url_input:
        # 1. Extract features from the user's URL
        features_dict = extract_features(url_input)
        
        # 2. Convert the dictionary of features into a pandas DataFrame
        # The model expects a 2D array, so we create a DataFrame with a single row
        features_df = pd.DataFrame([features_dict])
        
        # 3. Reorder columns to match the training data order (important!)
        # Create a list of expected feature names in the correct order
        feature_order = ['url_length', 'number_of_dots_in_url', 'having_repeated_digits_in_url',
                         'number_of_digits_in_url', 'number_of_special_char_in_url', 'number_of_hyphens_in_url',
                         'number_of_underline_in_url', 'number_of_slash_in_url', 'number_of_questionmark_in_url',
                         'number_of_equal_in_url', 'number_of_at_in_url', 'number_of_dollar_in_url',
                         'number_of_exclamation_in_url', 'number_of_hashtag_in_url', 'number_of_percent_in_url',
                         'domain_length', 'number_of_dots_in_domain', 'number_of_hyphens_in_domain',
                         'having_special_characters_in_domain', 'number_of_special_characters_in_domain',
                         'having_digits_in_domain', 'number_of_digits_in_domain',
                         'having_repeated_digits_in_domain', 'number_of_subdomains', 'having_dot_in_subdomain',
                         'having_hyphen_in_subdomain', 'average_subdomain_length',
                         'average_number_of_dots_in_subdomain', 'average_number_of_hyphens_in_subdomain',
                         'having_special_characters_in_subdomain', 'number_of_special_characters_in_subdomain',
                         'having_digits_in_subdomain', 'number_of_digits_in_subdomain',
                         'having_repeated_digits_in_subdomain', 'having_path', 'path_length', 'having_query',
                         'having_fragment', 'having_anchor', 'entropy_of_url', 'entropy_of_domain']
        features_df = features_df[feature_order]

        # 4. Scale the features using the loaded scaler
        scaled_features = scaler.transform(features_df)
        
        # 5. Make a prediction using the loaded model
        prediction = model.predict(scaled_features)
        prediction_proba = model.predict_proba(scaled_features)

        # Display the result
        st.write("---")
        if prediction[0] == 1:
            st.error("🚨 This URL is likely a **Phishing Site**!")
            st.write(f"**Confidence:** {prediction_proba[0][1]*100:.2f}%")
        else:
            st.success("✅ This URL seems to be **Legitimate**.")
            st.write(f"**Confidence:** {prediction_proba[0][0]*100:.2f}%")
    else:
        st.warning("Please enter a URL to analyze.")