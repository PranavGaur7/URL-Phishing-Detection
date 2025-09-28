import pandas as pd
import re
from urllib.parse import urlparse
from scipy.stats import entropy
import numpy as np

# Custom function to extract subdomain details without tldextract
def extract_domain_parts(domain):
    parts = domain.split('.')
    if len(parts) > 2:
        subdomain = '.'.join(parts[:-2])
        domain_name = parts[-2] + '.' + parts[-1]
    else:
        subdomain = ''
        domain_name = domain
    return subdomain, domain_name

# Modified extract_features function (compatible without tldextract)
def extract_features(url):
    features = {}
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url
    
    # Basic URL and Character-based Features (1-15)
    features['url_length'] = len(url)
    features['number_of_dots_in_url'] = url.count('.')
    features['having_repeated_digits_in_url'] = 1 if re.search(r'(\d)\1', url) else 0
    features['number_of_digits_in_url'] = len(re.findall(r'\d', url))
    features['number_of_special_char_in_url'] = len(re.findall(r'[^a-zA-Z0-9]', url))
    features['number_of_hyphens_in_url'] = url.count('-')
    features['number_of_underline_in_url'] = url.count('_')
    features['number_of_slash_in_url'] = url.count('/')
    features['number_of_questionmark_in_url'] = url.count('?')
    features['number_of_equal_in_url'] = url.count('=')
    features['number_of_at_in_url'] = url.count('@')
    features['number_of_dollar_in_url'] = url.count('$')
    features['number_of_exclamation_in_url'] = url.count('!')
    features['number_of_hashtag_in_url'] = url.count('#')
    features['number_of_percent_in_url'] = url.count('%')

    # Domain-based Features (16-23)
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    features['domain_length'] = len(domain)
    features['number_of_dots_in_domain'] = domain.count('.')
    features['number_of_hyphens_in_domain'] = domain.count('-')
    features['having_special_characters_in_domain'] = 1 if re.search(r'[^a-zA-Z0-9.-]', domain) else 0
    features['number_of_special_characters_in_domain'] = len(re.findall(r'[^a-zA-Z0-9.-]', domain))
    domain_digits = re.findall(r'\d', domain)
    features['having_digits_in_domain'] = 1 if len(domain_digits) > 0 else 0
    features['number_of_digits_in_domain'] = len(domain_digits)
    features['having_repeated_digits_in_domain'] = 1 if re.search(r'(\d)\1', domain) else 0

    # Subdomain-based Features (24-34)
    subdomain, main_domain = extract_domain_parts(domain)
    features['number_of_subdomains'] = len(subdomain.split('.')) if subdomain else 0
    features['having_dot_in_subdomain'] = 1 if '.' in subdomain else 0
    features['having_hyphen_in_subdomain'] = 1 if '-' in subdomain else 0
    subdomain_parts = subdomain.split('.') if subdomain else []
    features['average_subdomain_length'] = np.mean([len(p) for p in subdomain_parts]) if subdomain_parts else 0
    features['average_number_of_dots_in_subdomain'] = subdomain.count('.')
    features['average_number_of_hyphens_in_subdomain'] = subdomain.count('-')
    features['having_special_characters_in_subdomain'] = 1 if re.search(r'[^a-zA-Z0-9.-]', subdomain) else 0
    features['number_of_special_characters_in_subdomain'] = len(re.findall(r'[^a-zA-Z0-9.-]', subdomain))
    subdomain_digits = re.findall(r'\d', subdomain)
    features['having_digits_in_subdomain'] = 1 if len(subdomain_digits) > 0 else 0
    features['number_of_digits_in_subdomain'] = len(subdomain_digits)
    features['having_repeated_digits_in_subdomain'] = 1 if re.search(r'(\d)\1', subdomain) else 0

    # Path-based and Other Features (35-41)
    path = parsed_url.path
    features['having_path'] = 1 if path and path != '/' else 0
    features['path_length'] = len(path)
    features['having_query'] = 1 if parsed_url.query else 0
    features['having_fragment'] = 1 if parsed_url.fragment else 0
    features['having_anchor'] = 1 if '#' in url else 0

    # Entropy features
    url_chars = [char for char in url]
    if len(url_chars) > 0:
        _, counts = np.unique(url_chars, return_counts=True)
        features['entropy_of_url'] = entropy(counts, base=2)
    else:
        features['entropy_of_url'] = 0

    domain_chars = [char for char in domain]
    if len(domain_chars) > 0:
        _, counts = np.unique(domain_chars, return_counts=True)
        features['entropy_of_domain'] = entropy(counts, base=2)
    else:
        features['entropy_of_domain'] = 0

    return features

# Normalize type labels
def normalize_type_label(type_str):
    """Convert 'phishing'/'Phishing' to 1, 'legitimate' to 0"""
    if str(type_str).strip().lower() == 'phishing':
        return 1
    else:
        return 0

# Function to process a dataframe with url and type columns
def process_dataframe(df):
    """
    Process a DataFrame with 'url' and 'type' columns
    Returns a DataFrame with extracted features and normalized Type column
    """
    # Create a copy to avoid modifying original dataframe
    df_copy = df.copy()
    
    # Normalize type labels
    df_copy['Type'] = df_copy.iloc[:, 1].apply(normalize_type_label)  # Second column is type
    
    # Extract features for each URL
    feature_list = []
    print(f"Processing {len(df_copy)} URLs...")
    
    for i, url in enumerate(df_copy.iloc[:, 0]):  # First column is URL
        if i % 100 == 0:  # Progress indicator
            print(f"Processed {i} URLs...")
        try:
            features = extract_features(str(url))
            feature_list.append(features)
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            # Add empty features in case of error
            features = {feature: 0 for feature in ['url_length', 'number_of_dots_in_url', 'having_repeated_digits_in_url', 'number_of_digits_in_url', 'number_of_special_char_in_url', 'number_of_hyphens_in_url', 'number_of_underline_in_url', 'number_of_slash_in_url', 'number_of_questionmark_in_url', 'number_of_equal_in_url', 'number_of_at_in_url', 'number_of_dollar_in_url', 'number_of_exclamation_in_url', 'number_of_hashtag_in_url', 'number_of_percent_in_url', 'domain_length', 'number_of_dots_in_domain', 'number_of_hyphens_in_domain', 'having_special_characters_in_domain', 'number_of_special_characters_in_domain', 'having_digits_in_domain', 'number_of_digits_in_domain', 'having_repeated_digits_in_domain', 'number_of_subdomains', 'having_dot_in_subdomain', 'having_hyphen_in_subdomain', 'average_subdomain_length', 'average_number_of_dots_in_subdomain', 'average_number_of_hyphens_in_subdomain', 'having_special_characters_in_subdomain', 'number_of_special_characters_in_subdomain', 'having_digits_in_subdomain', 'number_of_digits_in_subdomain', 'having_repeated_digits_in_subdomain', 'having_path', 'path_length', 'having_query', 'having_fragment', 'having_anchor', 'entropy_of_url', 'entropy_of_domain']}
            feature_list.append(features)
    
    # Create dataframe from features
    features_df = pd.DataFrame(feature_list)
    
    # Add the Type column as first column (target variable)
    features_df['Type'] = df_copy['Type'].values
    
    # Reorder columns to match your desired format (Type first, then all features)
    cols = features_df.columns.tolist()
    cols = ['Type'] + [c for c in cols if c != 'Type']
    features_df = features_df[cols]
    
    return features_df

def main():
    print("Reading CSV files...")
    
    # Download 'file1.csv' and 'file2.csv' From Drive
    df1 = pd.read_csv('file1.csv')  
    df2 = pd.read_csv('file2.csv')  
    
    print(f"File 1 shape: {df1.shape}")
    print(f"File 2 shape: {df2.shape}")
    
    # Process both dataframes
    print("\nProcessing first file...")
    processed_df1 = process_dataframe(df1)
    
    print("\nProcessing second file...")
    processed_df2 = process_dataframe(df2)
    
    # Combine both datasets
    print("\nCombining datasets...")
    combined_df = pd.concat([processed_df1, processed_df2], ignore_index=True)
    
    print(f"\nCombined dataset shape: {combined_df.shape}")
    print(f"Features: {combined_df.shape[1] - 1}")  # -1 because Type is target, not feature
    print(f"Samples: {combined_df.shape[0]}")
    
    # Display first few rows
    print("\nFirst 5 rows of the dataset:")
    print(combined_df.head())
    
    # Display target distribution
    print(f"\nTarget distribution:")
    print(combined_df['Type'].value_counts())
    print(f"Legitimate: {(combined_df['Type'] == 0).sum()}")
    print(f"Phishing: {(combined_df['Type'] == 1).sum()}")
    
    # Save the processed dataset
    output_filename = 'phishing_dataset_features.csv'
    combined_df.to_csv(output_filename, index=False)
    print(f"\nDataset saved as '{output_filename}'")
    
    return combined_df

# Run the main function
if __name__ == "__main__":
    dataset = main()
