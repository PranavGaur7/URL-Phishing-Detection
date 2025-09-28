import re
from urllib.parse import urlparse
import tldextract
from scipy.stats import entropy
import numpy as np

def extract_features(url):
    """
    Extracts the 41 features from a given URL to match the training dataset.
    """
    features = {}

    # Make sure the URL has a scheme, otherwise urlparse might not work correctly
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    # --- Basic URL and Character-based Features (1-15) ---
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

    # --- Domain-based Features (16-23) ---
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

    # --- Subdomain-based Features (24-34) ---
    ext = tldextract.extract(url)
    subdomain = ext.subdomain
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

    # --- Path-based and Other Features (35-41) ---
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