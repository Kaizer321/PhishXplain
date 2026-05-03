import re
import urllib.parse
import tldextract

class FeatureExtractor:
    def __init__(self, top_14_features):
        self.feature_names = top_14_features

    def extract_features(self, url: str) -> dict:
        features = {}
        parsed_url = urllib.parse.urlparse(url)
        ext = tldextract.extract(url)
        domain_full = f"{ext.domain}.{ext.suffix}"
        
        # 1. having_ip_address: -1 if IP is present (suspicious/phishing in UCI logic scaled), else 1
        ip_pattern = re.compile(r"(([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5])\.([01]?\d\d?|2[0-4]\d|25[0-5]))")
        features['having_ip_address'] = -1 if ip_pattern.search(parsed_url.netloc) else 1

        # 2. prefix_suffix: Dash in domain name -> -1
        features['prefix_suffix'] = -1 if '-' in ext.domain else 1

        # 3. having_sub_domain: dots in domain structure.
        dots = ext.subdomain.count('.') + (1 if ext.subdomain else 0)
        if dots == 0:
            features['having_sub_domain'] = 1
        elif dots == 1:
            features['having_sub_domain'] = 0
        else:
            features['having_sub_domain'] = -1
            
        # 4. sslfinal_state: proxy check via scheme
        features['sslfinal_state'] = 1 if parsed_url.scheme == 'https' else -1

        # MOCK EXTERNAL / COMPLEX FEATURES for API latency constraints
        # Real-world deployment would replace these with API calls (e.g. WHOIS, PageRank API)
        features['web_traffic'] = 1
        features['url_of_anchor'] = 1
        features['links_in_tags'] = 1
        features['links_pointing_to_page'] = 1
        features['sfh'] = 1
        features['age_of_domain'] = 1
        features['request_url'] = 1
        features['dnsrecord'] = 1
        features['google_index'] = 1
        features['page_rank'] = 1

        return features

    def get_feature_vector(self, url: str) -> list:
        features_dict = self.extract_features(url)
        # Ensure ordered correctly based on the top 14 features trained on
        return [features_dict.get(f, 0) for f in self.feature_names]
