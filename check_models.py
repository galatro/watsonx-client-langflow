import requests
from datetime import datetime

# List of base URLs to compare
base_urls = [
    "https://us-south.ml.cloud.ibm.com",
    "https://eu-de.ml.cloud.ibm.com",
    "https://jp-tok.ml.cloud.ibm.com",
    "https://au-syd.ml.cloud.ibm.com"
]

endpoint = "/ml/v1/foundation_model_specs"
params = {
    "version": "2024-09-16",
    "filters": "function_embedding,!lifecycle_withdrawn" #"function_text_chat,!lifecycle_withdrawn"
}

# Get today's date
today = datetime.today().strftime("%Y-%m-%d")

results = {}
model_sets = {}

# Function to check if a model is deprecated or withdrawn
def is_deprecated_or_withdrawn(lifecycle):
    for entry in lifecycle:
        if entry["id"] in {"deprecated", "withdrawn"} and entry["start_date"] <= today:
            return True
    return False

# Fetch models from each region
for base_url in base_urls:
    url = f"{base_url}{endpoint}"
    print(f"Fetching models from {url} with params {params}")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Store response data
        results[base_url] = data

        # Extract and filter model IDs based on lifecycle status
        model_sets[base_url] = {
            model["model_id"]
            for model in data.get("resources", [])
            if not is_deprecated_or_withdrawn(model.get("lifecycle", []))
        }

    except requests.exceptions.RequestException as e:
        print(f"Error fetching from {base_url}: {e}")
        results[base_url] = None
        model_sets[base_url] = set()

# Reference model set (US-South)
us_south_models = model_sets["https://us-south.ml.cloud.ibm.com"]

# Compare models across regions
for base_url, models in model_sets.items():
    if base_url == "https://us-south.ml.cloud.ibm.com":
        continue  # Skip comparison for US-South itself

    missing_models = us_south_models - models
    if missing_models:
        print(f"\nModels missing in {base_url} compared to US-South (excluding deprecated/withdrawn models):")
        for model in missing_models:
            print(f" - {model}")
    else:
        print(f"\n{base_url} has all active models present in US-South.")

# Reference model set (US-South)
us_south_models = model_sets["https://us-south.ml.cloud.ibm.com"]

# Compare models in other regions and find those not in US-South
for base_url, models in model_sets.items():
    if base_url == "https://us-south.ml.cloud.ibm.com":
        continue  # Skip comparison for US-South itself

    # Find models in the current region that are NOT in US-South
    unique_models = models - us_south_models
    
    if unique_models:
        print(f"\nModels in {base_url} but not in US-South (excluding deprecated/withdrawn models):")
        for model in unique_models:
            print(f" - {model}")
    else:
        print(f"\nNo unique models in {base_url} compared to US-South.")

# Find models present in all regions
common_models = set.intersection(*model_sets.values())
if common_models:
    print("\nModels present in all regions (excluding deprecated/withdrawn models):")
    for model in common_models:
        print(f" - {model}")
else:
    print("\nNo models are present in all regions.")

print()