from serpapi import GoogleSearch

params = {
    "engine": "google",
    "q": "Drinking 8 glasses of water daily improves focus and energy",
    "api_key": "417454efa8608e4a2917c1f2c0e54a4de083ef2a4f650d18892a604c45a40162"  # Replace with your real SerpApi key
}

search = GoogleSearch(params)
results = search.get_dict()  # Fetch results as Python dictionary

for result in results.get("organic_results", []):
    print(result["title"], result["link"])


