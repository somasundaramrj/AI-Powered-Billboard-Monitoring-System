from serpapi import GoogleSearch

search = GoogleSearch(params)
results = search.get_dict()  # Fetch results as Python dictionary

for result in results.get("organic_results", []):
    print(result["title"], result["link"])


