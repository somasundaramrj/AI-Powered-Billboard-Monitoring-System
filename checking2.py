import requests

API_KEY = ""  # Replace with your real Gemini API key
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

headers = {
    "Content-Type": "application/json",
    "X-goog-api-key": API_KEY
}

# Improved prompt for clear, accurate yes/no answer with explanation
payload = {
    "contents": [
        {
            "parts": [
                {
                    "text": (
                        "Please answer with 'Yes' or 'No', followed by a clear and accurate explanation, "
                        "evaluating the following statement for truth and persuasiveness: \n\n"
                        "Join over 50% of IPL fans who passionately follow RCB’s journey — experience the thrill, "
                        "the ups and downs, and cheer for your favorite team. Don’t miss out on exclusive RCB merchandise and live match updates!"
                    )
                }
            ]
        }
    ]
}

def call_gemini_api():
    response = requests.post(API_URL, json=payload, headers=headers)
    if response.status_code == 200:
        data = response.json()
        answer_text = data['candidates'][0]['content']['parts'][0]['text']


        print("Gemini API response:\n", answer_text)
    else:
        print(f"Request failed with status {response.status_code}: {response.text}")

if __name__ == "__main__":
    call_gemini_api()
