import requests
import json


def query_ollama(prompt, model="llama3"):
    url = "http://localhost:11434/api/generate"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,  # Set to True if you want streaming response
    }

    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        result = response.json()
        return result["response"]  # You can access result['response']
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")


# Example usage
if __name__ == "__main__":
    user_prompt = "You should answer the prompt only on the following format, dont respond further: {statement_is_true: 0}. If the statement is true swap the 0 for 1. Statement: Reinforcement learning is not identical to supervised learning."
    output = query_ollama(user_prompt)
    print(json.dumps(output, indent=2))
