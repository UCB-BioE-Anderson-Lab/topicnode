import json
import os
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# Initialize client using your OPENAI_API_KEY environment variable
client = OpenAI()

def _extract_json_from_response(response_dict: dict) -> dict:
    """
    Given a full response dict from the Responses API, extract the assistant's
    JSON payload from the first output_text message. If parsing fails, return
    a dict with the raw text.
    """
    outputs = response_dict.get("output", [])
    for item in outputs:
        if item.get("type") == "message":
            for content in item.get("content", []):
                if content.get("type") == "output_text":
                    text = content.get("text", "").strip()
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        return {"raw_text": text}
    raise ValueError("No output_text content found in response")

def run_pdf_prompt(filepath: str, prompt: str, model: str = "gpt-5") -> dict:
    """
    Given a local PDF path and a prompt, upload the file and send it to the model.
    Returns the extracted JSON response as a Python dict.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File does not exist: {filepath}")

    # Create a file using the Files API
    with open(filepath, "rb") as f:
        file_obj = client.files.create(
            file=f,
            purpose="assistants"
        )

    file_id = file_obj.id

    # Send file + prompt to the Responses API
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": file_id},
                    {"type": "input_text", "text": prompt}
                ]
            }
        ]
    )

    # Convert the Response object to a plain dict and extract the model's JSON
    response_dict = response.model_dump()
    return _extract_json_from_response(response_dict)


def run_text_prompt(prompt: str, model: str = "gpt-5") -> dict:
    """
    Send a plain text prompt to the model and return the extracted JSON response as a dict.
    """
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt}
                ]
            }
        ]
    )

    response_dict = response.model_dump()
    return _extract_json_from_response(response_dict)


def main():
    # Test 1: PDF + prompt
    filepath = "/Users/jcandersonucb/Downloads/nissle_knockout/pdfs/aem.00031-25.pdf"
    prompt = 'What is the title of this paper? Respond with json in the form {"title": str}'

    print("Sending request to model (PDF prompt)...")
    response_json = run_pdf_prompt(filepath, prompt)

    print("\nModel Response JSON (PDF prompt):")
    print(json.dumps(response_json, indent=2))

    # Test 2: plain text prompt
    text_prompt = 'Who is the 8th US president? Respond with json of form {"name": str}'
    print("\nSending request to model (text prompt)...")
    text_response = run_text_prompt(text_prompt)

    print("\nModel Response JSON (text prompt):")
    print(json.dumps(text_response, indent=2))


if __name__ == "__main__":
    main()