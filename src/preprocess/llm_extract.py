import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_metadata(text: str, prompt: str) -> dict:
    """
    Extract metadata from scientific text using GPT-4o-mini.

    Args:
        text (str): The parsed scientific text (from PDF).
        prompt (str): The full prompt template with placeholder <<<TEXT GOES HERE>>>.

    Returns:
        dict: JSON response parsed into a Python dictionary.
    """
    final_prompt = prompt.replace("<TEXT>", text)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": final_prompt}
        ],
        temperature=0
    )

    output = response.choices[0].message.content.strip()

    try:
        return json.loads(output)
    except json.JSONDecodeError:
        raise ValueError(f"Model output was not valid JSON:\n{output}")