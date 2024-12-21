import openai
import json
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise ValueError("OpenAI API key is not set. Please add it to the .env file.")

# Load agents from JSON file
with open("agents.json", "r") as f:
    agents = json.load(f)

def generate_user_prompts(num_prompts=500):
    """
    Generates user prompts dynamically based on the agents list.
    """
    system_prompt = (
        "You are an assistant tasked with creating diverse user prompts. "
        "Each prompt should ask for advice or assistance that a student might need. This will be used to match to a potential agent later for transfer learning. DO NOT STATE WHICH AGENT WILL BE USED AT IN ANY CIRCUMSTANCE"
    )
    agents_text = json.dumps(agents, indent=4)

    user_prompts = []
    for _ in range(num_prompts // 10):  # Generate prompts in batches
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Generate 10 unique prompts for these agents:\n{agents_text}"}
                ],
                temperature=0.7
            )
            batch_prompts = response["choices"][0]["message"]["content"].strip().split("\n")
            user_prompts.extend(batch_prompts)
        except Exception as e:
            print(f"Error generating prompts: {e}")
    return user_prompts[:num_prompts]

if __name__ == "__main__":
    print("Generating user prompts...")
    prompts = generate_user_prompts(num_prompts=500)

    # Save prompts to JSON
    with open("prompts.json", "w") as f:
        json.dump(prompts, f, indent=4)

    print("Saved 500 user prompts to prompts.json!")
