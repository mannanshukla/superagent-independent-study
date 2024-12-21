import json
import re

# Load agents and prompts
with open("agents.json", "r") as f:
    agents = json.load(f)

with open("prompts.json", "r") as f:
    raw_prompts = json.load(f)

def clean_data(raw_prompts, agents):
    """
    Cleans and structures data into Alpaca format.
    Handles cases with index prefixes, missing agents, and malformed prompts gracefully.
    """
    agents_keys = set(agents.keys())  # For quick lookup
    alpaca_data = []

    for raw_prompt in raw_prompts:
        try:
            # Remove leading index (e.g., "1. ") if present
            raw_prompt = re.sub(r"^\d+\.\s*", "", raw_prompt.strip())
            
            # Ensure the prompt is valid and extract agent and input
            match = re.match(r'^([\w\s]+):\s*"(.*?)"$', raw_prompt)
            if not match:
                print(f"Skipping malformed prompt: {raw_prompt}")
                continue

            # Extract agent and input
            agent_name = match.group(1).strip().lower().replace(" ", "_")
            user_input = match.group(2).strip()

            if agent_name in agents_keys:
                # Format in Alpaca style
                alpaca_data.append({
                    "instruction": f"You are an assistant that strictly outputs JSON-formatted responses.\n\nAgents:\n{json.dumps(agents, indent=4)}\n\nRespond with the following JSON structure:\n{{\n    \"recommended_agent\": \"<agent_name>\",\n    \"justification\": \"<reason for selecting the agent>\"\n}}",
                    "input": f"User Input: '{user_input}'",
                    "output": json.dumps({
                        "recommended_agent": agent_name,
                        "justification": f"The user query matches the functionality of the {agent_name.replace('_', ' ')}."
                    }, indent=4)
                })
            else:
                print(f"Skipping unknown agent: {agent_name}")
        except Exception as e:
            print(f"Error processing prompt '{raw_prompt}': {e}")

    return alpaca_data

if __name__ == "__main__":
    print("Cleaning data...")
    cleaned_data = clean_data(raw_prompts, agents)

    # Save cleaned data to JSON
    with open("alpaca_cleaned.json", "w") as f:
        json.dump(cleaned_data, f, indent=4)

    print("Cleaned Alpaca-style dataset saved to alpaca_cleaned.json!")
