import openai
import ollama
import json
import pandas as pd
import inference
import re
from pydantic import BaseModel, Field
from typing import Literal, Optional

# Redirect all prints to a file
import sys
sys.stdout = open("ftoutput.txt", "w")

# API Keys
openai.api_key = "TOKEN"

# Ensure `data_prompt` is correctly formatted in `inference`
inference.data_prompt = """You are an assistant that strictly outputs JSON-formatted responses.

Agents:
{{
    "academic_agent": "Provides academic guidance, generates research insights, and assists with scholarly writing.",
    "math_agent": "Solves mathematical problems, explains concepts, and provides tutoring on complex equations.",
    "cocktail_mixlogist": "Suggests and customizes cocktail recipes based on ingredients and user preferences.",
    "cook_therapist": "Combines cooking advice with therapeutic insights to make meal prep a relaxing experience.",
    "creation_agent": "Assists with brainstorming and creating content for writing, art, or other creative projects.",
    "festival_card_designer": "Designs greeting cards and invitations for festivals and celebrations, customized to themes and styles.",
    "fitness_trainer": "Provides workout plans, fitness tips, and personalized training advice to meet health goals.",
    "logo_creator": "Generates logo designs for branding, using customizable templates and artistic styles.",
    "meme_creator": "Creates memes based on popular formats or user-submitted text for social media engagement.",
    "music_composer": "Composes original music, generates loops and samples, and customizes tunes based on mood and genre.",
    "story_teller": "Crafts stories, narratives, and interactive fiction, engaging users with creative storytelling.",
    "career_coach": "Offers career advice, job interview prep, and professional growth strategies tailored to different industries.",
    "language_tutor": "Teaches and practices foreign languages, including grammar explanations, conversation, and pronunciation help.",
    "health_nutritionist": "Provides personalized dietary recommendations, meal planning, and nutritional advice for health and wellness goals.",
    "coding_helper": "Guides through programming problems, offers coding examples, and explains programming concepts in various languages.",
    "resume_builder": "Helps create and refine resumes, offers suggestions for improvements, and optimizes formatting for job applications.",
    "financial_advisor": "Gives financial planning advice, budget creation, and insights on investments and savings strategies.",
    "travel_planner": "Helps plan vacations and trips, suggests itineraries, and recommends accommodations and activities.",
    "mental_health_companion": "Offers mental wellness advice, mindfulness practices, and coping mechanisms for stress management.",
    "sustainability_advisor": "Provides eco-friendly tips, sustainable product recommendations, and helps make greener lifestyle choices.",
    "parenting_support": "Supports parenting challenges, provides tips for child development, and offers age-appropriate activity ideas.",
    "pet_care_specialist": "Gives advice on pet care, training tips, and health recommendations for different types of pets.",
    "fashion_stylist": "Suggests outfits based on trends, occasion, and personal style, providing tips on colors and accessories.",
    "home_organizer": "Assists with home organization tips, decluttering techniques, and ideas for creating efficient storage spaces.",
    "garden_guru": "Offers gardening tips, plant care instructions, and seasonal planting advice for indoor and outdoor spaces.",
    "event_planner": "Assists in planning events, creating schedules, and managing guest lists for celebrations or professional gatherings.",
    "crypto_analyst": "Provides insights on cryptocurrency trends, trading advice, and explains blockchain concepts in simple terms.",
    "virtual_therapist": "Offers general mental health advice, listening support, and self-care practices without replacing professional therapy.",
    "diy_crafter": "Guides users through DIY projects, crafts, and home improvement ideas with step-by-step instructions.",
    "robotics_expert": "Assists with robotics projects, provides explanations on sensors, motors, and coding for robotics applications.",
    "urban_gardener": "Gives specialized tips for urban gardening, such as growing plants in small spaces and managing indoor plants."
}}

Respond with the following JSON structure:
{{
    "recommended_agent": "<agent_name>",
    "justification": "<reason for selecting the agent>"
}}

### Input:
{text}

### Response:
{answer}
"""

def recommend_agent_openai(prompt: str) -> dict:
    """Fetches a recommendation from OpenAI."""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": inference.data_prompt.format(text=prompt, answer="")}],
        max_tokens=300,
        temperature=0
    )
    return json.loads(response['choices'][0]['message']['content'])

def recommend_agent_ollama(prompt: str) -> dict:
    """Fetches a recommendation from Ollama."""
    response = ollama.chat(model="llama3.1", messages=[{'role': 'user', 'content': inference.data_prompt.format(text=prompt, answer="")}])
    return json.loads(response['message']['content'])

def recommend_agent_finetuned(prompt: str) -> dict:
    """Fetches a recommendation from the fine-tuned model."""
    formatted_prompt = inference.data_prompt.format(text=prompt, answer="")
    inputs = inference.tokenizer([formatted_prompt], return_tensors="pt").to("cuda")

    # Generate response
    outputs = inference.model.generate(**inputs, max_new_tokens=5020, use_cache=True)
    raw_answer = inference.tokenizer.batch_decode(outputs)[0]

    # Debugging: Print the raw output
    print(f"Raw Output from Fine-tuned Model:\n{raw_answer}")

    # Extract `recommended_agent` using regex
    match = re.search(r'"recommended_agent"\s*:\s*"([^"]+)"', raw_answer)
    if match:
        return {"recommended_agent": match.group(1)}
    else:
        print("Failed to extract recommended_agent from the response.")
        return {"recommended_agent": None}


def generate_user_inputs() -> list:
    """Generates a list of 50 diverse prompts for testing."""
    return [
        "I want to learn how to write a book",
        "Help me plan a vacation to Hawaii",
        "I need assistance with my resume",
        "Can you teach me Python programming?",
        "What workout plan should I follow to lose weight?",
        "Give me tips on sustainable living",
        "How can I create a logo for my brand?",
        "I need a meal plan for a high-protein diet",
        "How do I write a compelling short story?",
        "Suggest some gardening tips for a beginner",
        "Can you help me create a budget for saving money?",
        "What are the best ways to train my dog?",
        "I need advice on choosing a college major",
        "What are the steps to start a business?",
        "How can I learn public speaking?",
        "Can you suggest a skincare routine?",
        "I want to start meditating. How should I begin?",
        "Give me tips for improving my productivity",
        "How do I fix my posture while working at a desk?",
        "Help me organize my closet efficiently",
        "What are some creative gift ideas for a friend?",
        "Can you explain basic cryptocurrency trading?",
        "What are some beginner-friendly yoga poses?",
        "I need help planning a dinner party menu",
        "How do I prepare for a job interview?",
        "What are the best plants for an indoor garden?",
        "How do I train for a 5k run?",
        "Can you suggest some daily mindfulness practices?",
        "How do I learn basic car maintenance?",
        "Give me tips for creating a minimalist wardrobe",
        "Can you help me design a personal website?",
        "What are some tips for writing engaging social media posts?",
        "How do I prepare a healthy weekly meal plan?",
        "Can you suggest hobbies for stress relief?",
        "How can I improve my handwriting?",
        "What are the best practices for networking in my career?",
        "How do I stay motivated while working remotely?",
        "Give me ideas for decorating a small apartment",
        "What are some simple DIY home improvement projects?",
        "How do I create an eye-catching resume?",
        "Can you suggest books to improve my leadership skills?",
        "What are the basics of composting for beginners?",
        "Help me plan a family-friendly road trip itinerary",
        "How do I learn basic sewing skills?",
        "What are some beginner tips for digital art?",
        "How can I create an effective study schedule?",
        "Give me tips on improving my cooking skills",
        "What are some eco-friendly swaps I can make at home?",
        "How do I start learning a new language?",
        "What are the best ways to practice self-care?"
    ]

def compare_recommendations():
    """Compare recommendations across OpenAI, Ollama, and Fine-tuned model."""
    user_inputs = generate_user_inputs()
    results = []

    matches_ollama_gpt = 0
    matches_finetuned_gpt = 0

    for idx, prompt in enumerate(user_inputs, start=1):
        print(f"Processing {idx}/{len(user_inputs)}: {prompt}")

        gpt_response = recommend_agent_openai(prompt)
        ollama_response = recommend_agent_ollama(prompt)
        finetuned_response = recommend_agent_finetuned(prompt)

        # Compare only the `recommended_agent` fields
        matches_ollama_gpt += gpt_response["recommended_agent"] == ollama_response["recommended_agent"]
        matches_finetuned_gpt += gpt_response["recommended_agent"] == finetuned_response["recommended_agent"]

        results.append({
            "Prompt": prompt,
            "GPT Response": gpt_response["recommended_agent"],
            "Ollama Response": ollama_response["recommended_agent"],
            "Fine-tuned Response": finetuned_response["recommended_agent"]
        })

    gpt_ollama_match_pct = (matches_ollama_gpt / len(user_inputs)) * 100
    finetuned_gpt_match_pct = (matches_finetuned_gpt / len(user_inputs)) * 100

    print(f"\nGPT-4 to Ollama Match Percentage: {gpt_ollama_match_pct:.2f}%")
    print(f"Fine-tuned to GPT-4 Match Percentage: {finetuned_gpt_match_pct:.2f}%")

    df = pd.DataFrame(results)
    df.to_csv("results.csv", index=False)
    print("Results saved to results.csv")

if __name__ == "__main__":
    compare_recommendations()
sys.stdout.close()
