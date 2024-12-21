import unsloth

# Load model and tokenizer
model, tokenizer = unsloth.FastLanguageModel.from_pretrained(
    model_name="superagent/1B_finetuned_llama3.2",
    max_seq_length=5020,
    dtype=None,
    load_in_4bit=True
)

# Uncomment this line if necessary; otherwise, comment it out
model = unsloth.FastLanguageModel.for_inference(model)

# Define user input
text = "i wanna learn how to write a book"

# Corrected `data_prompt` with escaped curly braces
data_prompt = """You are an assistant that strictly outputs JSON-formatted responses.

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

# Ensure placeholders are passed correctly to `format()`
inputs = tokenizer(
    [
        data_prompt.format(
            text=text,  # User input
            answer="",  # Placeholder for the expected answer
        )
    ],
    return_tensors="pt"
).to("cuda")

# Generate response
outputs = model.generate(**inputs, max_new_tokens=5020, use_cache=True)

# Decode and process the response
answer = tokenizer.batch_decode(outputs)

# Extract the response after "### Response:"
answer = answer[0].split("### Response:")[-1].strip()

print("Answer of the question is:", answer)
