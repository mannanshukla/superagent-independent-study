import ollama
import json
from pydantic import BaseModel, ValidationError, Field
from typing import Literal, Optional

# List of all available agents with descriptions
AGENTS = {
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
    "urban_gardener": "Gives specialized tips for urban gardening, such as growing plants in small spaces and managing indoor plants.",
    "academic_advisor": "Helps students with course selection, academic planning, and strategies for managing workload effectively.",
    "study_buddy": "Supports students with study plans, time management techniques, and motivational strategies for exam preparation.",
    "mental_health_mentor": "Provides stress management tips, mindfulness exercises, and coping strategies for common university challenges.",
    "budget_planner": "Assists with managing student finances, creating budgets, and tips for saving money on a student budget.",
    "internship_coach": "Guides students in finding internships, preparing application materials, and advice on networking in their field.",
    "exam_preparation_guide": "Offers study schedules, exam techniques, and resources to effectively prepare for upcoming exams.",
    "note_taking_assistant": "Provides note-taking strategies, tools, and tips to help students retain information more efficiently.",
    "essay_helper": "Supports with structuring essays, thesis statements, citation guidelines, and tips for clear academic writing.",
    "public_speaking_coach": "Helps students build public speaking skills, offering tips on confidence, clarity, and engaging presentations.",
    "time_management_coach": "Offers personalized time management techniques, prioritization tips, and ways to avoid procrastination.",
    "scholarship_finder": "Assists in searching for scholarships, understanding application requirements, and crafting strong submissions.",
    "lab_report_assistant": "Guides students through writing lab reports, structuring experiments, and properly formatting scientific data.",
    "part_time_job_advisor": "Provides advice on finding part-time jobs, balancing work with studies, and resume-building tips for student jobs.",
    "language_exchange_partner": "Supports language learning through conversation practice, vocabulary building, and language immersion tips.",
    "research_assistant": "Helps with organizing research projects, finding credible sources, and tips on summarizing academic literature.",
    "student_wellness_advisor": "Promotes student well-being with advice on sleep, nutrition, physical activity, and work-life balance.",
    "campus_event_suggester": "Recommends on-campus events, student organizations, and resources to enhance the university experience.",
    "peer_tutor": "Provides help with specific subjects, explaining difficult concepts, and reviewing class material for better understanding.",
    "housing_advisor": "Offers advice on finding student housing, understanding lease agreements, and tips for roommate arrangements.",
    "career_planner": "Helps with career exploration, setting goals, and identifying resources to develop professional skills.",
    "lab_tutorials": "Provides step-by-step guidance on laboratory techniques, lab safety, and understanding experimental procedures.",
    "group_project_helper": "Offers advice on managing group projects, dividing tasks, and handling common group challenges.",
    "textbook_finder": "Suggests where to find affordable textbooks, rent options, and strategies to access free study resources.",
    "graduate_school_mentor": "Guides students through the grad school application process, program selection, and preparing necessary materials.",
    "networking_coach": "Offers tips for professional networking, LinkedIn profile optimization, and making industry connections.",
    "study_abroad_advisor": "Supports students interested in studying abroad with information on programs, applications, and travel prep.",
    "campus_resource_guide": "Provides information on campus resources like libraries, tutoring centers, and academic support services.",
    "online_learning_assistant": "Helps students navigate online courses, offering tips for staying focused and engaging in virtual classes.",
    "stress_relief_coach": "Provides stress-relief techniques such as breathing exercises, relaxation methods, and mindfulness practices."
}

# Define the AgentResponse model
class AgentResponse(BaseModel):
    recommended_agent: Optional[Literal[tuple(AGENTS.keys())]] = Field(None, description="Recommended agent, or null if none selected.")
    justification: str = Field(..., description="Reason for selecting the agent, or the reason for refusal.")

def generate_prompt(prompt: str) -> str:
    """Generates a formatted prompt for the AI model."""
    agents_json = json.dumps(AGENTS, indent=2)
    return f"""You are an assistant that strictly outputs JSON-formatted responses. 
Based on the following user input, select the best agent from the list and provide a justification in JSON format:
    
User Input: '{prompt}'

Agents:
{agents_json}

Respond with only the following JSON structure:
{{
    "recommended_agent": "<agent_name>",
    "justification": "<reason for selecting the agent>"
}}

Strictly follow this response format.
"""

def recommend_agent(prompt: str) -> None:
    """Processes the prompt and fetches a recommendation from Ollama."""
    ollama_prompt = generate_prompt(prompt)

    # Call Ollama and get the response
    response = ollama.chat(model="llama3.2", messages=[{'role': 'user', 'content': ollama_prompt}])

    # Attempt to parse and validate the response
    try:
        recommendation = json.loads(response['message']['content'])
        valid_recommendation = AgentResponse(**recommendation)
        print(json.dumps(valid_recommendation.dict(), indent=2))
    except json.JSONDecodeError:
        # Handle non-JSON response (likely a refusal)
        refusal_response = AgentResponse(
            recommended_agent=None,
            justification="Ollama refused to provide an answer to the prompt."
        )
        print(json.dumps(refusal_response.dict(), indent=2))
    except ValidationError as e:
        print("Error: Response did not match the expected schema.")
        print(e.json())

# Main program loop for user input
if __name__ == "__main__":
    while True:
        user_input = input("Enter your prompt (or type 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        recommend_agent(user_input)
