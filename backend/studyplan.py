import os
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

def get_study_plan(topics, deadline):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Please set the GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    prompt = (
        f"Create a detailed study plan for the following topics/concepts: {topics}. "
        f"The deadline to complete the study is {deadline}. "
        "Break down the plan into daily or weekly tasks, and include tips for effective learning. Donot add any extra content other than the study plan or study materials"
    )
    response = model.generate_content(prompt)
    return response.text

if __name__ == "__main__":
    topics = input("Enter topics or concepts to study (comma-separated): ")
    deadline = input("Enter your deadline (e.g., 2025-07-31): ")
    plan = get_study_plan(topics, deadline)
    print("\nGenerated Study Plan:\n")
    print(plan)