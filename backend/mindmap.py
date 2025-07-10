import google.generativeai as genai
import graphviz
import os
from dotenv import load_dotenv
load_dotenv()

# Add Graphviz to PATH environment variable
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# Set your Gemini API key from environment variable

#os.getenv('GEMINI_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")

def get_dot_from_gemini(prompt):
    dot_format = '''
Output DOT format example:
digraph Mindmap {
    node [style=filled, fillcolor=lightblue, fontcolor=black, shape=ellipse];
    "Main Topic" [fillcolor=gold];
    "Main Topic" -> "Branch 1" [color=red];
    "Main Topic" -> "Branch 2" [color=green];
    "Branch 1" -> "Subtopic 1" [color=red];
    "Branch 1" -> "Subtopic 2" [color=red];
}
'''
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content(
        f"""You are an assistant that generates mindmaps in Graphviz DOT format. 
Respond ONLY with valid DOT code, no explanation or formatting. 
Use a radial layout (main topic in the center, branches radiating out). 
Use different fill colors for each main branch and sub-branch, and color the edges to match. 
Use this format: {dot_format} 
Generate a detailed mindmap for: {prompt}. 
Keep node names short."""
    )
    content = response.text.strip()
    # Remove Markdown code block if present
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.strip().startswith("dot"):
            content = content.strip()[3:]
        content = content.rsplit("```", 1)[0]
    return content.strip()

def render_dot(dot_code, output_file="mindmap"):
    dot = graphviz.Source(dot_code)
    dot.format = "png"
    dot.render(filename=output_file, cleanup=True)
    output_file += ".png"
    return output_file

def display_image(image_path):
    from PIL import Image
    img = Image.open(image_path)
    img.show()

# Main program
if __name__ == "__main__":
    # List available models for debugging
    print("Available Gemini models:")
    for m in genai.list_models():
        print(f"- {m.name} (methods: {m.supported_generation_methods})")
    # Read prompt from input.txt
    with open("input.txt", "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    dot_code = get_dot_from_gemini(prompt)
    print("DOT code from Gemini:\n", dot_code)
    image_path = render_dot(dot_code)
    display_image(image_path)
