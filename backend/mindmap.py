import google.generativeai as genai
import json
import networkx as nx
import matplotlib.pyplot as plt
import os

# Add Graphviz to PATH environment variable
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# Set your Gemini API key from environment variable
gemini_api_key = "AIzaSyBYUOjfiCXkdTulBdM30lVBbDwd4DIppCo"
#os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")
genai.configure(api_key=gemini_api_key)

# Function to get a mindmap structure from Gemini
def get_mindmap(prompt):
    json_format = '''\nOutput JSON format example:\n{\n  "name": "Root Topic",\n  "children": [\n    {\n      "name": "Subtopic 1",\n      "children": [\n        {"name": "Sub-subtopic 1"},\n        {"name": "Sub-subtopic 2"}\n      ]\n    },\n    {\n      "name": "Subtopic 2"}\n  ]\n}\n'''
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content(
       f"You are an assistant that generates structured mindmaps. Respond ONLY with valid JSON, no explanation or formatting. Use this format: {json_format} Generate a detailed mindmap for: {prompt}"
    )
    content = response.text.strip()
    
    print("Gemini raw response:\n", content)  # Debug: print the raw output

    # Remove Markdown code block if present
    if content.startswith("```"):
        content = content.split("```")[1]  # Get the part after the first ```
        # Remove possible 'json' after ```
        if content.strip().startswith("json"):
            content = content.strip()[4:]
        # Remove trailing ```
        content = content.rsplit("```", 1)[0]
    content = content.strip()

    try:
        mindmap = json.loads(content)
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from Gemini response.")
        raise
    return mindmap

# Recursive function to add nodes and edges to a graph
def add_nodes_edges(graph, parent, children):
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict) and "name" in child:
                graph.add_edge(parent, child["name"])
                # Recursively add subchildren if present
                if "children" in child:
                    add_nodes_edges(graph, child["name"], child["children"])
    # If children is not a list, do nothing (leaf node)

# Generate mindmap graph
def generate_graph(mindmap):
    G = nx.DiGraph()
    root = mindmap.get("name", "Root")
    G.add_node(root)
    add_nodes_edges(G, root, mindmap.get("children", []))
    return G

# Visualize the mindmap
def visualize_mindmap(graph, prompt):
    plt.figure(figsize=(16, 10))
    if len(graph.nodes) <= 1 or len(graph.edges) == 0:
        nx.draw(graph, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight='bold')
    else:
        try:
            from networkx.drawing.nx_agraph import graphviz_layout  # pygraphviz backend
            pos = graphviz_layout(graph, prog="dot")
            nx.draw(graph, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight='bold', arrows=True)
        except Exception as e:
            print("Falling back to shell_layout due to error:", e)
            try:
                shells = [[list(graph.nodes)[0]]]
                children = list(graph.successors(list(graph.nodes)[0]))
                if children:
                    shells.append(children)
                pos = nx.shell_layout(graph, shells)
                nx.draw(graph, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight='bold', arrows=True)
            except Exception as e2:
                print("Shell layout also failed:", e2)
                nx.draw(graph, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight='bold')
    plt.title(f"Mindmap for '{prompt}'")
    plt.tight_layout()
    plt.show()

# Main program
if __name__ == "__main__":
    # List available models for debugging
    print("Available Gemini models:")
    for m in genai.list_models():
        print(f"- {m.name} (methods: {m.supported_generation_methods})")
    # Read prompt from input.txt
    with open("input.txt", "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    mindmap_structure = get_mindmap(prompt)
    print("Parsed mindmap object:", mindmap_structure)
    graph = generate_graph(mindmap_structure)
    visualize_mindmap(graph, prompt)
