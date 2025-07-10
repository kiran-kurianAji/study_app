from dotenv import load_dotenv
load_dotenv()

import os
import json
import networkx as nx
from pyvis.network import Network
import google.generativeai as genai
from jinja2 import Template

# Configure your Gemini API key
# Make sure GEMINI_API_KEY is set in your environment
gemini_api_key = os.getenv('GEMINI_API_KEY')
if not gemini_api_key:
    raise ValueError("Please set the GEMINI_API_KEY environment variable.")
genai.configure(api_key=gemini_api_key)


def fetch_mindmap(prompt: str) -> dict:
    """
    Call Gemini API to generate a mindmap as JSON structure.
    The prompt should ask for JSON output only.
    Returns a Python dict representing the tree.
    """
    json_format = '''
Output JSON format example:
{
  "name": "Root Topic",
  "children": [
    {
      "name": "Subtopic 1",
      "children": [
        {"name": "Sub-subtopic 1"},
        {"name": "Sub-subtopic 2"}
      ]
    },
    {
      "name": "Subtopic 2"}
  ]
}
'''
    system_prompt = f"You are an assistant that outputs ONLY valid JSON in the specified format. Use this format: {json_format}"
    full_prompt = f"{system_prompt}\nGenerate mindmap for the prompt:{prompt}\n keep node names short"
    model = genai.GenerativeModel('models/gemini-1.5-flash')
    response = model.generate_content(full_prompt, generation_config={"temperature": 0.2})
    text = response.text
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.strip().startswith("json"):
            text = text.strip()[4:]
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    print(text)
    return json.loads(text)


def build_graph(tree: dict, graph=None, parent=None):
    """
    Recursively build a NetworkX graph from the mindmap tree.
    Handles both 'name' and 'topic' as node keys, and 'children' or 'subtopics' as children keys.
    """
    if graph is None:
        graph = nx.DiGraph()

    name = tree.get("name") or tree.get("topic")
    if not name:
        return graph  # skip if no valid node name

    graph.add_node(name)
    if parent:
        graph.add_edge(parent, name)

    # Support both 'children' and 'subtopics'
    children = tree.get("children") or tree.get("subtopics") or []
    for child in children:
        build_graph(child, graph, name)

    return graph


def render_pyvis(graph: nx.Graph, output_file: str = "mindmap.html"):
    """
    Render the given NetworkX graph with PyVis and save to an HTML file.
    """
    net = Network(height="750px", width="100%", directed=True)
    net.from_nx(graph)
    net.show(output_file, notebook=False)
    print(f"Mindmap rendered to {output_file}")


if __name__ == "__main__":
    # Read prompt from input.txt
    with open("input.txt", "r", encoding="utf-8") as f:
        prompt = f.read().strip()
    output_file = "mindmap.html"
    print("Fetching mindmap structure from Gemini...")
    tree = fetch_mindmap(prompt)
    print("Building graph...")
    g = build_graph(tree)
    print("Rendering interactive map...")
    render_pyvis(g, output_file)
