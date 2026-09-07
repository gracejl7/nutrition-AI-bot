import gradio as gr
import torch
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer, util
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# add knowledge
facts = [
    "Iron absorption: Non-heme iron (plant-based) is better absorbed when paired with Vitamin C (ascorbic acid).",
    "Iron inhibitors: Phytic acid (in whole grains), polyphenols (in tea/coffee), and calcium can decrease iron absorption.",
    "Calcium sources: Dairy, fortified plant milks, sardines, and leafy greens like kale or bok choy are high in calcium.",
    "Vitamin D: Essential for calcium absorption; found in fatty fish, egg yolks, and produced via sunlight exposure.",
    "Protein Synthesis: Consuming protein with a complete amino acid profile (like quinoa or soy) is vital for muscle repair.",
    "Antioxidants: Berries and dark chocolate contain flavonoids that help reduce oxidative stress in cells.",
    "Electrolytes: Sodium, potassium, and magnesium are critical for hydration and nerve function, especially after exercise.",
    "Healthy Fats: Avocados and walnuts provide omega-3 and omega-6 fatty acids, which support brain health.",
    "Meal Prep Safety: Cooked poultry should be refrigerated within 2 hours and consumed within 3 to 4 days.",
    "Fiber: Soluble fiber (oats, beans) helps lower cholesterol, while insoluble fiber (whole wheat) aids digestion.",
    "Zinc: Found in oysters, red meat, and pumpkin seeds; it plays a key role in immune function and wound healing.",
    "B12 Deficiency: Common in vegan diets; B12 is mainly found in animal products or fortified nutritional yeast.",
    "Potassium/Sodium Balance: High potassium intake (bananas, potatoes) can help offset the blood pressure effects of sodium.",
    "Magnesium: Found in pumpkin seeds, spinach, and almonds; it supports over 300 biochemical reactions in the body."
]

# Pre-compute embeddings for the knowledge base
fact_embeddings = embedder.encode(facts, convert_to_tensor=True)

# RAG
def retrieve_relevant_info(query, top_k=2):
    """Finds the most relevant facts based on semantic similarity."""
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, fact_embeddings, top_k=top_k)
    
    relevant_facts = [facts[hit['corpus_id']] for hit in hits[0]]
    return " ".join(relevant_facts)


def respond(message, history):
    context = retrieve_relevant_info(message)
    
    # Initialize messages with System Prompt and RAG context
    messages = [{
        "role": "system",
        "content": (
            "You are a nutrition expert. You know about nutrients and meal prep. "
            "Use the following context to help answer the user, but do not mention "
            "that you are reading from a list. Context: " + context
        )
    }]

    # Append conversation history
    if history:
        for turn in history:
            messages.append(turn)
            
    # Add the current user message
    messages.append({"role": "user", "content": message})

    # Generate response from Inference Client
    response = client.chat_completion(messages, max_tokens=600)
    return response.choices[0].message.content

# Launch interface
chatbot = gr.ChatInterface(respond)

if __name__ == "__main__":
    chatbot.launch()
