import gradio as gr
from huggingface_hub import InferenceClient
import torch
from sentence_transformers import SentenceTransformer

client = InferenceClient(model="meta-llama/Llama-3.3-70B-Instruct")
model = SentenceTransformer('all-MiniLM-L6-v2')

with open("knowledge.txt", "r", encoding="utf-8") as f:
    knowledge_text = f.read()

def preprocess_text(text):
    cleaned_text = text.strip()
    chunks = cleaned_text.split("\n\n")
    cleaned_chunks = []
    for chunk in chunks:
        stripped_chunk = chunk.strip()
        if len(stripped_chunk) > 0 and not stripped_chunk.startswith("---") and not stripped_chunk.startswith("==="):
            cleaned_chunks.append(stripped_chunk)
    return cleaned_chunks

cleaned_chunks = preprocess_text(knowledge_text)


def create_embeddings(text_chunks):
    embeddings = model.encode(text_chunks, convert_to_tensor=True)
    print("Total chunks embedded:", len(text_chunks))
    print("Embeddings shape:", embeddings.shape)
    return embeddings

chunk_embeddings = create_embeddings(cleaned_chunks)


def get_top_chunks(query, chunk_embeddings, text_chunks):
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding_normalized = query_embedding / query_embedding.norm()
    chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)
        
    similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized)
        
    k = min(3, len(text_chunks))
    top_indices = torch.topk(similarities, k=k).indices
    top_chunks = [text_chunks[i] for i in top_indices]
    return top_chunks


def respond(message, history):
    top_context_chunks = get_top_chunks(message, chunk_embeddings, cleaned_chunks)
    context_str = "\n\n".join(top_context_chunks)
            
    system_prompt = f"""You are Nutritional AI, a practical and data-driven nutrition assistant specializing in calorie management, high-protein meal prepping, and targeted macronutrient strategies. Your core role is to deliver actionable meal prep ideas, batch-cooking instructions, and smart ingredient swaps that help users hit specific calorie and protein goals without complex cooking. Whenever providing recipes or meal suggestions, always include estimated breakdown values for Calories, Protein, Carbs, and Fats per serving, along with prep time and grocery aisle categories. Write in a motivating, direct tone, avoid medical advice or clinical diets, and ask for the user's primary goal (e.g., fat loss, muscle gain) and dietary restrictions if they aren't provided initially.

Context Information:
{context_str}"""

    messages = [{"role": "system", "content": system_prompt}]
        
   
    for val in history:
        if isinstance(val, (list, tuple)):
            messages.append({"role": "user", "content": val[0]})
            messages.append({"role": "assistant", "content": val[1]})
        elif isinstance(val, dict):
            messages.append(val)

    
    if not messages or messages[-1].get("role") != "user" or messages[-1].get("content") != message:
        messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=1000,
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


nutrition_theme = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate",
    font=[
        gr.themes.GoogleFont("Poppins"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
    font_mono=[
        gr.themes.GoogleFont("Quicksand"),
        "ui-monospace",
        "Consolas",
        "monospace",
    ],
)

custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&family=Quicksand:wght@500;700&display=swap');

body, .gradio-container {
    background-color: #f7fbf4 !important;
    color: #2d3748 !important;
    border-top: 6px solid #BAED91 !important;
}

h1, h2, h3 {
    font-family: 'Quicksand', sans-serif !important;
    color: #2e521c !important;
}

button.primary {
    background-color: #BAED91 !important;
    color: #214011 !important;
    border: 1px solid #d2f0b9 !important;
}

button.primary:hover {
    background-color: #a9e27c !important;
}

.user {
    background-color: #BAED91 !important;
    color: #1a330e !important;
    border-radius: 18px 18px 2px 18px !important;
}

.bot {
    background-color: #ffffff !important;
    border: 1px solid #d2f0b9 !important;
    border-radius: 18px 18px 18px 2px !important;
}
"""

# Launch Application
with gr.Blocks(theme=nutrition_theme, css=custom_css) as chatbot:
    gr.ChatInterface(
        respond,
        title="🥗 Nutritional AI Assistant",
        description="Your practical guide for high-protein meal prep, macros, and calorie management.",
        examples=[
            ["Give me a high-protein, 500-calorie lunch prep idea."],
            ["What are some smart ingredient swaps to cut calories?"],
            ["How do I structure my meal prep for a muscle gain goal?"],
        ],
    )

if __name__ == "__main__":
    chatbot.launch()
