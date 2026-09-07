import gradio as gr
from huggingface_hub import InferenceClient
import torch
from sentence_transformers import SentenceTransformer
client = InferenceClient("Qwen/Qwen2.5-7B-Instruct")

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
model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(text_chunks):
    chunk_embeddings = model.encode(text_chunks, convert_to_tensor=True)
    print("Total chunks embedded:", len(text_chunks))
    print("Embeddings shape:", chunk_embeddings.shape)
    return chunk_embeddings
    
chunk_embeddings = create_embeddings(cleaned_chunks)

def get_top_chunks(query, chunk_embeddings, text_chunks):
    query_embedding = model.encode(query, convert_to_tensor=True)
    query_embedding_normalized = query_embedding / query_embedding.norm()
    chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)
    similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized)
    
    top_indices = torch.topk(similarities, k=3).indices
    top_chunks = [text_chunks[i] for i in top_indices]
    return top_chunks
    



def respond(message, history):
    top_context_chunks = get_top_chunks(message, chunk_embeddings, cleaned_chunks)
    context_str = "\n\n".join(top_context_chunks)

    system_prompt = (
        "You are a nutrition expert. You know about nutrients and meal prep. "
            "Use the following context to help answer the user, but do not mention "
            "that you are reading from a list. Context: " + context
    )

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
