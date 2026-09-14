A data-driven nutrition chatbot built using HuggingFace, Gradio, Sentence Transformers, and RAG
Knowledge base was generated using Claude.
This AI chatbot assists users with calorie management, meal preparation, and ingredient substitutions. 
It features Meta Llama 3.3 70B Instruct Model through HuggingFace 
It utilizes retrieval augmented generation (RAG) in order to utilize knowledge from the knowledge.txt file.
It utilizes semantic search using all-MiniLM-L6-v2 for sentence embeddings
Cosine similarity retrieval identifies the three most relevant knowledge chunks for each question. 
The history of the conversation is taken into account in order to maintain content throughout the chat.
