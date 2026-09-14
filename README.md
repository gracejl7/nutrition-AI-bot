🥗 Nutritional AI Assistant
A Retrieval-Augmented Generation (RAG) conversational web application that provides actionable meal prep strategies, high-protein recipes, calorie management advice, and macronutrient breakdowns.

📌 Project Overview
This project builds a specialized chat assistant called Nutritional AI Assistant. It leverages local context from a custom knowledge base alongside a large language model to deliver data-driven meal recommendations with exact macro estimations (calories, protein, carbs, fats), prep times, and grocery aisle routing.

🚀 Key Features
RAG Infrastructure: Combines sentence-transformers and torch to retrieve top context chunks from knowledge.txt based on user queries.

Domain-Specific LLM: Uses Meta's Llama-3.3-70B-Instruct via the Hugging Face Inference API for clear, goal-oriented responses.

Structured Output: Enforces responses containing macro breakdowns (Calories, Protein, Carbs, Fats), preparation time, and grocery categories.

Custom Gradio Interface: Styled using custom CSS, Google Fonts (Poppins, Quicksand), and an emerald/slate theme.
