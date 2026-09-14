🥗 Nutritional AI Assistant
A Retrieval-Augmented Generation (RAG) conversational web application that provides actionable meal prep strategies, high-protein recipes, calorie management advice, and macronutrient breakdowns.

📌 Project Overview
This project builds a specialized chat assistant called Nutritional AI Assistant. It leverages local context from a custom knowledge base alongside a large language model to deliver data-driven meal recommendations with exact macro estimations (calories, protein, carbs, fats), prep times, and grocery aisle routing.

🚀 Key Features
RAG Infrastructure: Combines sentence-transformers and torch to retrieve top context chunks from knowledge.txt based on user queries.

Domain-Specific LLM: Uses Meta's Llama-3.3-70B-Instruct via the Hugging Face Inference API for clear, goal-oriented responses.

Structured Output: Enforces responses containing macro breakdowns (Calories, Protein, Carbs, Fats), preparation time, and grocery categories.

Custom Gradio Interface: Styled using custom CSS, Google Fonts (Poppins, Quicksand), and an emerald/slate theme.

🛠️ Getting Started
Prerequisites
Python 3.9+

A Hugging Face account and API token with access to meta-llama/Llama-3.3-70B-Instruct

Installation
Clone the repository

Bash
git clone https://github.com/your-username/nutritional-ai-assistant.git
cd nutritional-ai-assistant
Install dependencies

Bash
pip install gradio huggingface_hub torch sentence-transformers
Set up the knowledge base
Create a file named knowledge.txt in the root directory and add your custom nutrition, meal prep, or ingredient text.

Set your API token
Set your Hugging Face API token in your environment:

Bash
# On macOS/Linux
export HF_TOKEN="your_huggingface_token_here"

# On Windows (Command Prompt)
set HF_TOKEN="your_huggingface_token_here"
Run the application

Bash
python app.py
💬 Usage
Once launched, open the local URL provided by Gradio (typically [http://127.0.0.1:7860](http://127.0.0.1:7860)).

Example Queries
"Give me a high-protein, 500-calorie lunch prep idea."

"What are some smart ingredient swaps to cut calories?"

"How do I structure my meal prep for a muscle gain goal?"

🙋 Support & Help
App Issues: File an issue in the project repository for bugs or feature requests.

Model Access: Refer to the Hugging Face Hub Documentation for token and API issues.

UI Customization: Refer to the Gradio Documentation for interface modifications.

👥 Maintainers & Credits
Core Maintainer: Open-source contributor project.

Models Used:

Meta Llama-3.3-70B-Instruct by Meta

all-MiniLM-L6-v2 by Sentence-Transformers
