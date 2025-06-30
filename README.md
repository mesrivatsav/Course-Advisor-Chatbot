Course Advisor Chatbot - README
Project Title
🎓 Course Advisor Chatbot

An Open-Source AI Chatbot for Course Advising in Ireland
By Rakesh Srivatsav Velluvayala

Key Features
- 🔍 Retrieval-Augmented Generation (RAG)
- 🤖 Multi-Model Support: LLaMA2, LLaMA3, Mistral, Phi, and Gemma
- 🗺️ Wide University Coverage
- 🧪 RAGAS Evaluation
- 🖥️ Streamlit Interface

System Architecture
1. Web Scraping using LangChain & BeautifulSoup
2. Chunking & Embeddings using LangChain + nomic-embed-text via Ollama
3. Embeddings stored in ChromaDB
4. Streamlit interface for chat interaction

Installation
1. Clone the repo
    git clone https://github.com/mesrivatsav/Course-Advisor-Chatbot.git
2. Install dependencies
    pip install -r requirements.txt
3. Run Ollama with desired model
    ollama run llama3
4. Launch Streamlit app
    streamlit run app.py

Evaluation
Evaluated with RAGAS on:
- Answer Correctness
- Answer Relevancy
- Faithfulness
- Context Precision
- Context Recall

LLaMA 3 performed best across most metrics.
Universities Included
- Technological University of the Shannon (TUS)
- Technological University Dublin (TUD)
- Munster Technological University (MTU)
- South East Technological University (SETU)
- Atlantic Technological University (ATU)
Sample Use Cases
- What are the entry requirements for the BSc in Computer Science at TUS?
- Which process validation courses are offered at TUS Moylish campus?
- Does ATU offer any postgraduate courses in marketing?

Thesis Report
“An Open-Source AI Chatbot for Course Advising: Leveraging Retrieval Augmented Generation”
Master of Science in Data Analytics – 2024
Author: Rakesh Srivatsav Velluvayala
Future Improvements
- Add GPT-4/Claude for comparison
- Enhance hallucination handling
- Add more universities & languages
- Deploy publicly as web app or API

Acknowledgements
Thanks to thesis supervisor David Leonard and faculty at Technological University of the Shannon.
Contact
📧 rakeshsrivatsav@gmail.com
🔗 https://www.linkedin.com/in/mesrivatsav