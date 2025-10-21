
### **Project: Semantic Book Recommender Dashboard**

**Description:**
Developed an intelligent book recommendation system that combines natural language processing (NLP), vector search, and emotion-based filtering to provide personalized suggestions. The system embeds book descriptions using OpenAI embeddings, performs semantic similarity retrieval via ChromaDB, and visualizes results through an interactive **Gradio dashboard**. It supports emotion-aware ranking (joy, sadness, fear, anger, surprise) and category filtering, enabling users to discover books that align with their interests and moods.

**Key Features:**

* Built an **end-to-end pipeline** for data ingestion, text preprocessing, feature extraction, and vector-based retrieval.
* Designed an **interactive Gradio UI** allowing users to input descriptive queries and view semantic matches with thumbnails and summaries.
* Implemented **sentiment and text classification** using pre-trained transformers to label emotional tones of book descriptions.
* Used **vector embeddings** and **cosine similarity search** via **LangChain + Chroma** for semantic retrieval.
* Created **emotion-aware ranking logic** to dynamically prioritize recommendations based on user mood.
* Enhanced interpretability and performance through **data visualization and exploratory analysis** (Pandas, Matplotlib, Seaborn).

**Technologies & Tools Used:**

* **Languages:** Python
* **Libraries/Frameworks:**

  * **Data & NLP:** Pandas, NumPy, Scikit-learn, NLTK, spaCy, Hugging Face Transformers
  * **Vector Search & LLM Integration:** LangChain, Chroma, OpenAIEmbeddings
  * **Visualization:** Matplotlib, Seaborn
  * **Web/UI:** Gradio (Blocks API, themes, dynamic inputs/outputs)
  * **Environment Management:** dotenv
* **Data Sources:** `books_with_emotions.csv`, `tagged_descriptions.txt`
* **Techniques:** Sentiment Analysis, Text Classification, Embedding Similarity Search, Emotion Scoring, Data Cleaning, Feature Engineering
* **Deployment:** Gradio Interface with Local/Cloud Hosting

**Impact:**
Delivered a real-time, emotion-driven book recommendation platform that integrates NLP pipelines with interactive dashboards—demonstrating capabilities in **semantic retrieval, ML model integration, and user-centered design**.

---
![alt text](https://github.com/varunjose/SemanticBookRecommender/blob/main/result.png)