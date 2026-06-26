# Chatbot Project

This is a conversational AI chatbot to assist citizens with public service queries. The chatbot is built using **Streamlit**, **Python**, and **SQLite**.

## Project Structure

- `backend/`: Contains backend logic and API handlers.
- `frontend/`: Contains the Streamlit-based UI for interacting with the chatbot.
- `database/`: SQLite database initialization and configuration scripts.
- `tests/`: Test cases for the application.

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Navigate to the project directory:
   ```bash
   cd chatbot_project
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Initialize the database:
   ```bash
   python database/db_config.py
   ```

5. Run the Streamlit application:
   ```bash
   streamlit run frontend/streamlit_app.py
   ```

6. Interact with the chatbot through the web interface.

## Future Improvements

- Integration with a generative AI model for more dynamic responses.
- Adding support for multiple languages.
- Implementing a fallback mechanism for human agent intervention.
- Logging and analytics for user interactions.

## Embedding Preparation

The backend can now prepare preprocessed data for an in-memory vector database:

```python
from backend.embeddings import chunk_and_embed

records = chunk_and_embed(
    processed_data,
    source="uploaded-document.pdf",
)
```

`chunk_and_embed` returns records with chunk text, word positions, metadata, and
the embedding vector. It does not save anything yet, so the database layer can
persist those records when it is ready.

Set `OPENAI_API_KEY` before calling the embedding model. Optional overrides:
`OPENAI_EMBEDDING_MODEL` and `OPENAI_EMBEDDINGS_URL`.

## License
Licensed under the MIT License.
