# ai_model_bomare
# AI TV Troubleshooting Chatbot

This project is an AI-powered chatbot designed to assist users with TV troubleshooting, general TV-related questions, and information retrieval from PDF documents. It leverages Large Language Models (LLMs) via the Groq API for fast inference, LangChain for structuring interactions and memory, and a FAISS vector store for Retrieval Augmented Generation (RAG) from a knowledge base of TV issues.

## Features

*   **Multi-Turn Conversations:** Maintains context using LangChain memory.
*   **Multilingual Support:**
    *   Detects user language (English, French, Arabic with Darija dialect support).
    *   Responds in the user's detected language.
    *   Uses a dedicated microservice (configurable) for English-to-Darija translation and Darija detection.
*   **Intent Recognition:**
    *   **General Questions:** Answers general knowledge queries.
    *   **Standard TV Troubleshooting:** Provides generic troubleshooting steps when no specific TV model is identified.
    *   **Specific TV Troubleshooting (RAG):** Retrieves model-specific troubleshooting steps from a JSON knowledge base (`data.json`) using semantic search (FAISS + Sentence Transformers).
    *   **Media Requests:** Can provide images (block diagrams, motherboard, key components) and component lists based on the TV model, sourcing from `frontend/public/troubleshooting/` and `key_components.json`.
    *   **Step Explanation:** LLM elaborates on retrieved troubleshooting steps to make them more user-friendly.
*   **TV Model Management:**
    *   Recognizes TV model names mentioned by the user.
    *   Maintains a list of recognized models within a session.
    *   Focuses on an "active" TV model for specific queries.
    *   Allows users to switch focus between recognized models.
*   **Contextual Understanding:**
    *   Uses session state flags to manage expectations (e.g., waiting for a model name, yes/no confirmation).
    *   LLM-powered follow-up intent classification to interpret user replies like "yes," "tell me more," or model provisions.
*   **PDF Document Interaction (Currently Disabled by Default):**
    *   Functionality to upload and extract text from PDF documents. The chatbot can then answer questions based on the PDF's content. (This feature is currently disabled in the provided code due to a missing `pypdf` dependency but can be re-enabled).
*   **Session Management:**
    *   Supports multiple concurrent chat sessions.
    *   Stores chat history for users to revisit.
*   **Modular Design:** Separates concerns into different Python modules (API, core logic, handlers, utilities).

## Core Approach

1.  **User Input & Language Detection:**
    *   The Flask backend receives user input via an API endpoint.
    *   The `language_handler` detects the input language and any explicit language switch requests. The session language is set accordingly.

2.  **Session State & Expectation Management:**
    *   The `ChatSession` object (`session_manager.py`) tracks the current state: active language, recognized TV models, current active TV model, ongoing problem description, and specific expectations (e.g., if the bot just asked for a model number).

3.  **Intent Classification & Routing (`chatbot_core.py`, `initial_interaction_handler.py`, `session_flow_handler.py`):**
    *   **Initial Query:** If no specific expectation is set, `initial_interaction_handler.py` uses an LLM (`classify_main_intent_and_extract_model_lc`) to determine the primary intent (General Question, Standard TV Troubleshooting, Specific TV Troubleshooting, Media Request) and extracts any mentioned TV models.
    *   **Ongoing Conversation:** If an expectation is set (e.g., bot asked a question), `session_flow_handler.py` uses an LLM (`classify_follow_up_intent_lc`) to interpret the user's response (e.g., "yes," "no," providing a model name).
    *   Based on the intent and session state, the query is routed to the appropriate handler:
        *   `knowledge_handler.py`: For general questions.
        *   `troubleshooting_handler.py`:
            *   `handle_standard_tv_troubleshooting`: Provides generic steps and asks for a model.
            *   `handle_specific_tv_troubleshooting`: Performs RAG using the active TV model and user's problem. It then uses an LLM to explain the retrieved steps.
        *   `image_handler.py`: Retrieves and formats media (images, component lists) for the active TV model.

4.  **Retrieval Augmented Generation (RAG) for Specific Troubleshooting:**
    *   User's problem description (translated to English if necessary) is enhanced using HyDE (Hypothetical Document Embeddings via LLM).
    *   The resulting query is used to search a FAISS vector index built from the "issue" descriptions in `data.json` (`vector_search.py`).
    *   Top semantic matches are retrieved, then filtered to find the best match for the user's `active_tv_model`.
    *   The steps from the matched guide are then explained by an LLM.

5.  **LLM Interaction (Groq API & LangChain):**
    *   `groq_api.py` centralizes all calls to the Groq API using `ChatGroq` from LangChain.
    *   Functions are provided for:
        *   Generating final conversational answers.
        *   Translating text.
        *   Classifying intents.
        *   Generating HyDE queries.
    *   LangChain's `ConversationBufferWindowMemory` is used within each `ChatSession` to provide conversational history to the LLM.

6.  **Response Generation & Localization:**
    *   Handlers typically generate a core response in English.
    *   `chatbot_core.py` localizes this English response into the user's `session.current_language_name` using an LLM translation call, with special handling for Darija via a microservice.
    *   Some handlers (like `image_handler.py` for complex media presentations) may generate fully localized Markdown directly.

7.  **Frontend Interaction:**
    *   A React frontend (`frontend/`) communicates with the Flask backend via REST APIs.
    *   It displays messages, handles user input (text and file uploads), and manages chat history display.

## Project Structure
chatbot/
├── backend/
│ ├── uploads_temp/ # Temporary storage for file uploads
│ ├── venv/ # Python virtual environment
│ ├── app.py # Flask application, API routes
│ ├── chatbot_core.py # Core chatbot orchestration logic
│ ├── data.json # Knowledge base for TV models and troubleshooting
│ ├── key_components.json # Detailed component data for specific models
│ ├── language_keywords.json # Keywords for language/intent heuristics
│ ├── groq_api.py # Handles all LLM calls via Groq
│ ├── image_handler.py # Logic for media retrieval and formatting
│ ├── initial_interaction_handler.py # Handles first user turn or new topics
│ ├── knowledge_handler.py # Handles general knowledge questions
│ ├── language_handler.py # Language detection, Darija services
│ ├── pdf_utils.py # PDF text extraction (currently disabled)
│ ├── session_flow_handler.py # Handles ongoing conversation flow, follow-ups
│ ├── session_manager.py # Defines the ChatSession class
│ ├── troubleshooting_handler.py # Handles troubleshooting logic (standard & specific)
│ ├── utils.py # Utility functions (e.g., model extraction regex)
│ ├── vector_search.py # FAISS index creation and semantic search
│ ├── Dockerfile # For containerizing the backend
│ ├── requirements.txt # Python dependencies
│ └── .env # Environment variables (GROQ_API_KEY, etc.)
└── frontend/
├── public/
│ ├── troubleshooting/ # Directory for TV model images
│ └── ... # Other static assets
├── src/
│ ├── components/
│ │ ├── ChatArea.js
│ │ ├── InputBar.js
│ │ ├── Sidebar.js
│ │ └── TopBar.js
│ ├── pages/
│ │ └── ChatInterface.js # Main UI component
│ ├── App.js
│ ├── index.js
│ └── ChatInterface.css
├── package.json
└── ...

## Setup and Installation

### Prerequisites

*   Python 3.10+
*   Node.js and npm (or yarn) for the frontend
*   A Groq API Key

### Backend Setup

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd chatbot/backend
    ```

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    pip install "flask[async]" # Ensure Flask async support is installed
    ```
    *Note: If `pypdf` was causing issues and you've disabled PDF functionality, you can remove it from `requirements.txt` for now.*

4.  **Set up environment variables:**
    Create a `.env` file in the `backend/` directory with your Groq API key:
    ```env
    GROQ_API_KEY="your_groq_api_key_here"
    # Optional: For Darija microservices if you have them running
    # DZIRIBERT_DETECTION_SERVICE_URL="http://localhost:8001/process_darija"
    # ENG_TO_DARIJA_TRANSLATION_SERVICE_URL="http://localhost:8001/translate_en_to_darija"
    # Optional: To change data file names
    # RAG_DATA_FILE="my_data.json"
    # COMPONENTS_DATA_FILE="my_components.json"
    ```

5.  **Data Files:**
    *   Ensure `data.json` (for RAG) and `key_components.json` are present in the `backend/` directory with the correct structure.
    *   Ensure `language_keywords.json` is present for language/intent heuristics.

### Frontend Setup

1.  **Navigate to the frontend directory:**
    ```bash
    cd ../frontend 
    ```
    (Assuming you are in `chatbot/backend`)
    or from the root:
    ```bash
    cd chatbot/frontend
    ```

2.  **Install Node.js dependencies:**
    ```bash
    npm install
    # or if you use yarn:
    # yarn install
    ```

3.  **Place Images:**
    *   Ensure your TV model images (e.g., `EL.RT2864-FG48_block_diagram.png`) are placed in the `frontend/public/troubleshooting/` directory. The paths in your `data.json` or `key_components.json` should correspond to these filenames.

## Running the Application

1.  **Start the Backend (Flask Server):**
    Open a terminal, navigate to the `backend/` directory, activate your virtual environment, and run:
    ```bash
    python app.py
    ```
    The backend will typically run on `http://localhost:5000`. Check the console output for the exact address.

2.  **Start the Frontend (React Development Server):**
    Open another terminal, navigate to the `frontend/` directory, and run:
    ```bash
    npm start
    # or if you use yarn:
    # yarn start
    ```
    The frontend will typically open in your browser at `http://localhost:3000`.

3.  **Interact with the Chatbot:**
    Open `http://localhost:3000` in your web browser to use the chatbot.

## Key Files for Customization

*   **`backend/data.json`:** The primary knowledge base for TV models, their common issues, and troubleshooting steps. This is used for RAG.
*   **`backend/key_components.json`:** Contains detailed information about key components for specific TV models, often including references to specific diagram images.
*   **`backend/language_keywords.json`:** Used for keyword-based heuristics in language detection and simple intent spotting (e.g., reset commands).
*   **`backend/utils.py` (`extract_tv_model_from_query`):** If you need to refine how TV model names are extracted via regex.
*   **`backend/groq_api.py`:** To change LLM models, temperatures, or fine-tune system prompts for LLM calls.
*   **Prompts within handler files (e.g., `troubleshooting_handler.py`, `image_handler.py`):** The English context provided to the LLM in these files heavily influences the bot's behavior and the quality of explanations or generated text.
*   **`frontend/public/troubleshooting/`:** Store your TV-related images here.
*   **`.env`:** For API keys and other environment-specific configurations.

## Troubleshooting Common Issues

*   **`ModuleNotFoundError`:** Ensure all dependencies in `requirements.txt` are installed in your active virtual environment. If you added a new import, add the library to `requirements.txt` and reinstall.
*   **Circular Imports:** If you encounter these, a common solution is to move shared functions/classes to a utility module (like `utils.py`).
*   **Groq API Errors:** Check your `GROQ_API_KEY` in `.env`. Ensure the model names in `groq_api.py` are valid and not deprecated. Check the Groq dashboard for any API usage issues.
*   **Images Not Displaying:**
    *   Verify the exact image filenames in `data.json` or `key_components.json` match the files in `frontend/public/troubleshooting/` (case-sensitive).
    *   Check the `IMAGE_BASE_PATH_USER_MSG` in `chatbot_core.py` (should be `"troubleshooting/"`).
    *   Use browser developer tools (Network tab, Console) to inspect failed image requests and paths.
*   **Incorrect RAG Results:**
    *   Add detailed logging in `vector_search.py` and `troubleshooting_handler.py` to see the query being used, the semantic search candidates, and why a specific document might not be chosen.
    *   The phrasing of "issue" texts in `data.json` is crucial for good semantic matches.
    *   Tune `NUMBER_OF_SEMANTIC_CANDIDATES_TO_CHECK` in `troubleshooting_handler.py`.

## Future Enhancements

*   More robust state management for complex conversational flows.
*   Support for more languages.
*   Fine-tuning embedding models or LLMs for domain-specific knowledge.
*   Integration with external APIs for real-time data (e.g., product availability).
*   User authentication and personalized history.
*   Deployment using Docker and a production-grade WSGI/ASGI server.

