# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import os
import logging
import asyncio
import sys
from werkzeug.utils import secure_filename
import datetime
import json 

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s %(levelname)-8s %(name)-25s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# --- Import Core Chatbot Logic and Utilities ---
try:
    from chatbot_core import initialize_chatbot_core, process_user_turn
    from groq_api import call_groq_llm_final_answer_lc as call_groq_llm_general_purpose 
    from session_manager import ChatSession 
    # from pdf_utils import extract_text_from_pdf # PDF DISABLED
except ImportError as e:
    log.critical(f"CRITICAL_IMPORT_ERROR: Failed to import core modules: {e}.", exc_info=True)
    print(f"CRITICAL_IMPORT_ERROR: {e}. Exiting.", file=sys.stderr)
    sys.exit(1)

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}}, supports_credentials=True)
log.info("CORS configured to allow http://localhost:3000 for /api/* routes with credentials support.")

SESSIONS: dict[str, ChatSession] = {} 
log.info(f"APP_STARTUP: In-memory SESSIONS dictionary initialized (ID: {id(SESSIONS)}).")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads_temp')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
log.info(f"Temporary upload folder set to: {UPLOAD_FOLDER}")


if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
    log.info("APP_INIT: Main process or production mode. Initializing chatbot core...")
    if not initialize_chatbot_core():
        log.critical("APP_INIT_FATAL: Chatbot core system initialization FAILED.")
        sys.exit(1) 
    else:
        log.info("APP_INIT: Chatbot core system initialized successfully.")

@app.route('/api/new_chat', methods=['POST'])
def new_chat_route():
    try:
        session_id = str(uuid.uuid4())
        new_session = ChatSession() 
        SESSIONS[session_id] = new_session
        log.info(f"API_NEW_CHAT: New session created: {session_id}. Initial lang: {new_session.current_language_name}. SESSIONS count: {len(SESSIONS)}")
        return jsonify({
            "sessionId": session_id,
            "messages": [], 
            "languageCode": new_session.current_language,
            "languageName": new_session.current_language_name,
            "activeTVModel": new_session.active_tv_model, 
            "recognizedTVModels": new_session.recognized_tv_models 
        }), 200
    except Exception as e:
        log.error(f"API_NEW_CHAT: Error creating new chat session: {e}", exc_info=True)
        return jsonify({"error": "Failed to create new chat session.", "reply": "Sorry, I couldn't start a new chat."}), 500

@app.route('/api/chat', methods=['POST'])
async def chat_route():
    log.info(f"API_CHAT: Route hit. Current SESSIONS keys count: {len(SESSIONS)}")
    raw_request_data_str = "N/A"
    try:
        raw_request_data_str = request.get_data(as_text=True)
        log.debug(f"API_CHAT: Raw request data: {raw_request_data_str[:500]}...")
    except Exception as e_raw:
        log.warning(f"API_CHAT: Could not get raw request data: {e_raw}")

    session_id_from_request = None
    user_message_text = None
    language_code_from_frontend = 'en' 
    file_obj = None
    parsed_data_for_session_id_check = None
    content_type = request.content_type.lower() if request.content_type else ""
    log.debug(f"API_CHAT: Request Content-Type: {content_type}")

    if 'application/json' in content_type:
        try:
            data_from_flask_parse = request.get_json(silent=True)
            if data_from_flask_parse is None:
                log.error(f"API_CHAT: Flask's request.get_json() returned None. Raw data: {raw_request_data_str[:200]}")
                try: 
                    manual_parsed_data = json.loads(raw_request_data_str)
                    parsed_data_for_session_id_check = manual_parsed_data
                except json.JSONDecodeError as jde:
                    log.error(f"API_CHAT: Manual JSON parse failed: {jde}. Malformed JSON.")
                    return jsonify({"error": "Invalid JSON format.", "reply": "Could not understand request (malformed JSON)."}), 400
            else:
                parsed_data_for_session_id_check = data_from_flask_parse
            
            if not parsed_data_for_session_id_check: 
                log.error("API_CHAT: Received empty or unparsable JSON payload after attempts.")
                return jsonify({"error": "Empty or invalid request body.", "reply": "Request was empty or malformed."}), 400

            session_id_from_request = parsed_data_for_session_id_check.get('sessionId')
            user_message_text = parsed_data_for_session_id_check.get('message')
            language_code_from_frontend = parsed_data_for_session_id_check.get('language', language_code_from_frontend)
        except Exception as e_json: 
            log.error(f"API_CHAT: Unexpected error processing JSON request body: {e_json}", exc_info=True)
            return jsonify({"error": "Error processing JSON request.", "reply": "Could not understand request."}), 400

    elif 'multipart/form-data' in content_type:
        try:
            parsed_data_for_session_id_check = request.form
            session_id_from_request = request.form.get('sessionId')
            user_message_text = request.form.get('message') 
            language_code_from_frontend = request.form.get('language', language_code_from_frontend)
            file_obj = request.files.get('file')
        except Exception as e_form:
            log.error(f"API_CHAT: Error parsing multipart/form-data: {e_form}", exc_info=True)
            return jsonify({"error": "Invalid form data.", "reply": "Could not understand request form."}), 400
    else:
        log.error(f"API_CHAT: Unsupported Content-Type: {content_type}")
        return jsonify({"error": "Unsupported request format.", "reply": "Request format not supported."}), 415

    if not session_id_from_request:
        log.error(f"API_CHAT: CRITICAL - 'sessionId' is missing. Parsed data: {parsed_data_for_session_id_check}")
        return jsonify({"error": "Session ID is required.", "reply": "Your session ID is missing."}), 400

    session = SESSIONS.get(session_id_from_request)
    if not session:
        log.error(f"API_CHAT: Invalid session ID: '{session_id_from_request}'.")
        return jsonify({"error": "Invalid session ID.", "reply": "Your session is invalid. Please start a new chat."}), 400

    session.set_language(language_code_from_frontend) 
    log.debug(f"API_CHAT: Session '{session_id_from_request}' lang: '{session.current_language_name}'.")

    if user_message_text and user_message_text.strip():
        session.add_to_history("user", user_message_text)

    input_for_core = user_message_text.strip() if user_message_text and user_message_text.strip() else ""

    if file_obj:
        filename = secure_filename(file_obj.filename)
        if not filename: 
            log.warning(f"API_CHAT: File upload invalid filename. Session: {session_id_from_request}")
            ack_text = "(System note: Uploaded file had an invalid name and was ignored.)"
            session.add_to_history("system", ack_text) 
            input_for_core = ack_text if not input_for_core else f"{input_for_core} {ack_text}"
        else:
            unique_filename = f"{session_id_from_request}_{uuid.uuid4().hex[:8]}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            try:
                file_obj.save(filepath)
                log.info(f"API_CHAT: File '{filename}' (as '{unique_filename}') uploaded. Session: {session_id_from_request}")

                if filename.lower().endswith(".pdf"):
                    log.warning(f"API_CHAT: PDF processing for '{filename}' is currently disabled.")
                    pdf_disabled_ack_context = (
                        f"The PDF document '{filename}' was received, but PDF processing is currently disabled. "
                        f"I cannot analyze its content at this time."
                    )
                    pdf_disabled_ack_msg = await call_groq_llm_general_purpose(
                        pdf_disabled_ack_context,
                        session.current_language_name, session.last_detected_dialect_info,
                        memory_messages=session.get_lc_memory_messages(),
                    ) or f"PDF '{filename}' received, but processing is disabled."
                    session.add_to_history("system", pdf_disabled_ack_msg)
                    if not input_for_core: input_for_core = pdf_disabled_ack_msg
                else: 
                    non_pdf_ack_context = (
                        f"The file '{filename}' was received. "
                        f"I am currently set up to acknowledge non-PDF files but not process their content deeply."
                    )
                    ack_msg_localized = await call_groq_llm_general_purpose(
                        non_pdf_ack_context, session.current_language_name, 
                        session.last_detected_dialect_info, memory_messages=session.get_lc_memory_messages(),
                    ) or non_pdf_ack_context 
                    session.add_to_history("system", ack_msg_localized)
                    if not input_for_core: input_for_core = ack_msg_localized
            except Exception as e_file:
                log.error(f"API_CHAT: Error during file upload for '{filename}': {e_file}", exc_info=True)
                err_file_ctx = f"An error occurred while handling the uploaded file '{filename}'. Please try again."
                error_response_localized = await call_groq_llm_general_purpose(
                    err_file_ctx, session.current_language_name, 
                    session.last_detected_dialect_info, memory_messages=session.get_lc_memory_messages(),
                ) or err_file_ctx 
                session.add_to_history("assistant", error_response_localized)
                return jsonify({"reply": error_response_localized, "sessionId": session_id_from_request, 
                                "languageCode": session.current_language, "languageName": session.current_language_name,
                                "activeTVModel": session.active_tv_model, "recognizedTVModels": session.recognized_tv_models}), 200
            finally: 
                if 'filepath' in locals() and os.path.exists(filepath):
                    try: os.remove(filepath)
                    except Exception as e_rem: log.error(f"API_CHAT: Error removing temp file {filepath}: {e_rem}")

    if not input_for_core: 
        log.warning(f"API_CHAT: No effective text input after file processing. Session: {session_id_from_request}.")
        empty_input_ctx = "It seems your message was empty or only contained a file I couldn't turn into a query. How can I help you today?"
        final_bot_reply_localized = await call_groq_llm_general_purpose(
            empty_input_ctx, session.current_language_name, 
            session.last_detected_dialect_info, memory_messages=session.get_lc_memory_messages(),
        ) or "How can I help you?" 
        session.add_to_history("assistant", final_bot_reply_localized)
        return jsonify({"reply": final_bot_reply_localized, "sessionId": session_id_from_request, 
                        "languageCode": session.current_language, "languageName": session.current_language_name,
                        "activeTVModel": session.active_tv_model, "recognizedTVModels": session.recognized_tv_models}), 200

    log.debug(f"API_CHAT: Final input for core: '{input_for_core[:150]}' (Sess: {session_id_from_request})")

    try:
        final_bot_reply = await process_user_turn(session, input_for_core)
        if final_bot_reply:
            session.add_to_history("assistant", final_bot_reply) 
        else: 
            log.error(f"API_CHAT: Core processing returned empty. Session: {session_id_from_request}.")
            fallback_err_ctx = "I'm having trouble formulating a response. Please try rephrasing."
            final_bot_reply = await call_groq_llm_general_purpose(
                fallback_err_ctx, session.current_language_name, 
                session.last_detected_dialect_info, memory_messages=session.get_lc_memory_messages(),
            ) or fallback_err_ctx 
            session.add_to_history("assistant", final_bot_reply)

        log.info(f"API_CHAT: Sess {session_id_from_request} - Reply Lang: {session.current_language_name}, Reply: '{(str(final_bot_reply)[:100])}'")
        return jsonify({
            "reply": final_bot_reply, "sessionId": session_id_from_request,
            "languageCode": session.current_language, "languageName": session.current_language_name,
            "activeTVModel": session.active_tv_model, "recognizedTVModels": session.recognized_tv_models 
        }), 200

    except Exception as e_core: 
        log.error(f"API_CHAT: Exception in process_user_turn. Session {session_id_from_request}: {e_core}", exc_info=True)
        user_input_preview = (str(user_message_text)[:30] + '...') if user_message_text else 'your request'
        # Corrected f-string:
        core_error_ctx = f"An unexpected error occurred processing your request ('{user_input_preview}'). Please try again."
        
        error_reply_localized = "An unexpected internal error occurred. Please try again later." 
        try:
            error_reply_localized = await call_groq_llm_general_purpose(
                core_error_ctx, session.current_language_name, 
                session.last_detected_dialect_info, memory_messages=session.get_lc_memory_messages(),
            ) or core_error_ctx # Use the English context if LLM fails for localization
        except Exception as e_llm_err:
            log.error(f"API_CHAT: Further exception localizing core error message: {e_llm_err}", exc_info=True)
            # error_reply_localized remains the hardcoded one
        
        session.add_to_history("assistant", error_reply_localized) 
        return jsonify({
            "reply": error_reply_localized, "sessionId": session_id_from_request,
            "error_detail": "Core processing error.", 
            "languageCode": session.current_language, "languageName": session.current_language_name,
            "activeTVModel": session.active_tv_model, "recognizedTVModels": session.recognized_tv_models
        }), 500


@app.route('/api/chat_history', methods=['GET'])
def get_chat_history_route():
    try:
        history_summary = []
        active_sessions_with_history = {sid: sess for sid, sess in SESSIONS.items() if sess.get_ui_history()}
        sorted_sids = sorted(
            active_sessions_with_history.keys(),
            key=lambda sid: active_sessions_with_history[sid].get_ui_history()[-1]['timestamp'] 
                            if active_sessions_with_history[sid].get_ui_history() 
                            else datetime.datetime.min.isoformat(),
            reverse=True
        )
        for sid in sorted_sids:
            sess = SESSIONS[sid]
            session_ui_history = sess.get_ui_history() 
            title = f"Chat {sid[:8]}" 
            first_user_message = next((h['content'] for h in session_ui_history if h['role'] == 'user'), None)
            
            if sess.active_tv_model: 
                title_prefix = f"TV: {sess.active_tv_model[:20]}"
                if first_user_message:
                    title = f"{title_prefix} - {first_user_message[:20]}"
                else:
                    title = title_prefix
            elif first_user_message:
                title = first_user_message[:30]
            elif session_ui_history : 
                title = session_ui_history[0]['content'][:30]
            
            title += "..." if len(title) >= 30 and not title.endswith("...") else ""
            
            history_summary.append({
                "id": sid, "title": title,
                "activeTVModel": sess.active_tv_model, 
                "recognizedTVModels": sess.recognized_tv_models 
            })
        return jsonify(history_summary), 200
    except Exception as e:
        log.error(f"API_HISTORY: Error fetching chat history: {e}", exc_info=True)
        return jsonify({"error": "Failed to fetch chat history."}), 500

@app.route('/api/chat_session/<session_id>', methods=['GET'])
def get_chat_session_route(session_id: str):
    session = SESSIONS.get(session_id)
    if not session:
        log.warning(f"API_SESSION: Session not found: {session_id}.")
        return jsonify({"error": "Session not found. Please start a new chat."}), 404
    try:
        ui_messages = session.get_ui_history()
        formatted_messages = [{
            "sender": 'bot' if entry["role"] == 'assistant' else entry["role"],
            "text": entry["content"], "timestamp": entry["timestamp"], "type": "text"
        } for entry in ui_messages]
        log.info(f"API_SESSION: Loaded {session_id}. Lang: {session.current_language_name}, ActiveModel: {session.active_tv_model}")
        return jsonify({
            "sessionId": session_id, "messages": formatted_messages,
            "languageCode": session.current_language, "languageName": session.current_language_name,
            "activeTVModel": session.active_tv_model, "recognizedTVModels": session.recognized_tv_models
        }), 200
    except Exception as e:
        log.error(f"API_SESSION: Error loading session {session_id}: {e}", exc_info=True)
        return jsonify({"error": f"Failed to load session {session_id}."}), 500

if __name__ == '__main__':
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true" or not app.debug:
        log.info("APP_INIT (Flask Main/Production): Initializing chatbot core...")
        if not initialize_chatbot_core():
            log.critical("APP_INIT_FATAL (Flask Main/Production): Chatbot core system initialization FAILED. Exiting.")
            sys.exit(1)
        else:
            log.info("APP_INIT (Flask Main/Production): Chatbot core system initialized successfully.")
    log.info("Starting Flask backend server...")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)