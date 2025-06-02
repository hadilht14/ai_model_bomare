# backend/chatbot_core.py
import asyncio
import logging
import os
import sys
import json
import re # For parsing NEW_PROBLEM_SUGGESTION from LLM response
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage # For type hinting

try:
    # Handlers
    from initial_interaction_handler import handle_initial_query
    from session_flow_handler import handle_ongoing_session_turn
    
    # LLM and API related
    from groq_api import (
        call_groq_llm_final_answer_lc, 
        translate_text_lc
        # classify_follow_up_intent_lc is used within session_flow_handler
        # classify_main_intent_and_extract_model_lc is used within initial_interaction_handler
    )
    # Data and Search
    from vector_search import load_data, create_faiss_index
    # Session and Language
    from session_manager import ChatSession 
    from language_handler import (
        detect_language_and_intent, 
        get_language_name, 
        get_localized_keywords, 
        # SUPPORTED_LANGUAGES_MAP, # Not directly used here, but by language_handler itself
        translate_english_to_darija_via_service
    )
    # `utils.py` is not directly imported here as its functions are used by the handlers.
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR in chatbot_core.py: {e}. Application will likely fail.", file=sys.stderr)
    sys.exit(1) # Exit if critical components can't be imported

log = logging.getLogger(__name__)

# --- Global Configuration & State ---
DATA_FILE_NAME = os.getenv("RAG_DATA_FILE", "data.json")
COMPONENTS_DATA_FILE_NAME = os.getenv("COMPONENTS_DATA_FILE", "key_components.json")
IMAGE_BASE_PATH_USER_MSG = "troubleshooting/" 

data_store = None
index_store = None
text_to_original_data_idx_map_store = None
components_data_store: list = [] 
is_core_initialized = False

def initialize_chatbot_core():
    global data_store, index_store, text_to_original_data_idx_map_store, components_data_store
    global is_core_initialized, DATA_FILE_NAME, COMPONENTS_DATA_FILE_NAME
    
    if is_core_initialized:
        log.info("Chatbot core already initialized.")
        return True
    
    load_dotenv() 
    log.info("--- Initializing Chatbot Core System ---")
    
    current_module_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_path = os.path.join(current_module_dir, DATA_FILE_NAME)
    components_data_file_path = os.path.join(current_module_dir, COMPONENTS_DATA_FILE_NAME)

    if not os.getenv("GROQ_API_KEY"):
        log.critical("CRITICAL: GROQ_API_KEY is not set. LLM calls will fail.")
        # Consider returning False or raising an error to halt initialization
        # return False
    if not os.path.exists(data_file_path):
        log.critical(f"CRITICAL: RAG data file '{data_file_path}' is missing.")
        return False
    
    current_data_store = load_data(data_file_path)
    if not current_data_store:
        log.critical(f"CRITICAL: No RAG data loaded from {data_file_path}.")
        return False
    data_store = current_data_store
    log.info(f"Loaded {len(data_store)} RAG entries from {data_file_path}.")

    current_index_store, current_text_to_original_data_idx_map_store = create_faiss_index(data_store)
    if not current_index_store:
        log.critical("CRITICAL: FAISS RAG index creation failed.")
        return False
    index_store = current_index_store
    text_to_original_data_idx_map_store = current_text_to_original_data_idx_map_store
    log.info("FAISS RAG index created successfully.")

    if os.path.exists(components_data_file_path):
        try:
            with open(components_data_file_path, 'r', encoding='utf-8') as f:
                loaded_components_data = json.load(f)
                if isinstance(loaded_components_data, list):
                    components_data_store = loaded_components_data
                    log.info(f"Loaded {len(components_data_store)} component entries from {components_data_file_path}")
                else:
                    log.error(f"Components data file '{components_data_file_path}' does not contain a JSON list. Using empty list.")
                    components_data_store = []
        except Exception as e:
            log.error(f"Failed to load/parse components data from '{components_data_file_path}': {e}", exc_info=True)
            components_data_store = []
    else:
        log.warning(f"Components data file '{components_data_file_path}' not found. Component-specific features might be limited.")
        components_data_store = []

    is_core_initialized = True
    log.info("--- Chatbot Core Initialization Complete ---")
    return True


async def process_user_turn(session: ChatSession, user_input_raw: str) -> str:
    if not is_core_initialized:
        log.error("CORE_PROCESS: Chatbot core is not initialized! Cannot process turn.")
        lang_name = session.current_language_name or get_language_name("en") 
        dialect = session.last_detected_dialect_info
        try:
            return await call_groq_llm_final_answer_lc(
                user_context_for_current_turn="The system is not ready yet. Please apologize and ask the user to try again in a few moments.",
                target_language_name=lang_name,
                dialect_context_hint=dialect,
                memory_messages=[], 
                system_prompt_template_str="Apologize for system error. Respond in {{target_language_name}}."
            ) or "The system is currently unavailable. Please try again shortly." 
        except Exception as e_llm:
            log.error(f"CORE_PROCESS: LLM call failed during uninitialized state error: {e_llm}")
            return "A system issue occurred. Please try again soon."
    
    # 1. Language Detection and Explicit Switch
    detected_lang_code, detected_dialect_req_type, _dziribert_info = await detect_language_and_intent(user_input_raw)
    is_explicit_lang_request = detected_dialect_req_type and "_request" in detected_dialect_req_type

    if is_explicit_lang_request:
        if detected_lang_code != session.current_language or \
           (detected_dialect_req_type and detected_dialect_req_type != session.last_detected_dialect_info): 
            log.info(f"CORE_PROCESS: Explicit language/dialect request detected. "
                     f"Switching from '{session.current_language_name}/{session.last_detected_dialect_info or 'N/A'}' "
                     f"to '{get_language_name(detected_lang_code)}/{detected_dialect_req_type}'.")
            session.set_language(detected_lang_code, detected_dialect_req_type)
            
            ack_ctx = (f"The user has explicitly requested to switch the conversation to {session.current_language_name} "
                       f"(specific request context: {session.last_detected_dialect_info or 'N/A'}). "
                       f"Acknowledge this language switch clearly. Then, ask them what they need or what question they have "
                       f"in the new language ({session.current_language_name}).")
            
            # This response is generated directly in the target language.
            return await call_groq_llm_final_answer_lc(
                user_context_for_current_turn=ack_ctx,
                target_language_name=session.current_language_name,
                dialect_context_hint=session.last_detected_dialect_info,
                memory_messages=session.get_lc_memory_messages(),
                system_prompt_template_str="You are a helpful AI. Respond in {{target_language_name}}. If a dialect is specified ({{dialect_context_hint}}), use it if appropriate."
            ) or f"Language switched to {session.current_language_name}. How can I help you?" # Fallback
    
    elif detected_lang_code != session.current_language and not session.get_expectation(): 
        # Implicit language switch only if bot is not expecting a specific reply (e.g., "yes/no" to a prior question).
        log.info(f"CORE_PROCESS: Implicit language switch detected. "
                 f"From '{session.current_language_name}' to '{get_language_name(detected_lang_code)}'. "
                 f"Dialect/Req Type: '{detected_dialect_req_type or 'N/A'}'.")
        session.set_language(detected_lang_code, detected_dialect_req_type)

    # 2. Handle simple commands first (illustrative)
    if user_input_raw.lower().startswith("/loadpdf "): 
        # This response is directly localized.
        return await call_groq_llm_final_answer_lc(
            user_context_for_current_turn="To upload a PDF, please use the attachment button (usually a paperclip icon) in the chat interface.",
            target_language_name=session.current_language_name,
            dialect_context_hint=session.last_detected_dialect_info,
            memory_messages=session.get_lc_memory_messages()
        ) or "Use the attachment button to upload a PDF."

    if user_input_raw.lower() == "/clearpdf":
        cleared_file = session.pdf_context_source_filename
        session.clear_pdf_context()
        ack_msg_english = f"The PDF context from '{cleared_file}' has been cleared from our current discussion." if cleared_file else "No PDF context was loaded, so there's nothing to clear."
        # This response is localized.
        return await call_groq_llm_final_answer_lc(
            user_context_for_current_turn=f"Acknowledge the following system action: {ack_msg_english}",
            target_language_name=session.current_language_name,
            dialect_context_hint=session.last_detected_dialect_info,
            memory_messages=session.get_lc_memory_messages()
        ) or ack_msg_english

    # 3. Check for session reset/closing remarks
    reset_kws = get_localized_keywords("session_reset_keywords", session.current_language)
    closing_kws = get_localized_keywords("simple_closing_remarks", session.current_language)
    is_reset_command_by_keyword = any(k.lower() in user_input_raw.lower() for k in reset_kws if k)
    is_simple_closing_by_keyword = any(k.lower() == user_input_raw.lower() for k in closing_kws if k)

    # Only reset if not expecting a specific input (e.g., "yes/no", model name)
    if not session.get_expectation() and (is_reset_command_by_keyword or is_simple_closing_by_keyword):
        log.info(f"CORE_PROCESS: Session reset/closing (keyword-based) due to: '{user_input_raw[:50]}...'")
        reset_ctx_parts_english = [f"User (lang: {session.current_language_name}, dialect: {session.last_detected_dialect_info or 'N/A'}) input: '{user_input_raw}'."]
        system_prompt_for_closing = "You are a polite AI assistant. Respond in {{target_language_name}}."

        # Check for explicit "new session" keywords
        if any(kw.lower() in user_input_raw.lower() for kw in ["new session", "nouvelle session", "جلسة جديدة", "/new", "/reset", "start over"]):
            reset_ctx_parts_english.append("The user explicitly wants to start a new session. Confirm this and say you are ready for their new query.")
            system_prompt_for_closing = "The user wants to start a new session. Confirm this and state you are ready. Respond in {{target_language_name}}."
            session.end_session(reason=f"user_explicit_new_session_keyword: {user_input_raw.lower()}")
        elif is_simple_closing_by_keyword:
            reset_ctx_parts_english.append("The user made a simple closing remark (e.g., 'thank you', 'bye'). Respond politely and briefly. You can ask if there's anything else you can help with today, or simply wish them well if it's a definitive goodbye.")
            # Optionally end session for definitive closing remarks like "bye"
            # if user_input_raw.lower() in ["bye", "goodbye", "au revoir", "مع السلامة"]:
            #    session.end_session(reason=f"user_simple_definitive_closing: {user_input_raw.lower()}")
        else: # General reset keywords
            reset_ctx_parts_english.append("The user used a keyword suggesting they want to end the current discussion or reset the context. Acknowledge this politely. Ask if they need further assistance or if they'd like to start over with a new topic.")
            session.end_session(reason=f"user_general_keyword_reset: {user_input_raw.lower()}")

        localized_closing_response = await call_groq_llm_final_answer_lc(
            user_context_for_current_turn=" ".join(reset_ctx_parts_english),
            target_language_name=session.current_language_name,
            dialect_context_hint=session.last_detected_dialect_info,
            memory_messages=session.get_lc_memory_messages(), 
            system_prompt_template_str=system_prompt_for_closing
        )
        return localized_closing_response or "Okay. Let me know if you need anything else!" # Fallback

    # 4. Main Handler Routing
    intermediate_response_content: str | None = None 
    current_expectation = session.get_expectation()
    
    if current_expectation or session.in_troubleshooting_flow or session.active_tv_model:
        log.debug(f"CORE_PROCESS: Routing to handle_ongoing_session_turn. Expectation: {current_expectation}, "
                  f"InTroubleshootingFlow: {session.in_troubleshooting_flow}, ActiveModel: {session.active_tv_model}")
        intermediate_response_content = await handle_ongoing_session_turn(
            session=session, user_input_raw=user_input_raw,
            data_store=data_store, index_store=index_store,
            text_to_original_data_idx_map_store=text_to_original_data_idx_map_store,
            components_data_store=components_data_store, image_base_path=IMAGE_BASE_PATH_USER_MSG
        )
    else: 
        log.debug(f"CORE_PROCESS: Routing to handle_initial_query as no specific session state detected.")
        intermediate_response_content = await handle_initial_query(
            session=session, user_input_raw=user_input_raw,
            data_store=data_store, index_store=index_store,
            text_to_original_data_idx_map_store=text_to_original_data_idx_map_store,
            components_data_store=components_data_store, image_base_path=IMAGE_BASE_PATH_USER_MSG
        )

    # 5. Localization of Handler Response (if it's English core logic)
    final_assistant_response = None
    if intermediate_response_content:
        # Heuristic: Check if it's already Markdown and potentially localized by a sub-handler
        # (e.g., image_handler might directly produce localized Markdown for complex image lists)
        is_already_markdown_final = isinstance(intermediate_response_content, str) and \
                                     ("![alt text]" in intermediate_response_content or \
                                      "## " in intermediate_response_content or \
                                      "```" in intermediate_response_content) 
        is_error_msg_from_handler = isinstance(intermediate_response_content, str) and \
                                    intermediate_response_content.startswith("Error:")

        if session.current_language != "en" and \
           isinstance(intermediate_response_content, str) and \
           not is_already_markdown_final and \
           not is_error_msg_from_handler:
            
            # Attempt Darija microservice translation if applicable
            if session.current_language == "ar" and \
               session.last_detected_dialect_info and "darija" in session.last_detected_dialect_info.lower():
                log.info(f"CORE_PROCESS: Attempting Eng-to-Darija microservice for: '{intermediate_response_content[:60]}...'")
                darija_translation = await translate_english_to_dari_ja_via_service(intermediate_response_content)
                if darija_translation:
                    final_assistant_response = darija_translation
                    log.info("CORE_PROCESS: Eng-to-Darija microservice translation successful.")
                else:
                    log.warning("CORE_PROCESS: Eng-to-Darija microservice failed. Falling back to Groq LLM for Darija localization.")
                    # Fall through to generic LLM localization

            # Generic LLM localization if not Darija or Darija service failed
            if not final_assistant_response: 
                log.info(f"CORE_PROCESS: Localizing English core response to {session.current_language_name} "
                         f"for content: '{intermediate_response_content[:60]}...'")
                localized_response = await translate_text_lc(
                    text_to_translate=intermediate_response_content,
                    source_language_name="English",
                    target_language_name=session.current_language_name,
                    dialect_context_hint=session.last_detected_dialect_info,
                    context_hint_for_translation="This is a chatbot response related to TV troubleshooting or general assistance."
                )
                if localized_response and not localized_response.startswith("Error:"):
                    final_assistant_response = localized_response
                else:
                    log.error(f"CORE_PROCESS: Localization to {session.current_language_name} failed via LLM. "
                              f"LLM response: {localized_response}. Using English core response as fallback.")
                    final_assistant_response = intermediate_response_content # Fallback to English
        else: # Already English, or an error message, or pre-formatted Markdown
             final_assistant_response = intermediate_response_content
    
    # 6. Final Fallback if no response generated by any handler or localization
    if not final_assistant_response:
         log.warning(f"CORE_PROCESS: No specific response generated by handlers or localization failed for input '{user_input_raw[:50]}...'. Using final LLM fallback.")
         final_lang_name = session.current_language_name
         final_dialect = session.last_detected_dialect_info
         
         fallback_context_english = "I'm not sure how to respond to that. Could you please rephrase your query, or let me know if you need help with a TV problem or have a general question?"
         
         final_assistant_response = await call_groq_llm_final_answer_lc(
             user_context_for_current_turn=fallback_context_english, 
             target_language_name=final_lang_name,
             dialect_context_hint=final_dialect,
             memory_messages=session.get_lc_memory_messages(), 
             system_prompt_template_str="You are a helpful AI assistant. The previous attempt to generate a response failed. Respond to the user in {{target_language_name}} based on the English fallback context provided. {{dialect_context_hint}}"
         )
         
         if not final_assistant_response or final_assistant_response.startswith("Error:"):
            log.error(f"CORE_PROCESS: LLM call failed for FINAL FALLBACK message. LLM Error: {final_assistant_response}")
            default_error_responses = {
                "English": "I'm currently experiencing technical difficulties and cannot process your request. Please try again later.",
                "French": "Je rencontre actuellement des difficultés techniques et ne peux pas traiter votre demande. Veuillez réessayer plus tard.",
                "Arabic": "أواجه حاليًا صعوبات فنية ولا يمكنني معالجة طلبك. يرجى المحاولة مرة أخرى في وقت لاحق."
                # Add other languages if needed
            }
            final_assistant_response = default_error_responses.get(final_lang_name, default_error_responses["English"])

    return str(final_assistant_response) # Ensure it's a string