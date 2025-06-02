# backend/initial_interaction_handler.py
import logging
# import re # No longer needed here if extract_tv_model_from_query is moved
import sys

try:
    from groq_api import (
        classify_main_intent_and_extract_model_lc, 
        MainIntentOutput 
    )
    from session_manager import ChatSession
    # from language_handler import get_localized_keywords # If still needed for specific keyword checks
    from troubleshooting_handler import (
        handle_specific_tv_troubleshooting, 
        handle_standard_tv_troubleshooting  
    )
    # from image_handler import handle_image_component_query # REMOVE THIS IMPORT
    from knowledge_handler import handle_general_knowledge_query
    from utils import extract_tv_model_from_query # <--- ADD THIS IMPORT
except ImportError as e:
    # Ensure sys is imported if you use sys.stderr here
    print(f"CRITICAL IMPORT ERROR in initial_interaction_handler.py: {e}. Application will likely fail.", file=sys.stderr)
    raise

log = logging.getLogger(__name__)

# extract_tv_model_from_query FUNCTION IS NOW MOVED TO utils.py

async def handle_initial_query(
    session: ChatSession,
    user_input_raw: str, 
    data_store, index_store, text_to_original_data_idx_map_store,
    components_data_store, image_base_path: str # image_base_path might not be needed here anymore
) -> str | None: # Returns ENGLISH core response or localized Markdown
    log.info(f"INITIAL_HANDLER: Processing initial query: '{user_input_raw[:50]}...' Lang: {session.current_language_name}")
    
    assistant_response_content: str | None = None 

    lc_memory_messages = session.get_lc_memory_messages()
    history_summary_for_intent = None
    if lc_memory_messages:
        summary_parts = []
        for msg in lc_memory_messages[-2:]: 
            role = "User" if msg.type == "human" else "Assistant"
            summary_parts.append(f"{role}: {msg.content[:50]}...")
        history_summary_for_intent = " ".join(summary_parts)

    intent_result: MainIntentOutput | None = await classify_main_intent_and_extract_model_lc(
        user_query=user_input_raw,
        target_language_name=session.current_language_name,
        dialect_context_hint=session.last_detected_dialect_info,
        chat_history_summary_for_intent=history_summary_for_intent
    )

    if not intent_result: 
        log.error("INITIAL_HANDLER: Main intent classification failed critically. Falling back.")
        extracted_model_regex = extract_tv_model_from_query(user_input_raw)
        if extracted_model_regex:
            session.add_recognized_model(extracted_model_regex) 
            assistant_response_content = f"I recognized TV model '{session.active_tv_model}'. How can I help you with this model?"
        else:
            assistant_response_content = "I'm having trouble understanding your request. Could you please rephrase?"
        return assistant_response_content

    llm_extracted_model = intent_result.extracted_model_if_any
    if llm_extracted_model:
        session.add_recognized_model(llm_extracted_model) 
        log.info(f"INITIAL_HANDLER: LLM extracted model '{llm_extracted_model}', now active (if was none): '{session.active_tv_model}'. Recognized: {session.recognized_tv_models}")

    if not session.active_tv_model and intent_result.intent in ['specific_tv_troubleshooting', 'media_request_model_specific']:
        regex_model = extract_tv_model_from_query(user_input_raw)
        if regex_model:
            session.add_recognized_model(regex_model)
            log.info(f"INITIAL_HANDLER: Regex extracted model '{regex_model}' as fallback, now active: '{session.active_tv_model}'. Recognized: {session.recognized_tv_models}")

    intent_category = intent_result.intent
    active_model_for_handler = session.active_tv_model

    # The image_base_path is used by handle_image_component_query, which is called from session_flow_handler
    # or chatbot_core based on routing. It might not be directly needed here if handle_initial_query
    # doesn't directly call handle_image_component_query anymore for "initial" media requests.
    # Let's assume for now that media requests are distinct enough to be routed.
    # If an initial query IS a media request, this logic will handle it:

    if intent_category == 'specific_tv_troubleshooting':
        if active_model_for_handler:
            log.info(f"INITIAL_HANDLER: Routing to Specific TV Troubleshooting for model '{active_model_for_handler}'.")
            session.start_troubleshooting_flow(user_input_raw, active_model_for_handler)
            assistant_response_content = await handle_specific_tv_troubleshooting(
                user_problem_original_lang=user_input_raw, 
                session=session, 
                data_store=data_store, index_store=index_store,
                text_to_original_data_idx_map_store=text_to_original_data_idx_map_store,
                components_data_store=components_data_store
            )
        else: 
            log.warning(f"INITIAL_HANDLER: Intent was 'specific_tv_troubleshooting' but no model identified. Routing to Standard Troubleshooting and asking for model.")
            session.start_troubleshooting_flow(user_input_raw) 
            assistant_response_content = await handle_standard_tv_troubleshooting(user_input_raw, session, ask_for_model_explicitly=True)
            session.set_expectation("model_for_problem", problem_context_for_model_request=user_input_raw)

    elif intent_category == 'standard_tv_troubleshooting':
        log.info("INITIAL_HANDLER: Routing to Standard TV Troubleshooting.")
        session.start_troubleshooting_flow(user_input_raw) 
        assistant_response_content = await handle_standard_tv_troubleshooting(user_input_raw, session, ask_for_model_explicitly=True)
        session.set_expectation("model_for_problem", problem_context_for_model_request=user_input_raw)

    elif intent_category == 'media_request_model_specific':
        if active_model_for_handler:
            log.info(f"INITIAL_HANDLER: Routing to Image/Component Query for model '{active_model_for_handler}'.")
            # This import should be here if handle_initial_query can directly call it
            from image_handler import handle_image_component_query 
            assistant_response_content = await handle_image_component_query(
                user_query=user_input_raw, session=session, data_store=data_store,
                components_data_store=components_data_store, image_base_path=image_base_path
            )
            if assistant_response_content: 
                session.start_troubleshooting_flow(f"Media request: {user_input_raw[:30]}", active_model_for_handler)
        else: 
            log.warning("INITIAL_HANDLER: Intent was 'media_request_model_specific' but no model found. Asking for model first.")
            assistant_response_content = (f"I can help with images or component lists. Which TV model are you asking about for '{user_input_raw[:30]}...'?")
            session.set_expectation("model_for_media_request", details={"media_query": user_input_raw})

    elif intent_category == 'media_request_generic':
        log.info("INITIAL_HANDLER: Generic media request. Asking for model.")
        assistant_response_content = (f"I can help with images or component lists. Which TV model are you interested in for your request: '{user_input_raw[:30]}...'?")
        session.set_expectation("model_for_media_request", details={"media_query": user_input_raw})

    elif intent_category == 'general_question':
        log.info("INITIAL_HANDLER: Routing to General Knowledge Query.")
        assistant_response_content = await handle_general_knowledge_query(user_input_raw, session)

    elif intent_category == 'follow_up_clarification':
        log.warning("INITIAL_HANDLER: Intent 'follow_up_clarification' received without prior expectation. Treating as general query or asking to rephrase.")
        if len(user_input_raw.strip().split()) <= 2: 
             assistant_response_content = "I'm not sure what that refers to. Could you please provide more context or ask a full question?"
        else: 
            assistant_response_content = await handle_general_knowledge_query(user_input_raw, session) 

    else: # 'other_unclear' or unexpected
        log.warning(f"INITIAL_HANDLER: Intent classified as '{intent_category}' (unclear/other). Trying general knowledge handler.")
        assistant_response_content = await handle_general_knowledge_query(user_input_raw, session)
        if not assistant_response_content or "I'm not sure how to respond" in assistant_response_content or "I cannot answer" in assistant_response_content: 
             assistant_response_content = "I'm not quite sure how to help with that. Could you try rephrasing your request, or tell me if you're looking for TV troubleshooting help or general information?"

    if session.active_tv_model and not session.in_troubleshooting_flow and \
       intent_category in ['specific_tv_troubleshooting', 'media_request_model_specific']:
        session.in_troubleshooting_flow = True 

    return assistant_response_content