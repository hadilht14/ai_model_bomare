# backend/session_flow_handler.py
import logging
import re
import sys

try:
    from groq_api import (
        classify_follow_up_intent_lc, 
        # FollowUpIntentOutput, # Not directly used as a type hint here, tuple is used
        call_groq_llm_final_answer_lc 
    )
    from session_manager import ChatSession
    # from language_handler import get_localized_keywords # Not used in this version of the handler
    from troubleshooting_handler import (
        handle_specific_tv_troubleshooting,
        handle_standard_tv_troubleshooting,
        # handle_list_all_model_issues, # Can be called if handle_session_follow_up_llm_call identifies this intent
        handle_session_follow_up_llm_call 
    )
    from image_handler import handle_image_component_query
    from utils import extract_tv_model_from_query 
    from knowledge_handler import handle_general_knowledge_query 
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR in session_flow_handler.py: {e}.", file=sys.stderr)
    raise

log = logging.getLogger(__name__)

async def _handle_bot_expectation_response(
    session: ChatSession, user_input_raw: str,
    data_store, index_store, text_to_original_data_idx_map_store, 
    components_data_store, image_base_path: str
) -> str | None: # Returns ENGLISH core response or localized Markdown
    expectation = session.get_expectation()
    if not expectation: 
        log.warning("SF_HANDLER: _handle_bot_expectation_response called without an expectation set.")
        session.clear_expectations() 
        return None 

    log.info(f"SF_HANDLER: Handling expectation. Type: '{expectation.get('type')}'. Details: {expectation.get('details')}. User: '{user_input_raw[:50]}...'")

    bot_q_context = "I previously asked for more information or confirmation." 
    expectation_type = expectation.get("type")
    expectation_details = expectation.get("details", {})

    # Construct more specific bot_q_context based on expectation type
    if expectation_type == "model_for_problem" and session.expecting_model_for_problem:
        bot_q_context = f"I previously asked for the TV model related to the problem: '{session.expecting_model_for_problem[:50]}...'."
    elif expectation_type == "model_for_media_request" and expectation_details.get("media_query"):
        bot_q_context = f"I previously asked for the TV model to find media for: '{expectation_details['media_query'][:50]}...'."
    elif expectation_type == "elaboration_confirmation" and expectation_details.get("last_topic"):
        bot_q_context = f"I previously asked if you wanted more details on: '{expectation_details['last_topic'][:50]}...'."
    elif expectation_type == "new_problem_confirmation" and expectation_details.get("description"):
        problem_desc_for_ctx = expectation_details['description']
        model_for_ctx = expectation_details.get('model', 'the current TV')
        bot_q_context = f"I asked if you want to focus on a new problem: '{problem_desc_for_ctx[:50]}...' for model '{model_for_ctx}'."
    elif expectation_type == "model_switch_confirmation" and expectation_details.get("target_model"):
        bot_q_context = f"I asked if you want to switch our focus to TV model '{expectation_details['target_model']}' based on your mention of it."

    intent_cat, extracted_model_from_intent = await classify_follow_up_intent_lc(
        user_query=user_input_raw,
        bot_s_previous_question_context=bot_q_context,
        target_language_name=session.current_language_name,
        dialect_context_hint=session.last_detected_dialect_info,
        memory_messages=session.get_lc_memory_messages()
    )
    
    final_model_to_use = extracted_model_from_intent
    if not final_model_to_use: 
        final_model_to_use = extract_tv_model_from_query(user_input_raw)
        if final_model_to_use:
             log.info(f"SF_HANDLER: Regex extracted model '{final_model_to_use}' as fallback for expectation.")

    english_core_response = None # This handler primarily returns English for chatbot_core to localize

    if final_model_to_use: 
        session.add_recognized_model(final_model_to_use) 
        log.info(f"SF_HANDLER: Model '{final_model_to_use}' provided/confirmed by user. Current active: {session.active_tv_model}")
        
        original_problem_context = session.expecting_model_for_problem
        original_media_query = expectation_details.get("media_query")
        
        session.clear_expectations() 

        if expectation_type == "model_for_problem" and original_problem_context:
            session.start_troubleshooting_flow(original_problem_context, final_model_to_use) # This sets active model
            english_core_response = await handle_specific_tv_troubleshooting(
                user_problem_original_lang=original_problem_context, session=session,
                data_store=data_store, index_store=index_store,
                text_to_original_data_idx_map_store=text_to_original_data_idx_map_store,
                components_data_store=components_data_store
            )
        elif expectation_type == "model_for_media_request" and original_media_query:
            session.set_active_model(final_model_to_use, "media_request_model_provided")
            # handle_image_component_query can return localized Markdown or English
            english_core_response = await handle_image_component_query(
                user_query=original_media_query, session=session, data_store=data_store,
                components_data_store=components_data_store, image_base_path=image_base_path
            )
        else: 
            if session.active_tv_model != final_model_to_use: 
                session.set_active_model(final_model_to_use, "direct_provision_during_expectation")
            english_core_response = f"Okay, I've noted the TV model as '{session.active_tv_model}'. How can I help you with it?"

    elif intent_cat == "affirmative":
        log.info(f"SF_HANDLER: User responded affirmatively to expectation '{expectation_type}'.")
        last_topic_for_elaboration = expectation_details.get("last_topic") # Store before clearing
        new_problem_details_for_confirmation = expectation_details if expectation_type == "new_problem_confirmation" else None
        model_switch_target = expectation_details.get("target_model") if expectation_type == "model_switch_confirmation" else None

        session.clear_expectations() 

        if expectation_type == "elaboration_confirmation" and last_topic_for_elaboration:
            english_core_response = await handle_session_follow_up_llm_call(
                f"Yes, please tell me more about '{last_topic_for_elaboration}'.", session 
            )
        elif new_problem_details_for_confirmation and new_problem_details_for_confirmation.get("description"):
            new_problem_desc = new_problem_details_for_confirmation['description']
            new_problem_model = new_problem_details_for_confirmation.get('model', session.active_tv_model)
            session.start_troubleshooting_flow(new_problem_desc, new_problem_model)
            english_core_response = await handle_specific_tv_troubleshooting(
                user_problem_original_lang=session.current_problem_description, session=session,
                data_store=data_store, index_store=index_store,
                text_to_original_data_idx_map_store=text_to_original_data_idx_map_store,
                components_data_store=components_data_store
            )
        elif model_switch_target:
            session.set_active_model(model_switch_target, "user_confirmed_switch")
            english_core_response = f"Okay, we are now focusing on TV model '{model_switch_target}'. How can I assist you with this model?"
            # If there was an original query associated with this switch, it could be re-processed.
            # original_query_for_target = expectation_details.get("original_query_for_target_model")
            # if original_query_for_target:
            #    # This would re-enter the main logic, perhaps via a special flag or direct call
            #    # For simplicity, just acknowledge the switch for now.
            #    pass
        else: 
            english_core_response = "Okay, understood. What would you like to do next?"

    elif intent_cat == "negative":
        log.info(f"SF_HANDLER: User responded negatively to expectation '{expectation_type}'.")
        original_problem_for_model_expectation = session.expecting_model_for_problem
        session.clear_expectations() 

        if expectation_type == "model_for_problem" and original_problem_for_model_expectation:
            # User said 'no' to providing model for a problem
            standard_ts_response = await handle_standard_tv_troubleshooting(
                original_problem_for_model_expectation, session, ask_for_model_explicitly=False 
            )
            english_core_response = (f"Okay, we'll proceed without a specific model for now for the issue: "
                                     f"'{original_problem_for_model_expectation[:50]}...'. "
                                     f"{standard_ts_response or ''}")
        elif expectation_type == "elaboration_confirmation":
            english_core_response = "Alright. How else can I assist you?"
        elif expectation_type == "new_problem_confirmation":
             english_core_response = (f"Okay, we'll stick to our previous topic then: "
                                      f"'{session.current_problem_description or 'the last discussion'}' "
                                      f"for model '{session.active_tv_model or 'your TV'}'.")
        elif expectation_type == "model_switch_confirmation":
            english_core_response = f"Okay, we'll continue focusing on model '{session.active_tv_model or 'our current TV model'}'. How can I help?"
        else:
            english_core_response = "Okay. What would you like to do instead?"
    
    else: # unclear_or_other for the follow-up intent
        log.info(f"SF_HANDLER: Follow-up intent unclear ('{intent_cat}') for expectation '{expectation_type}'. Re-prompting.")
        if expectation_type == "model_for_problem":
            english_core_response = f"I wasn't sure if that was the TV model for the issue: '{session.expecting_model_for_problem[:50]}...'. Could you please provide the model number, or say 'no' if you don't have it?"
        elif expectation_type == "model_for_media_request":
            media_q_ctx = expectation_details.get('media_query','your request')
            english_core_response = f"Sorry, I need the TV model to find the media for '{media_q_ctx[:50]}...'. What is the model?"
        elif expectation_type == "elaboration_confirmation":
            last_topic_ctx = expectation_details.get('last_topic','our last point')
            english_core_response = f"I didn't catch if you wanted more details on '{last_topic_ctx[:50]}...'. Please say 'yes' or 'no'."
        elif expectation_type == "new_problem_confirmation":
            problem_desc_ctx = expectation_details.get('description','the suggested topic')
            english_core_response = f"Sorry, I didn't understand if you want to switch to the new problem: '{problem_desc_ctx[:50]}...'. Please confirm with 'yes' or 'no'."
        elif expectation_type == "model_switch_confirmation":
            target_model_ctx = expectation_details.get('target_model','the other model')
            english_core_response = f"I wasn't sure if you wanted to switch to model '{target_model_ctx}'? Please say 'yes' or 'no'."
        else: 
            english_core_response = "I'm not sure how to proceed with that. Could you please clarify?"
            session.clear_expectations() 

    return english_core_response


async def handle_ongoing_session_turn(
    session: ChatSession,
    user_input_raw: str,
    data_store, index_store, text_to_original_data_idx_map_store,
    components_data_store, image_base_path: str
) -> str | None: 

    # 1. Handle direct responses to bot's questions (if any expectation is set)
    if session.get_expectation():
        return await _handle_bot_expectation_response(
            session, user_input_raw, data_store, index_store,
            text_to_original_data_idx_map_store, components_data_store, image_base_path
        )

    # 2. If no specific expectation, process as a general follow-up in an active session
    assistant_response_content: str | None = None

    # Check if user mentions a *different recognized* model, suggesting a context switch
    mentioned_model_in_query = extract_tv_model_from_query(user_input_raw)
    if mentioned_model_in_query and \
       session.active_tv_model and \
       mentioned_model_in_query != session.active_tv_model and \
       mentioned_model_in_query in session.recognized_tv_models:
        log.info(f"SF_HANDLER: User mentioned recognized model '{mentioned_model_in_query}', "
                 f"different from active '{session.active_tv_model}'. Asking to confirm switch.")
        session.set_expectation(
            "model_switch_confirmation", 
            details={
                "target_model": mentioned_model_in_query, 
                "current_active_model": session.active_tv_model,
                "original_query_for_target_model": user_input_raw 
            }
        )
        # This is an English response, chatbot_core will localize.
        return (f"I see you mentioned TV model '{mentioned_model_in_query}'. "
                f"We are currently focused on model '{session.active_tv_model}'. "
                f"Would you like to switch our focus to '{mentioned_model_in_query}' to address "
                f"your query about '{user_input_raw[:50].strip()}'? (yes/no)")
    elif mentioned_model_in_query and not session.active_tv_model and mentioned_model_in_query in session.recognized_tv_models:
        log.info(f"SF_HANDLER: No active model, user mentioned recognized model '{mentioned_model_in_query}'. Setting as active.")
        session.set_active_model(mentioned_model_in_query, "user_re_mention_no_active")
        # Now proceed as if it's an ongoing turn for this newly active model.

    # 3. Standard ongoing turn processing
    if session.in_troubleshooting_flow or session.active_tv_model:
        log.info(f"SF_HANDLER: General follow-up. ActiveModel: '{session.active_tv_model}', Problem: '{session.current_problem_description or 'N/A'}'")
        
        english_core_response = await handle_session_follow_up_llm_call(user_input_raw, session)

        if english_core_response and "NEW_PROBLEM_SUGGESTION:" in english_core_response:
            parts = english_core_response.split("NEW_PROBLEM_SUGGESTION:", 1)
            user_facing_reply_part = parts[0].strip()
            suggestion_part = parts[1].strip() 
            
            new_problem_desc_from_llm = suggestion_part
            # Default to current active model if LLM doesn't specify one with the suggestion
            new_problem_model_from_llm = session.active_tv_model 
            
            model_match = re.search(r"Model:\s*([A-Z0-9_.\-]+)", suggestion_part, re.IGNORECASE)
            if model_match:
                extracted_model_name = model_match.group(1).strip().upper()
                if extracted_model_name: 
                    new_problem_model_from_llm = extracted_model_name
                new_problem_desc_from_llm = re.sub(r"Model:\s*[A-Z0-9_.\-]+", "", new_problem_desc_from_llm, flags=re.IGNORECASE).strip()
            
            if not new_problem_model_from_llm and session.active_tv_model: # If LLM suggestion didn't include model, use current active
                 new_problem_model_from_llm = session.active_tv_model
            elif not new_problem_model_from_llm and not session.active_tv_model:
                 log.warning("SF_HANDLER: LLM suggested new problem but no model context available from suggestion or session. Cannot confirm problem switch effectively.")
                 assistant_response_content = user_facing_reply_part + " It sounds like you might be describing a new issue. Could you tell me more about it, and if it relates to a specific TV model?"
                 return assistant_response_content # Return English for localization

            # Ensure the suggested model is added to recognized if new.
            if new_problem_model_from_llm:
                session.add_recognized_model(new_problem_model_from_llm)
            
            log.info(f"SF_HANDLER: LLM suggested new problem: '{new_problem_desc_from_llm}' for model '{new_problem_model_from_llm}'. Asking confirmation.")
            session.set_expectation(
                "new_problem_confirmation", 
                details={"description": new_problem_desc_from_llm, "model": new_problem_model_from_llm}
            )
            
            confirmation_q_en = (f" It sounds like you might be describing a new issue: '{new_problem_desc_from_llm}' "
                                 f"(for model '{new_problem_model_from_llm}'). Would you like to focus on that now? (yes/no)")
            assistant_response_content = user_facing_reply_part + confirmation_q_en
        
        elif english_core_response and session.active_tv_model and \
             any(phrase.lower() in english_core_response.lower() for phrase in [
                "looking for the", "let me see if i can provide that", "i can check on that", "you'd like to see the", "show you the"
             ]) and \
             any(media_kw.lower() in user_input_raw.lower() for media_kw in ["image", "diagram", "component", "picture", "photo", "schema", "list"]):
            log.info(f"SF_HANDLER: LLM acknowledged a media request for model {session.active_tv_model}. Routing to image_handler for fulfillment.")
            media_response = await handle_image_component_query(
                user_query=user_input_raw, 
                session=session, data_store=data_store,
                components_data_store=components_data_store, image_base_path=image_base_path
            )
            assistant_response_content = media_response 
        else:
            assistant_response_content = english_core_response

    else: 
        log.warning(f"SF_HANDLER: Ongoing turn but no clear session context (no expectation, no active_model, not in_troubleshooting_flow). User: '{user_input_raw[:50]}...' Treating as general knowledge.")
        assistant_response_content = await handle_general_knowledge_query(user_input_raw, session)
        if not assistant_response_content or "I'm not sure how to respond" in assistant_response_content:
            assistant_response_content = "I seem to have lost our previous context. Could you please rephrase or state your current need?"
            # Potentially call session.end_session("context_lost_in_ongoing_turn") or session.clear_active_problem()

    return assistant_response_content