# backend/troubleshooting_handler.py
import logging
import json
import sys # Only if you were directly loading data here, but it's passed now

try:
    from groq_api import (
        call_groq_llm_final_answer_lc as call_groq_llm_final_answer,
        generate_hypothetical_document_lc as generate_hypothetical_document,
        translate_text_lc as translate_input_for_rag,
    )
    from vector_search import search_relevant_guides 
    from session_manager import ChatSession 
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR in troubleshooting_handler.py: {e}. Application will likely fail.", file=sys.stderr)
    raise

log = logging.getLogger(__name__)

async def handle_specific_tv_troubleshooting(
    user_problem_original_lang: str, 
    session: ChatSession, 
    data_store, index_store, text_to_original_data_idx_map_store,
    components_data_store, 
) -> str | None: # Returns ENGLISH core response string
    active_model = session.active_tv_model # Get active model from session
    if not active_model:
        log.error("TS_HANDLER_SPECIFIC: Called without an active TV model in session. This should be set before calling.")
        # This case should ideally be caught before calling this handler.
        # If it happens, ask for model again or route to standard troubleshooting.
        return ("I need to know which TV model you're referring to for specific troubleshooting. "
                "Could you please provide the model name?")

    log.info(f"TS_HANDLER_SPECIFIC: Preparing RAG content for model='{active_model}', Problem='{user_problem_original_lang[:60]}...', UserLang='{session.current_language_name}'")

    if not index_store or not data_store:
        log.error("TS_HANDLER_SPECIFIC: RAG index or data_store not available for troubleshooting.")
        return "System error: The troubleshooting knowledge base is currently unavailable. Please try again later."

    problem_for_rag_en = user_problem_original_lang
    if session.current_language != "en":
        translated_problem = await translate_input_for_rag(
            text_to_translate=user_problem_original_lang,
            source_language_name=session.current_language_name,
            target_language_name="English",
            dialect_context_hint=session.last_detected_dialect_info,
            context_hint_for_translation="TV problem description for troubleshooting lookup"
        )
        if translated_problem and not translated_problem.startswith("Error:"):
            problem_for_rag_en = translated_problem
            log.info(f"TS_HANDLER_SPECIFIC: Translated problem for RAG/HyDE: '{problem_for_rag_en[:60]}...'")
        else:
            log.warning(f"TS_HANDLER_SPECIFIC: English translation of problem failed. LLM response: {translated_problem}. Using original language input for RAG: '{user_problem_original_lang[:60]}'")
            
    hypothetical_query_en = await generate_hypothetical_document(problem_for_rag_en)
    search_query_text_en = hypothetical_query_en if hypothetical_query_en and not hypothetical_query_en.startswith("Error:") else problem_for_rag_en
    log.info(f"TS_HANDLER_SPECIFIC: Using search query for RAG: '{search_query_text_en[:60]}...' for model '{active_model}'")

    rag_result_guide_dict = search_relevant_guides(
        query_text=search_query_text_en, target_model=active_model, data=data_store,
        index=index_store, text_to_original_data_idx_map=text_to_original_data_idx_map_store,
        k_results=1 
    )

    # Store general images in session if found via RAG or generic model entry
    if rag_result_guide_dict and rag_result_guide_dict.get("images"):
        session.current_model_general_images = rag_result_guide_dict.get("images")
        log.info(f"TS_HANDLER_SPECIFIC: General images found via RAG for model {active_model} and stored in session.")
    elif not session.current_model_general_images: 
        # Attempt to load general images if not already present and RAG failed or had no images
        # This assumes data_store contains entries that might just have model and images
        first_model_rag_entry = next((item for item in data_store if item.get("model","").upper() == active_model.upper() and item.get("images")), None)
        if first_model_rag_entry:
            session.current_model_general_images = first_model_rag_entry.get("images")
            log.info(f"TS_HANDLER_SPECIFIC: General images for model {active_model} loaded into session from a generic model entry in data_store.")


    if not rag_result_guide_dict:
        log.info(f"TS_HANDLER_SPECIFIC: No specific RAG guide found for model '{active_model}', query '{search_query_text_en[:60]}'.")
        offer_text_en = ""
        image_offers_en = [] 
        if session.current_model_general_images: 
            if session.current_model_general_images.get('motherboard'): image_offers_en.append("a motherboard image")
            if session.current_model_general_images.get('key_components'): image_offers_en.append("a general key components image")
            if session.current_model_general_images.get('block_diagram'): image_offers_en.append("a block diagram")
        
        model_comp_data_entry = next((item for item in (components_data_store or []) if item.get("tv_model","").upper() == active_model.upper()), None)
        if model_comp_data_entry:
            if model_comp_data_entry.get("image_filename") and "a detailed key components diagram" not in image_offers_en:
                 image_offers_en.append("a detailed key components diagram")
            if model_comp_data_entry.get("key_components") and "a list of key components with descriptions" not in image_offers_en:
                image_offers_en.append("a list of key components with descriptions")
        
        if image_offers_en:
            types_str_en = ", ".join(sorted(list(set(image_offers_en)))) # Unique and sorted
            offer_text_en = f" However, I can check if I have {types_str_en} for this model if you'd like."

        return (f"I searched for information on TV model '{active_model}' regarding an issue like '{problem_for_rag_en}', "
                f"but I couldn't find a specific troubleshooting guide for it in my current knowledge base. "
                f"Could you please try rephrasing the problem, or perhaps describe it in more detail?{offer_text_en}")

    guide_issue_en = rag_result_guide_dict.get("issue", "the relevant troubleshooting information") 
    steps_en_list = rag_result_guide_dict.get("steps") 

    if not steps_en_list or not isinstance(steps_en_list, list) or not any(s and isinstance(s, dict) and s.get("description") for s in steps_en_list):
        log.warning(f"TS_HANDLER_SPECIFIC: RAG Guide '{guide_issue_en}' for model {active_model} has no valid steps or steps are malformed.")
        offer_text_en = ""
        image_offers_en = [] # Re-populate as it might not have been done if rag_result_guide_dict was true initially
        if session.current_model_general_images:
            if session.current_model_general_images.get('motherboard'): image_offers_en.append("a motherboard image")
            if session.current_model_general_images.get('key_components'): image_offers_en.append("a general key components image")
            if session.current_model_general_images.get('block_diagram'): image_offers_en.append("a block diagram")
        model_comp_data_entry = next((item for item in (components_data_store or []) if item.get("tv_model","").upper() == active_model.upper()), None)
        if model_comp_data_entry:
            if model_comp_data_entry.get("image_filename") and "a detailed key components diagram" not in image_offers_en: image_offers_en.append("a detailed key components diagram")
            if model_comp_data_entry.get("key_components") and "a list of key components" not in image_offers_en : image_offers_en.append("a list of key components with descriptions")
        
        if image_offers_en:
            types_str_en = ", ".join(sorted(list(set(image_offers_en))))
            offer_text_en = f" However, I might be able to show you {types_str_en} for this model if you'd like."

        return (f"I found a guide titled '{guide_issue_en}' for your TV model '{active_model}', "
                f"but unfortunately, it doesn't contain detailed step-by-step instructions that I can share. "
                f"Would you like me to provide some general troubleshooting advice for this type of issue, or perhaps we can explore other options?{offer_text_en}")

    english_response_parts = [
        f"Okay, for your TV model '{active_model}' and the issue related to '{guide_issue_en}', here are some troubleshooting steps I found (these are in English and will be translated for you if needed):",
    ]
    valid_steps_count = 0
    for i, step_info in enumerate(steps_en_list):
        if step_info and isinstance(step_info, dict):
            step_desc_en = step_info.get('description', '').strip() 
            if step_desc_en:
                 english_response_parts.append(f"{valid_steps_count+1}. {step_desc_en}")
                 valid_steps_count +=1
    
    if valid_steps_count == 0: 
        log.error(f"TS_HANDLER_SPECIFIC: Logic error - steps_en_list present but no valid step descriptions extracted for '{guide_issue_en}', model {active_model}.")
        return (f"I found information about '{guide_issue_en}' for TV model '{active_model}', but I couldn't extract clear troubleshooting steps from it. "
                f"Would you like to try discussing this issue generally, or ask about available diagrams or component lists for this model?")

    english_response_parts.append(
        "\n**Important Safety Note:** Before attempting any internal checks or component replacements, "
        "please ensure your TV is completely unplugged from the power source. If you are unsure or uncomfortable "
        "with any step, it's always best to consult a qualified electronics technician."
    )

    image_offer_text_en_parts = []
    if session.current_model_general_images:
        if session.current_model_general_images.get('motherboard'): image_offer_text_en_parts.append("a motherboard image")
        if session.current_model_general_images.get('key_components'): image_offer_text_en_parts.append("a general key components image")
        if session.current_model_general_images.get('block_diagram'): image_offer_text_en_parts.append("a block diagram")

    model_comp_data_entry = next((item for item in (components_data_store or []) if item.get("tv_model","").upper() == active_model.upper()), None)
    if model_comp_data_entry:
        if model_comp_data_entry.get("image_filename") and "a detailed key components diagram" not in image_offer_text_en_parts:
             image_offer_text_en_parts.append("a detailed key components diagram")
        if model_comp_data_entry.get("key_components") and "a list of key components" not in " ".join(pt.lower() for pt in image_offer_text_en_parts):
            image_offer_text_en_parts.append("a list of key components with descriptions")

    if image_offer_text_en_parts:
        unique_offers_en = sorted(list(set(image_offer_text_en_parts)))
        if unique_offers_en:
            types_str_en = unique_offers_en[0]
            if len(unique_offers_en) == 2: types_str_en = f"{unique_offers_en[0]} and {unique_offers_en[1]}"
            elif len(unique_offers_en) > 2: types_str_en = ", ".join(unique_offers_en[:-1]) + f", or {unique_offers_en[-1]}"
            
            english_response_parts.append(
                f"\nAfter you've reviewed these steps, I might also be able to provide {types_str_en} for your TV model ({active_model}). "
                f"Just let me know if you'd like to see any of these, or if you have other questions about these steps."
            )

    english_core_response = "\n\n".join(english_response_parts)
    log.info(f"TS_HANDLER_SPECIFIC: Prepared English core response for RAG result: '{english_core_response[:150]}...'")
    session.start_troubleshooting_flow(problem_for_rag_en, active_model) # Ensure flow is marked active
    return english_core_response


async def handle_standard_tv_troubleshooting(
    user_problem_original_lang: str, 
    session: ChatSession,
    ask_for_model_explicitly: bool = True 
) -> str | None: # Returns ENGLISH core response string
    log.info(f"TS_HANDLER_STANDARD: Generating generic TV advice for problem: '{user_problem_original_lang[:60]}...'. Ask for model: {ask_for_model_explicitly}")

    llm_context_for_general_advice_en = f"""
The user has described a TV problem: "{user_problem_original_lang}"
The specific TV model is NOT yet known.

Your task is to generate a helpful response in **English**. This response should:
1.  Acknowledge the user's problem briefly.
2.  Provide a list of general, common troubleshooting steps applicable to most TVs for such an issue.
    Consider steps like:
    - Checking power connections (TV and outlet).
    - Verifying the correct input source is selected.
    - Inspecting all cables (HDMI, coaxial, audio, etc.) for damage or loose connections.
    - Restarting the TV and any connected devices (e.g., cable box, streaming player).
    - Checking the remote control (batteries, direct TV buttons not stuck).
    - Ensuring TV ventilation is not obstructed.
3.  Include an "Important Safety Note" like: "Important Safety Note: Before attempting any internal checks or component replacements, please ensure your TV is completely unplugged from the power source. If you are unsure or uncomfortable with any step, it's always best to consult a qualified electronics technician."
"""
    if ask_for_model_explicitly:
        llm_context_for_general_advice_en += (
            "\n4. **Crucially, at the very end of your response, ask the user for their TV model number to provide more specific help.** "
            "Phrase this request clearly, for example: 'To help me provide more specific advice, could you please tell me the model number of your TV? "
            "You can usually find it on a sticker on the back or side of the TV. If you don't know it or don't want to provide it at this time, just say 'no' or that you don't know.'"
        )
    else:
        llm_context_for_general_advice_en += "\n4. Conclude by asking if these general steps help or if they have more details to share about the problem."


    llm_context_for_general_advice_en += (
        "\nFormat your entire response in **English** using **Markdown** (e.g., headings for sections like 'General Troubleshooting Steps', bolding for emphasis, numbered lists for steps).\n"
        "Your response should be directly usable as chatbot output (it will be localized later if the user's language is not English).\n"
        "Do NOT add any conversational fluff like 'Sure, I can help with that.' Just provide the structured advice and the question (if any)."
    )
    
    system_prompt_template_en_advice = (
        "You are a helpful TV troubleshooting assistant. Your current task is to generate general "
        "troubleshooting advice in **English** based on a user's problem description, as the TV model is not yet known. "
        "Follow all instructions in the user's message, especially regarding content, Markdown formatting, and the "
        "**(if requested) mandatory request for the TV model number at the end of your response**. "
        "Output only the advice and the question, without any introductory or concluding chat phrases from your side."
    )
    
    # Minimal history, primarily for language context if needed by LLM for understanding the problem, but not for memory of past steps
    memory_for_this_call = session.get_lc_memory_messages()[:1] if session.get_lc_memory_messages() else []

    english_advice = await call_groq_llm_final_answer( 
        user_context_for_current_turn=llm_context_for_general_advice_en,
        target_language_name="English", 
        dialect_context_hint=None,      
        memory_messages=memory_for_this_call,             
        system_prompt_template_str=system_prompt_template_en_advice
    )

    if english_advice and not english_advice.startswith("Error:"):
        log.info(f"TS_HANDLER_STANDARD: LLM generated English general advice: '{english_advice[:150]}...'")
        if ask_for_model_explicitly and \
           ("model number" not in english_advice.lower() and \
            "model of your tv" not in english_advice.lower() and \
            "model name" not in english_advice.lower()):
            log.warning("TS_HANDLER_STANDARD: LLM response for general advice did not include model request. Appending default English request.")
            model_request_fallback_en = (
                "\n\nTo help me provide more specific advice, could you please tell me the model number of your TV? "
                "You can usually find it on a sticker on the back or side of the TV. If you don't know it or "
                "don't want to provide it at this time, just say 'no' or that you don't know."
            )
            english_advice += model_request_fallback_en
        session.start_troubleshooting_flow(user_problem_original_lang) # Mark flow as active, even without model
        if ask_for_model_explicitly:
            session.set_expectation("model_for_problem", problem_context_for_model_request=user_problem_original_lang)
        return english_advice 
    else:
        log.error(f"TS_HANDLER_STANDARD: Failed to generate general advice from LLM for problem '{user_problem_original_lang[:50]}...'. LLM response: {english_advice}")
        fallback_en = (
            f"I understand you're having an issue with your TV related to: \"{user_problem_original_lang}\". "
            "Some general things to check are power connections, cables, and the selected input source. Restarting the TV can often help. "
            "Remember, for safety, always unplug the TV before checking any internal components."
        )
        if ask_for_model_explicitly:
             fallback_en += (
                "\n\nTo help me provide more specific advice, could you please tell me the model number of your TV? "
                "You can usually find it on a sticker on the back or side of the TV. If you don't know it or "
                "don't want to provide it at this time, just say 'no' or that you don't know."
            )
        session.start_troubleshooting_flow(user_problem_original_lang)
        if ask_for_model_explicitly:
            session.set_expectation("model_for_problem", problem_context_for_model_request=user_problem_original_lang)
        return fallback_en


async def handle_list_all_model_issues(session: ChatSession, data_store) -> str | None:
    # This function should return an ENGLISH core response string.
    active_model = session.active_tv_model
    if not active_model:
        log.warning("TS_HANDLER_LIST_ISSUES: Called without an active TV model in session.")
        # This implies the calling logic should have ensured an active model or asked for one.
        # For robustness, ask for it here if absolutely necessary, though it's better handled upstream.
        session.set_expectation("model_for_list_issues", details={"original_request": "list all issues"})
        return "I need a TV model to list its specific issues. Which model are you interested in?" 

    log.info(f"TS_HANDLER_LIST_ISSUES: Preparing English list of documented issues for model {active_model}.")

    documented_issues_en = [] 
    if data_store:
        for item in data_store: 
            if item.get("model", "").upper() == active_model.upper(): 
                issue_title_en = item.get("issue") 
                if issue_title_en and isinstance(issue_title_en, str):
                    documented_issues_en.append(issue_title_en.strip())

    if documented_issues_en:
        unique_issues_en = sorted(list(set(documented_issues_en))) 
        issues_list_str_md_en = "\n- ".join(unique_issues_en)
        return (
            f"For TV model '{active_model}', here are some of the documented issues I have information about:\n- {issues_list_str_md_en}\n\n"
            f"You can ask me for troubleshooting details on any specific issue from this list. "
            f"You can also ask about diagrams or component lists if available for this model."
        ) 
    else:
        log.info(f"TS_HANDLER_LIST_ISSUES: No documented issues found in RAG data for model {active_model}.")
        return (
            f"I don't have a pre-compiled list of all known issues for TV model '{active_model}' in my current knowledge base. "
            f"However, if you can describe a specific problem you are facing with this model, I'll do my best to help. "
            f"You can also ask if I have diagrams or component information for this model."
        ) 


async def handle_session_follow_up_llm_call(user_query_original_lang: str, session: ChatSession) -> str | None:
    # Returns ENGLISH core response string
    if not session.active_tv_model and not session.in_troubleshooting_flow and not session.current_problem_description:
        log.warning("TS_HANDLER_FOLLOWUP_LLM: Called without sufficient context (active model, flow, or problem). Misrouted?")
        # This might indicate a need to re-route to initial_interaction_handler if session state is too vague.
        # For now, provide a generic response.
        return "I'm not sure what TV or problem we were discussing. Could you clarify or start a new topic?"

    log.info(f"TS_HANDLER_FOLLOWUP_LLM: Preparing English response for follow-up query '{user_query_original_lang[:50]}...' "
             f"for model '{session.active_tv_model or 'N/A'}', problem '{session.current_problem_description or 'General context'}'.")

    lc_memory_messages = session.get_lc_memory_messages() 
    pdf_context_str = session.get_pdf_context_for_llm(max_chars=500) # Limit PDF context for this specific call

    context_for_llm_to_formulate_english_reply = f"""
    The user is in an ongoing session.
    - Current TV Model in Focus: {session.active_tv_model or 'Not specifically set yet.'}
    - Current Problem/Topic: {session.current_problem_description or ('General discussion about the TV model' if session.active_tv_model else 'General TV help or other topic.')}
    - User's latest message (in their language, which you should understand from history if different from English): "{user_query_original_lang}"
    {pdf_context_str}

    Your task is to formulate a helpful **English** response to the user's latest message, considering the conversation history provided.
    - If they are asking for clarification on previous steps you (the assistant) provided, try to clarify in English.
    - If they say a step didn't work or provide an update, acknowledge this in English and suggest what to do next (e.g., "Let's try the next step," or "Could you tell me more about what happened?"). If appropriate, you can also offer to check for related diagrams/components for the active model.
    - If they ask a new question related to the current problem or active model, answer it in English.
    - If they seem to describe a completely new problem (distinct from the current '{session.current_problem_description}'):
        - If the new problem clearly mentions a TV model (either the current active one or a new one), your English response should first briefly address their immediate query if possible, and then **at the very end of your English response, include the special token `NEW_PROBLEM_SUGGESTION: [Your concise English summary of the new problem, max 15 words]. Model: [NewlyMentionedOrActiveModelName]`**.
        - If the new problem is generic (no model mentioned), use `NEW_PROBLEM_SUGGESTION: [Concise summary]. Model: None`.
        - Do NOT include this token if it's not clearly a new, distinct problem.
    - If they are asking for media (images, diagrams, component lists) for the **active model** (e.g., "show me the motherboard for {session.active_tv_model}"), acknowledge this request clearly in English. For example: "Okay, you're looking for the motherboard image for model {session.active_tv_model}. I can check on that." The actual image display or list generation will be handled by another specialized function based on your English acknowledgement.
    - If they ask for media for a **different model** than the active one, acknowledge this and also use the NEW_PROBLEM_SUGGESTION token to flag a potential context switch for the media request. E.g., "You want the block diagram for SONY-XYZ. NEW_PROBLEM_SUGGESTION: Media request for different model. Model: SONY-XYZ"
    - Be concise and helpful in your English response.
    - Use Markdown for good formatting (paragraphs, lists, bolding) in your English response.

    Generate only the **English** response to the user. Do not add any conversational intros like "Okay, I will respond in English."
    """
    
    system_prompt_for_english_formulation = (
        "You are an AI assistant helping to formulate an intermediate **English** response "
        "within a TV troubleshooting/assistance chatbot. The user might be speaking another language (evident from history). "
        "Your output will be an English text that will later be translated if needed. "
        "Follow the instructions in the user-provided context precisely, especially regarding the `NEW_PROBLEM_SUGGESTION` token "
        "and how to acknowledge requests for images/components. Your response should be well-formatted Markdown."
    )

    english_response_from_llm = await call_groq_llm_final_answer( 
        user_context_for_current_turn=context_for_llm_to_formulate_english_reply,
        target_language_name="English", 
        dialect_context_hint=None,      
        memory_messages=lc_memory_messages, 
        system_prompt_template_str=system_prompt_for_english_formulation
    )

    if english_response_from_llm and not english_response_from_llm.startswith("Error:"):
        log.info(f"TS_HANDLER_FOLLOWUP_LLM: Successfully formulated English response: '{english_response_from_llm[:100]}...'")
        return english_response_from_llm 
    else:
        log.warning(f"TS_HANDLER_FOLLOWUP_LLM: Failed to get English formulation from LLM for query '{user_query_original_lang[:50]}...'. LLM Error: {english_response_from_llm}")
        return (f"I'm having a bit of trouble processing that follow-up request in the context of TV model '{session.active_tv_model or 'your TV'}' "
                f"and the problem '{session.current_problem_description or 'our current topic'}'. "
                f"Could you please try rephrasing your last message?")