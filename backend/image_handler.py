# backend/image_handler.py
import logging
import sys

try:
    from groq_api import call_groq_llm_final_answer_lc as call_groq_llm_final_answer
    from session_manager import ChatSession
    from utils import extract_tv_model_from_query # <--- CHANGE THIS IMPORT
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR in image_handler.py: {e}. Application will likely fail.", file=sys.stderr)
    raise
# ... rest of image_handler.py remains the same as the last version you provided for it
# (Ensure all internal logic uses active_model_for_this_request as derived)
# ...
log = logging.getLogger(__name__)

async def format_component_info_for_llm( 
    tv_model_data: dict, 
    session: ChatSession,
) -> str | None: # Returns localized Markdown
    target_language_name = session.current_language_name
    dialect_hint = session.last_detected_dialect_info
    
    model_name_for_display = session.active_tv_model or tv_model_data.get("tv_model", "this TV model")
    if tv_model_data.get("tv_model") and session.active_tv_model and \
       tv_model_data.get("tv_model").upper() != session.active_tv_model.upper():
        log.warning(f"IMAGE_HANDLER_FORMAT: Mismatch - tv_model_data is for '{tv_model_data.get('tv_model')}' "
                    f"but active session model is '{session.active_tv_model}'. Using active session model for display.")
        model_name_for_display = session.active_tv_model

    lc_memory_messages = session.get_lc_memory_messages()

    if not tv_model_data or "key_components" not in tv_model_data or not tv_model_data.get("key_components"):
        log.warning(f"IMAGE_HANDLER_FORMAT: No key component info for model {model_name_for_display}")
        return await call_groq_llm_final_answer(
            user_context_for_current_turn=f"User asked for key component info for TV model '{model_name_for_display}', "
                                          f"but this is unavailable or empty. Politely inform user in {target_language_name} "
                                          f"that detailed component info isn't available for this model.",
            target_language_name=target_language_name,
            dialect_context_hint=dialect_hint,
            memory_messages=lc_memory_messages,
        ) 

    components = tv_model_data.get("key_components", [])
    overview_en = tv_model_data.get("overview", "") 

    comp_details_list_md_en = [
        f"- **{c.get('component_id', f'C{i+1}')} ({c.get('name_en', 'Component')})**: "
        f"{c.get('description_en', c.get('description', 'N/A'))}"
        for i, c in enumerate(components)
    ]
    comp_details_md_str_en = "\n".join(comp_details_list_md_en)

    user_context_for_llm_en = (
        f"User Language is: {target_language_name} (Dialect context: {dialect_hint or 'General'}).\n"
        f"Task: Present key components information for TV model '{model_name_for_display}'.\n"
        f"Model Overview (English, if available): \"{overview_en or 'No overview available.'}\"\n\n"
        f"Key Components List (English, raw data, to be presented as a Markdown list):\n{comp_details_md_str_en}\n\n"
        f"Instructions for your response (MUST FOLLOW EXACTLY for {target_language_name}):\n"
        f"1. Respond ENTIRELY in {target_language_name}. If a dialect is specified, use it naturally if appropriate.\n"
        f"2. Use Markdown formatting extensively: main heading like `## Key Components for {model_name_for_display}` (translated), "
        f"subheadings like `### Overview` (translated, if overview exists) and `### Key Components List` (translated).\n"
        f"3. If an overview is provided above (English), include its translation as a paragraph under the `### Overview` subheading (translated).\n"
        f"4. Present the 'Key Components List' under the `### Key Components List` subheading. Use Markdown bullet points. "
        f"Translate the component names (e.g., 'Motherboard', 'Power Supply') and their descriptions from the provided English list into {target_language_name}. "
        f"If not confident for a specific technical term, you can leave it in English or transliterate if common practice for {target_language_name}.\n"
        f"5. Ensure good readability with blank lines between sections.\n"
        f"6. Do NOT add any conversational fluff like 'Sure, here is...' or 'Okay, I can help...'. Just provide the structured information."
    )
    
    system_prompt_template = (
        "You are an AI assistant specializing in presenting technical TV component information. "
        "Your task is to take the provided English data about TV components and generate a clearly formatted, localized explanation for the user. "
        "You MUST use Markdown for all formatting (headings, subheadings, bulleted lists, bold text). "
        "Adhere strictly to the formatting instructions provided in the user's message. "
        "Respond ENTIRELY in {{target_language_name}}. {{dialect_context_hint}}"
    )
    
    return await call_groq_llm_final_answer(
        user_context_for_current_turn=user_context_for_llm_en, 
        target_language_name=target_language_name,
        dialect_context_hint=dialect_hint,
        memory_messages=lc_memory_messages, 
        system_prompt_template_str=system_prompt_template
    )


async def handle_image_component_query(
    user_query: str, 
    session: ChatSession,
    data_store, 
    components_data_store, 
    image_base_path: str
) -> str | None: 
    target_lang_name = session.current_language_name
    dialect_hint = session.last_detected_dialect_info
    active_model_for_this_request = session.active_tv_model 
    lc_memory_messages = session.get_lc_memory_messages()

    model_mentioned_in_query = extract_tv_model_from_query(user_query)
    if model_mentioned_in_query and model_mentioned_in_query != active_model_for_this_request:
        log.info(f"IMAGE_HANDLER: Model '{model_mentioned_in_query}' mentioned in media query. "
                 f"Using this for current request instead of session active '{active_model_for_this_request}'.")
        active_model_for_this_request = model_mentioned_in_query
        if active_model_for_this_request not in session.recognized_tv_models:
             session.add_recognized_model(active_model_for_this_request)


    if not active_model_for_this_request:
        ask_model_ctx_parts_en = [f"I can help you with images or component details for a specific TV model."]
        if session.recognized_tv_models:
            models_list_str = ", ".join(session.recognized_tv_models)
            ask_model_ctx_parts_en.append(f"You've mentioned these models before: {models_list_str}.")
            ask_model_ctx_parts_en.append(f"Which of these are you asking about for your request: '{user_query[:30].strip()}'?")
            ask_model_ctx_parts_en.append(f"Or, if it's a different model, please provide its name.")
            session.set_expectation("model_for_media_request", details={"media_query": user_query, "options": session.recognized_tv_models})
        else:
            ask_model_ctx_parts_en.append(f"To find what you're looking for ('{user_query[:30].strip()}...'), I need the TV model name.")
            session.set_expectation("model_for_media_request", details={"media_query": user_query})
        
        ask_model_ctx_en = " ".join(ask_model_ctx_parts_en)
        return ask_model_ctx_en 

    log.info(f"IMAGE_HANDLER: Processing media query for model '{active_model_for_this_request}': '{user_query[:100]}...' UserLang: {target_lang_name}")

    requested_items = [] 
    not_found_image_types_en = [] 

    general_model_images_to_use = None
    if session.active_tv_model == active_model_for_this_request and session.current_model_general_images:
        general_model_images_to_use = session.current_model_general_images
    else: 
        if data_store and active_model_for_this_request:
            model_entry_for_images = next((item for item in data_store if item.get("model","").upper() == active_model_for_this_request.upper() and item.get("images")), None)
            if model_entry_for_images and model_entry_for_images.get("images"):
                general_model_images_to_use = model_entry_for_images.get("images")
                if session.active_tv_model == active_model_for_this_request:
                    session.current_model_general_images = general_model_images_to_use
                log.info(f"IMAGE_HANDLER: Loaded general images for model '{active_model_for_this_request}'.")
            else:
                log.warning(f"IMAGE_HANDLER: Could not find general images for model '{active_model_for_this_request}' in data_store.")

    model_specific_component_data = next((item for item in (components_data_store or []) if item.get("tv_model","").upper() == active_model_for_this_request.upper()), None)

    image_definitions = [
        {"type_en": "Motherboard Image", "keywords_en": ["motherboard image", "main board image", "logic board image", "carte mere image"], "data_key": "motherboard", "source": "general"},
        {"type_en": "Block Diagram", "keywords_en": ["block diagram", "schema", "schematic", "diagramme fonctionnel"], "data_key": "block_diagram", "source": "general"},
        {"type_en": "Key Components Overview Image", "keywords_en": ["key components image", "general components image", "components picture", "parts image", "image composants clés"], "data_key": "key_components", "source": "general_or_specific_image"}, 
        {"type_en": "Detailed Key Components Diagram", "keywords_en": ["component diagram", "detailed component image", "detailed diagram", "exploded view", "parts list diagram", "diagramme détaillé des composants"], "data_key": "image_filename", "source": "specific_image_only"} 
    ]

    user_query_lower = user_query.lower() 
    for definition in image_definitions:
        if any(kw_en in user_query_lower for kw_en in definition["keywords_en"]): 
            log.debug(f"IMAGE_HANDLER: Keyword match for '{definition['type_en']}' in query '{user_query_lower[:50]}...' for model {active_model_for_this_request}.")
            filename = None
            if definition["source"] == "general" and general_model_images_to_use:
                filename = general_model_images_to_use.get(definition["data_key"])
            elif definition["source"] == "specific_image_only" and model_specific_component_data:
                filename = model_specific_component_data.get(definition["data_key"]) 
            elif definition["source"] == "general_or_specific_image":
                if general_model_images_to_use: 
                    filename = general_model_images_to_use.get(definition["data_key"]) 
                if not filename and model_specific_component_data: 
                    filename = model_specific_component_data.get("image_filename") 

            if filename:
                if not any(item['filename'] == filename for item in requested_items):
                    requested_items.append({
                        "type_en": definition["type_en"], "filename": filename,
                        "markdown_path": f"{image_base_path}{filename}", 
                        "alt_text_en": f"{definition['type_en']} for {active_model_for_this_request}"
                    })
                    log.info(f"IMAGE_HANDLER: Added '{filename}' for '{definition['type_en']}' (model {active_model_for_this_request}).")
            else: 
                if definition["type_en"] not in [item['type_en'] for item in requested_items] and \
                   definition["type_en"] not in not_found_image_types_en:
                    not_found_image_types_en.append(definition["type_en"])
                log.warning(f"IMAGE_HANDLER: Requested '{definition['type_en']}' for {active_model_for_this_request}, but no filename found.")

    list_components_keywords_en = ["list components", "key components list", "what components", "component details", "tell me about components", "liste composants", "détails composants"]
    is_asking_for_list_explicitly = any(kw_en in user_query_lower for kw_en in list_components_keywords_en)

    component_list_md_str_en = None
    component_list_overview_en = None
    if model_specific_component_data and model_specific_component_data.get("key_components"):
        if is_asking_for_list_explicitly or \
           any(item['type_en'] in ["Key Components Overview Image", "Detailed Key Components Diagram"] for item in requested_items):
            components = model_specific_component_data.get("key_components", [])
            component_list_overview_en = model_specific_component_data.get("overview", "") 
            comp_details_list_md_parts_en = [
                f"- **{c.get('component_id', f'C{i+1}')} ({c.get('name_en', 'Component')})**: {c.get('description_en', c.get('description', 'N/A'))}"
                for i, c in enumerate(components)
            ]
            component_list_md_str_en = "\n".join(comp_details_list_md_parts_en)
            log.info(f"IMAGE_HANDLER: Prepared textual component list (English) for {active_model_for_this_request}.")

    if requested_items or (component_list_md_str_en and is_asking_for_list_explicitly):
        llm_context_parts_en = [
            f"User Language is: {target_lang_name} (Dialect context: {dialect_hint or 'General'}).",
            f"TV Model in Focus: '{active_model_for_this_request}'. User's original query was: '{user_query}'.\n",
            "Your Task: Respond to the user by presenting the information found. Use Markdown for all formatting. "
            "Translate all descriptive text, headings, and alt text into the User Language."
        ]
        
        if not requested_items and component_list_md_str_en and is_asking_for_list_explicitly:
             llm_context_parts_en.append(f"\nThe user specifically asked for a list of key components for TV model '{active_model_for_this_request}'.")
        elif requested_items:
            llm_context_parts_en.append(f"\nThe user asked for images/diagrams related to TV model '{active_model_for_this_request}'. Here's what was found (present each item clearly):")
        
        item_number = 1
        for item in requested_items:
            llm_context_parts_en.append(
                f"\nItem {item_number} (Type: {item['type_en']}):\n"
                f"- Present this under a translated heading like: `## {item['type_en']} for {active_model_for_this_request}`.\n"
                f"- Include a brief introductory sentence (translated).\n"
                f"- Display the image using this exact Markdown structure: `![Translated Alt Text for: {item['alt_text_en']}]({item['markdown_path']})` (Ensure the alt text is translated from '{item['alt_text_en']}')"
            )
            if (item['type_en'] == "Key Components Overview Image" or item['type_en'] == "Detailed Key Components Diagram") and component_list_md_str_en:
                llm_context_parts_en.append(
                    f"- After this image, if an overview is available (English Overview: \"{component_list_overview_en or 'None'}\"), translate and include it under a translated '### Overview' subheading.\n"
                    f"- Then, present the key components list (provided below in English) under a translated '### Key Components List' subheading. Translate component names and descriptions.\n"
                )
                llm_context_parts_en.append(f"English Component List Data to translate and format:\n{component_list_md_str_en}\n")
                component_list_md_str_en = None 
                component_list_overview_en = None
            item_number += 1

        if component_list_md_str_en: 
            llm_context_parts_en.append(
                f"\nItem {item_number} (Type: Key Components List):\n"
                f"- Present this under a translated heading like: `## Key Components for {active_model_for_this_request}`.\n"
            )
            if component_list_overview_en:
                 llm_context_parts_en.append(f"- If an overview is available (English Overview: \"{component_list_overview_en or 'None'}\"), translate and include it under a translated '### Overview' subheading.\n")
            llm_context_parts_en.append(
                f"- Present the key components list (provided below in English) under a translated '### Key Components List' subheading. Translate component names and descriptions.\n"
                f"English Component List Data to translate and format:\n{component_list_md_str_en}\n"
            )

        if not_found_image_types_en:
            not_found_str_en = ", ".join(f"'{t}'" for t in not_found_image_types_en)
            llm_context_parts_en.append(
                f"\nInformation Not Found:\n- Politely inform the user (translated) that the following item(s) could not be found for model '{active_model_for_this_request}': {not_found_str_en}.\n"
                f"- Suggest they check the TV's official manual or website if available."
            )
        
        llm_context_parts_en.append("\nGeneral Formatting: Ensure the entire response is in {target_language_name}. Use Markdown for headings, lists, and emphasis. Separate sections with blank lines for readability. Do NOT add any conversational fluff like 'Sure!' or 'Okay'.")

        final_llm_context_en = "".join(llm_context_parts_en)
        
        system_prompt_template = (
            "You are a helpful TV technical assistant. Your goal is to clearly present images and component information to the user in their language, using Markdown. "
            "You will receive structured English data and instructions. Your output should be the final, formatted, and localized response for the user. "
            "Follow all instructions provided in the user message for structure and content. Respond ENTIRELY in {{target_language_name}}. {{dialect_context_hint}}"
        )
        return await call_groq_llm_final_answer(
            user_context_for_current_turn=final_llm_context_en, 
            target_language_name=target_lang_name,
            dialect_context_hint=dialect_hint,
            memory_messages=lc_memory_messages, 
            system_prompt_template_str=system_prompt_template
        )

    elif is_asking_for_list_explicitly and model_specific_component_data :
        log.info(f"IMAGE_HANDLER: Explicit request for component list ONLY for {active_model_for_this_request}. Formatting with format_component_info_for_llm.")
        return await format_component_info_for_llm(model_specific_component_data, session)
        
    else: 
        log.info(f"IMAGE_HANDLER: No specific images/list identified, or data not found for query '{user_query}' for model {active_model_for_this_request}. Clarifying.")
        
        clarification_context_parts_en = [
            f"User Language is: {target_lang_name} (Dialect: {dialect_hint or 'General'}). TV Model in Focus: '{active_model_for_this_request}'. User's original query was: '{user_query}'\n",
            f"Analysis: The system could not pinpoint a specific image or component list based on the user's query, or the necessary data for model '{active_model_for_this_request}' is unavailable.\n",
            f"Task for your response (translate all user-facing text to {target_lang_name}):\n",
            f"1. Acknowledge the user's query about images/components for TV model '{active_model_for_this_request}'.\n",
            f"2. State that the specific item(s) from their query (e.g., \"{user_query[:30].strip()}...\") could not be found or clearly identified for model '{active_model_for_this_request}'.\n",
            f"3. Ask for clarification. Provide examples of what *might* be available for TV models in general. Present these examples as a Markdown bulleted list (translate these English examples):\n",
            f"   - A motherboard image?\n",
            f"   - A general key components overview image?\n",
            f"   - A block diagram (schematic)?\n",
            f"   - A detailed diagram of key components (if available for the model)?\n",
            f"   - Or a textual list of key components with descriptions?\n",
            f"4. Conclude by asking the user to specify which of these (or something else) they are looking for regarding model '{active_model_for_this_request}'."
        ]
        clarification_context_en = "".join(clarification_context_parts_en)
        system_prompt_clarify = (
            "You are a helpful TV technical assistant. Your task is to clarify the user's request for images or component information "
            "when their initial query was too vague or the data was not found for the specified model. Respond clearly in {{target_language_name}}. Use Markdown for lists. {{dialect_context_hint}}"
        )
        # This call generates the final, localized, Markdown-formatted clarification
        return await call_groq_llm_final_answer(
            user_context_for_current_turn=clarification_context_en, 
            target_language_name=target_lang_name, 
            dialect_context_hint=dialect_hint,
            memory_messages=lc_memory_messages, 
            system_prompt_template_str=system_prompt_clarify
        )