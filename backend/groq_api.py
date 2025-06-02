# backend/groq_api.py
import httpx # For potential future direct API calls, though not used in current Langchain functions
import logging
import os
import json
from dotenv import load_dotenv

# Langchain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser # JsonOutputParser is also available if needed
from pydantic import BaseModel, Field # <--- UPDATED FOR PYDANTIC V2
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage 
from langchain_groq import ChatGroq

from typing import List, Dict, Any, Union 

load_dotenv()
log = logging.getLogger(__name__)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    log.warning("GROQ_API_KEY not set. LLM calls will likely fail.")

# --- Reusable Langchain LLM Instances ---
DEFAULT_GROQ_CHAT_MODEL = "llama-3.1-8b-instant" 
DEFAULT_GROQ_TRANSLATE_MODEL = "llama-3.1-8b-instant"
DEFAULT_GROQ_CLASSIFY_MODEL = "llama-3.1-8b-instant"

try:
    chat_llm = ChatGroq(temperature=0.5, model_name=DEFAULT_GROQ_CHAT_MODEL, groq_api_key=GROQ_API_KEY)
    translate_llm = ChatGroq(temperature=0.05, model_name=DEFAULT_GROQ_TRANSLATE_MODEL, groq_api_key=GROQ_API_KEY)
    classify_llm = ChatGroq(temperature=0.0, model_name=DEFAULT_GROQ_CLASSIFY_MODEL, groq_api_key=GROQ_API_KEY)
    hyde_llm = ChatGroq(temperature=0.05, model_name=DEFAULT_GROQ_CHAT_MODEL, groq_api_key=GROQ_API_KEY)
except Exception as e:
    log.critical(f"Failed to initialize ChatGroq instances: {e}. Check API key and model names.", exc_info=True)
    chat_llm = translate_llm = classify_llm = hyde_llm = None


AFFIRMATIVE_WORDS_API = ["yes", "yeah", "yep", "yup", "sure", "ok", "okay", "alright", "affirmative", "indeed", "certainly", "please do", "go ahead", "absolutely", "fine"]
NEGATIVE_WORDS_API = ["no", "nope", "nah", "negative", "don't", "do not", "cancel", "stop", "not really", "not now", "nay", "never"]


async def translate_text_lc(
    text_to_translate: str, source_language_name: str, target_language_name: str,
    dialect_context_hint: str | None = None,
    context_hint_for_translation: str | None = None
) -> str | None:
    if not translate_llm:
        log.error("Translate LLM not initialized. Cannot perform translation.")
        return "Error: Translation service unavailable."
    if not text_to_translate or not text_to_translate.strip():
        log.warning("Translate_text_lc called with empty input.")
        return ""

    system_prompt_parts = [f"You are an expert multilingual translator."]
    if context_hint_for_translation:
        system_prompt_parts.append(f"The text to translate is related to: {context_hint_for_translation}.")

    effective_source_lang_desc = source_language_name
    if source_language_name == "Arabic" and dialect_context_hint and \
       any(d_indicator in dialect_context_hint.lower() for d_indicator in ["darija", "dziribert", "heuristic"]):
        effective_source_lang_desc = "Algerian Darija (an Arabic dialect)"
        system_prompt_parts.append(f"The source text is specifically in {effective_source_lang_desc}.")

    system_prompt_parts.append(f"Your task is to translate the following text accurately and naturally from {effective_source_lang_desc} to {target_language_name}.")
    system_prompt_parts.append(f"Preserve the original meaning and tone as much as possible.")
    system_prompt_parts.append(f"Output ONLY the translated text, without any additional explanations, comments, or quotation marks wrapping the entire translation.")
    system_message = " ".join(system_prompt_parts)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "Text to translate:\n\"\"\"\n{text_to_translate}\n\"\"\"")
    ])
    chain = prompt | translate_llm | StrOutputParser()
    try:
        translated_text = await chain.ainvoke({
            "text_to_translate": text_to_translate
        })
        if translated_text:
            if translated_text.startswith('"') and translated_text.endswith('"') and len(translated_text) > 1:
                translated_text = translated_text[1:-1]
            if translated_text.startswith("'") and translated_text.endswith("'") and len(translated_text) > 1:
                translated_text = translated_text[1:-1]
            log.info(f"LC Translation to {target_language_name} successful: '{translated_text[:70]}...'")
            return translated_text.strip()
        log.warning(f"LC Translation to {target_language_name} resulted in empty or None output for input: '{text_to_translate[:50]}...'")
        return None
    except Exception as e:
        log.error(f"LC Translation failed for '{text_to_translate[:50]}...': {e}", exc_info=True)
        return f"Error: Translation LLM call failed."


async def call_groq_llm_final_answer_lc(
    user_context_for_current_turn: str, target_language_name: str,
    dialect_context_hint: str | None = None,
    memory_messages: List[BaseMessage] | None = None, # Updated type hint
    system_prompt_template_str: str | None = None,
) -> str | None:
    if not chat_llm:
        log.error("Chat LLM not initialized. Cannot generate final answer.")
        return "Error: Chat service unavailable."

    effective_system_prompt = ""
    if system_prompt_template_str:
        temp_prompt = system_prompt_template_str.replace("{{target_language_name}}", target_language_name)
        temp_prompt = temp_prompt.replace("{{dialect_context_hint}}", dialect_context_hint or f"general {target_language_name}")
        effective_system_prompt = temp_prompt
    else:
        effective_system_prompt = f"You are a helpful AI assistant. Respond ENTIRELY in {target_language_name}."
        if dialect_context_hint:
             effective_system_prompt += f" Pay attention to any dialect context: {dialect_context_hint}."

    is_darija_context = dialect_context_hint and \
                        any(d_indicator in dialect_context_hint.lower() for d_indicator in [
                            "darija_confirmed_dziribert", "darija_explicit_request", "darija_heuristic_keywords"])
    is_arabic_target = target_language_name == "Arabic"
    if is_darija_context and is_arabic_target:
        darija_instruction = (
            f" IMPORTANT INSTRUCTION: The user is likely communicating in or expecting Algerian Darija. "
            f"When responding in Arabic, you MUST use clear, simple, and natural-sounding Algerian Darija. "
            # ... (rest of Darija instruction)
        )
        if "Darija" not in effective_system_prompt and "Algerian" not in effective_system_prompt:
             effective_system_prompt += f" {darija_instruction}"
    elif dialect_context_hint and is_arabic_target and \
         any(msa_indicator in dialect_context_hint for msa_indicator in [
            "arabic_msa_request", "arabic_generic_not_darija_dzb", "arabic_generic_content"]):
         if "Modern Standard Arabic" not in effective_system_prompt and "Fusha" not in effective_system_prompt:
            effective_system_prompt += f" Ensure your Arabic response is in clear Modern Standard Arabic (Fusha)."

    messages_for_prompt_list: list = [("system", effective_system_prompt.strip())] # Ensure list type for from_messages
    if memory_messages:
        messages_for_prompt_list.append(MessagesPlaceholder(variable_name="history"))
    messages_for_prompt_list.append(("human", "{input}"))

    prompt = ChatPromptTemplate.from_messages(messages_for_prompt_list)
    chain = prompt | chat_llm | StrOutputParser()

    input_dict: Dict[str, Any] = {"input": user_context_for_current_turn}
    if memory_messages:
        input_dict["history"] = memory_messages

    try:
        response = await chain.ainvoke(input_dict)
        log.info(f"LC Final Answer (target: {target_language_name}, input: '{user_context_for_current_turn[:50]}...'): '{str(response)[:100]}...'")
        return response
    except Exception as e:
        log.error(f"LC Final Answer LLM call failed for input '{user_context_for_current_turn[:50]}...': {e}", exc_info=True)
        return f"Error: LLM call failed while processing your request."


async def generate_hypothetical_document_lc(user_query_english: str) -> str | None:
    if not hyde_llm:
        log.error("HyDE LLM not initialized. Cannot generate hypothetical document.")
        return "Error: HyDE service unavailable."
    system_prompt = (
        "You are an expert system that generates concise, technical, English search query titles "
        # ... (rest of HyDE prompt)
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "User's TV problem (English): \"{user_query_english}\"\n\nTechnical Search Title (English):")
    ])
    chain = prompt | hyde_llm | StrOutputParser()
    try:
        response = await chain.ainvoke({"user_query_english": user_query_english})
        if response:
            response = response.replace('"', '').replace("'", '').replace("Title:", "").strip()
            log.info(f"LC HyDE generation successful: '{response}' for query '{user_query_english[:50]}...'")
            return response if response else None
        return None
    except Exception as e:
        log.error(f"LC HyDE generation failed for query '{user_query_english[:50]}...': {e}", exc_info=True)
        return f"Error: HyDE LLM call failed."


class MainIntentOutput(BaseModel):
    intent: str = Field(description=(
        "Classify the user's primary intent. Categories: "
        "'general_question', "
        "'standard_tv_troubleshooting' (user describes a TV problem but NO specific model is mentioned), "
        "'specific_tv_troubleshooting' (user describes a TV problem AND mentions a specific TV model name/number), "
        "'media_request_model_specific' (user asks for image/diagram for a specific model), "
        "'media_request_generic' (user asks for image/diagram without specific model yet), "
        "'follow_up_clarification' (user is responding to a bot's question, e.g. 'yes', 'tell me more', or asking for more details on previous bot statement), "
        "'other_unclear' (intent is not clear from the above categories)."
    ))
    extracted_model_if_any: str | None = Field(None, description="If 'specific_tv_troubleshooting' or 'media_request_model_specific', extract the TV model. Otherwise null.")


async def classify_main_intent_and_extract_model_lc(
    user_query: str,
    target_language_name: str,
    dialect_context_hint: str | None = None,
    chat_history_summary_for_intent: str | None = None 
) -> MainIntentOutput | None:
    if not classify_llm:
        log.error("Classify LLM not initialized. Cannot classify main intent.")
        return None # Or return MainIntentOutput(intent='other_unclear', extracted_model_if_any=None)

    try:
        structured_llm_intent = classify_llm.with_structured_output(MainIntentOutput)
    except Exception as e_struct:
        log.error(f"Failed to set classify_llm for structured output (MainIntent): {e_struct}. Classification may fail or be less reliable.")
        return MainIntentOutput(intent='other_unclear', extracted_model_if_any=None)

    system_prompt_parts = [
        f"You are an expert intent classification assistant for a TV troubleshooting chatbot.",
        # ... (rest of the detailed system prompt for MainIntentOutput as provided before) ...
         "Focus on the LATEST user query. The chat history summary is for overall context if the query is short or ambiguous."
    ]
    if chat_history_summary_for_intent:
        system_prompt_parts.append(f"\nBrief Chat History Context (for ambiguity resolution only):\n{chat_history_summary_for_intent}")

    system_message = "\n".join(system_prompt_parts)
    human_message_template = "User's latest query: \"{query}\"\n\nYour JSON classification:"

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", human_message_template)
    ])
    chain = prompt | structured_llm_intent

    try:
        response_data: MainIntentOutput = await chain.ainvoke({"query": user_query})
        log.info(f"LC Main Intent: Intent='{response_data.intent}', Model='{response_data.extracted_model_if_any}' for query: '{user_query[:50]}...'")
        
        if response_data.extracted_model_if_any:
            model_cand = response_data.extracted_model_if_any.strip().upper()
            if not (3 <= len(model_cand) <= 25 and any(c.isalnum() for c in model_cand)):
                log.warning(f"Invalid model '{model_cand}' from LLM (main intent), nullifying.")
                response_data.extracted_model_if_any = None
            else: 
                if len(model_cand) <= 4 and model_cand.lower() in (AFFIRMATIVE_WORDS_API + NEGATIVE_WORDS_API + ["tv", "t.v.", "model", "help", "fix", "sony", "lg", "samsung"]):
                    log.warning(f"Common short word '{model_cand}' from LLM (main intent), nullifying.")
                    response_data.extracted_model_if_any = None
                else:
                    response_data.extracted_model_if_any = model_cand
        return response_data
    except Exception as e:
        log.error(f"LC Main Intent LLM call/parsing failed for query '{user_query[:50]}...': {e}", exc_info=True)
        return MainIntentOutput(intent='other_unclear', extracted_model_if_any=None)


class FollowUpIntentOutput(BaseModel):
    intent: str = Field(description="One of 'affirmative', 'negative', 'provided_model', 'unclear_or_other', 'problem_solved', 'new_topic_unrelated'")
    extracted_model: str | None = Field(None, description="TV_MODEL_IDENTIFIER_OR_NULL if intent is 'provided_model'")

async def classify_follow_up_intent_lc(
    user_query: str, bot_s_previous_question_context: str, target_language_name: str,
    dialect_context_hint: str | None = None,
    memory_messages: List[BaseMessage] | None = None, # Updated type hint
) -> tuple[str, str | None]: 
    if not classify_llm:
        log.error("Classify LLM (follow-up) not initialized.")
        return "unclear_or_other", None

    try:
        structured_llm_follow_up = classify_llm.with_structured_output(FollowUpIntentOutput)
    except Exception as e_struct_follow_up:
        log.error(f"Failed to set classify_llm for structured output (FollowUp): {e_struct_follow_up}.")
        return "unclear_or_other", None

    system_prompt_for_json = f"""You are an AI assistant. Your task is to interpret the user's response in the context of an ongoing conversation.
The user is speaking {target_language_name} (additional dialect context: {dialect_context_hint or 'general'}).
The chatbot previously asked something like: "{bot_s_previous_question_context}"
# ... (rest of the detailed system prompt for FollowUpIntentOutput as provided before) ...
Respond ONLY with a valid JSON object matching the defined schema.
"""
    messages_for_prompt_list: list = [("system", system_prompt_for_json)] # Ensure list type
    if memory_messages:
        messages_for_prompt_list.append(MessagesPlaceholder(variable_name="history"))
    messages_for_prompt_list.append(("human",
        "Chatbot's previous question context was: \"{bot_q_context}\"\n"
        "The user's current response (in {target_lang}, dialect hint: {dialect}) is: \"{user_input}\"\n\n"
        "Your JSON classification based on the user's current response and history:"
    ))

    prompt = ChatPromptTemplate.from_messages(messages_for_prompt_list)
    chain = prompt | structured_llm_follow_up

    input_dict: Dict[str, Any] = {
        "bot_q_context": bot_s_previous_question_context,
        "target_lang": target_language_name,
        "dialect": dialect_context_hint or "N/A",
        "user_input": user_query
    }
    if memory_messages:
        input_dict["history"] = memory_messages
    
    try:
        data: FollowUpIntentOutput = await chain.ainvoke(input_dict)
        intent_cat = data.intent
        extracted_mod_candidate = data.extracted_model

        valid_intents = ["affirmative", "negative", "provided_model", "unclear_or_other", "problem_solved", "new_topic_unrelated"]
        if intent_cat in valid_intents:
            if intent_cat == "provided_model":
                if not extracted_mod_candidate or not isinstance(extracted_mod_candidate,str) or \
                   len(extracted_mod_candidate.strip()) < 3 or \
                   (len(extracted_mod_candidate.strip()) <= 4 and \
                    extracted_mod_candidate.strip().lower() in (AFFIRMATIVE_WORDS_API + NEGATIVE_WORDS_API + ["tv","model","t.v."])):
                    log.warning(f"LC Follow-up: 'provided_model' but model '{extracted_mod_candidate}' invalid/common. Re-classifying as 'unclear_or_other'. Query: '{user_query}'")
                    return "unclear_or_other", None
                log.info(f"LC Follow-up: Intent='{intent_cat}', Model='{extracted_mod_candidate.strip().upper()}' for query '{user_query[:50]}...'")
                return intent_cat, extracted_mod_candidate.strip().upper()
            
            log.info(f"LC Follow-up: Intent='{intent_cat}' for query '{user_query[:50]}...'")
            return intent_cat, None
        else:
            log.warning(f"LC Follow-up: LLM returned invalid intent category '{intent_cat}'. Full response: {data}. Query: '{user_query}'")
            return "unclear_or_other", None
    except Exception as e:
        log.error(f"LC Follow-up intent JSON parsing/LLM error for query '{user_query[:50]}...': {e}", exc_info=True)
        return "unclear_or_other", None