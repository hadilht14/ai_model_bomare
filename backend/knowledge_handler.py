# backend/knowledge_handler.py
import logging
try:
    # Use the new Langchain based function
    from groq_api import call_groq_llm_final_answer_lc as call_groq_llm_final_answer # Alias
    # DEFAULT_GROQ_CHAT_MODEL is imported in groq_api and used by call_groq_llm_final_answer
    from session_manager import ChatSession # Updated
except ImportError as e:
    print(f"CRITICAL IMPORT ERROR in knowledge_handler.py: {e}. Application will likely fail.")
    raise

log = logging.getLogger(__name__)

async def handle_general_knowledge_query(user_input: str, session: ChatSession) -> str | None:
    # This function's output is expected to be a fully formed Markdown string,
    # already localized by the LLM call.
    target_lang_name = session.current_language_name
    dialect_hint = session.last_detected_dialect_info
    lc_memory_msgs = session.get_lc_memory_messages() # Get Langchain memory messages
    pdf_context_str = session.get_pdf_context_for_llm() # Get formatted PDF context

    # Construct the main context for the LLM (in English, LLM will handle localization based on target_lang_name)
    llm_context_parts_en = [f"The user's question or statement is: \"{user_input}\""]
    if pdf_context_str:
        llm_context_parts_en.append(f"\nThere is also context available from a PDF document named '{session.pdf_context_source_filename or 'loaded PDF'}'. If this context is relevant to the user's question, please use it in your answer:\n---PDF Context Start---\n{pdf_context_str}\n---PDF Context End---")
    
    llm_context_parts_en.append("\nYour Task: Provide a concise and informative answer to the user's latest question/statement. "
                                "Prioritize any relevant PDF context if provided and applicable. "
                                "Use Markdown for clear formatting (e.g., paragraphs, lists, bold emphasis) in your response.")
    
    full_llm_context_en = "\n".join(llm_context_parts_en)
    
    # System prompt to guide the LLM's role and output style
    system_prompt_template = (
        "You are a helpful and knowledgeable AI assistant. Your primary goal is to answer the user's general questions accurately. "
        "Respond ENTIRELY in {{target_language_name}}. {{dialect_context_hint}} "
        "Focus on answering the LATEST user question/statement from the provided context. "
        "If the question is outside your primary domain (e.g., TV troubleshooting), answer it using your general knowledge. "
        "**Ensure your response is well-formatted using Markdown (headings, lists, paragraphs, bold text) for clarity and readability.**"
    )
    
    # This call generates the final, localized, Markdown-formatted response
    llm_response = await call_groq_llm_final_answer(
        user_context_for_current_turn=full_llm_context_en, # English context for LLM
        target_language_name=target_lang_name,
        dialect_context_hint=dialect_hint,
        memory_messages=lc_memory_msgs, # Pass the Langchain memory messages
        system_prompt_template_str=system_prompt_template
    )
    
    if llm_response and not llm_response.startswith("Error:"):
        log.info(f"KNOWLEDGE_HANDLER: Successfully generated response for query '{user_input[:50]}...'")
        return llm_response
    else:
        log.error(f"KNOWLEDGE_HANDLER: LLM error or no response for general query '{user_input[:50]}...'. LLM raw output: {llm_response}")
        # Fallback if LLM fails to generate a proper response
        error_ctx_en = f"I'm sorry, I encountered an issue while trying to answer your question: \"{user_input[:30]}...\". The specific error was: ({llm_response or 'No detailed error available'}). Could you please try rephrasing your question, or ask something different?"
        # Get a localized version of this fallback error
        return await call_groq_llm_final_answer(
            user_context_for_current_turn=error_ctx_en,
            target_language_name=target_lang_name,
            dialect_context_hint=dialect_hint,
            memory_messages=lc_memory_msgs # Pass memory for context even in error
        )