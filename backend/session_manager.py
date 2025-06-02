# backend/session_manager.py
import logging
import datetime
from typing import List, Dict, Any, Union

# Langchain imports
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

try:
    from language_handler import DEFAULT_LANGUAGE_CODE, SUPPORTED_LANGUAGES_MAP, get_language_name
except ImportError:
    logging.getLogger(__name__).critical(
        "session_manager: FAILED to import from language_handler. Using hardcoded defaults."
    )
    DEFAULT_LANGUAGE_CODE = "en"
    SUPPORTED_LANGUAGES_MAP = {"en": "English", "fr": "French", "ar": "Arabic"}
    def get_language_name(code): return SUPPORTED_LANGUAGES_MAP.get(code, "English")


log = logging.getLogger(__name__)
MAX_HISTORY_TURNS_UI = 15
MAX_HISTORY_TURNS_LLM = 7

class ChatSession:
    def __init__(self):
        # Core Session State
        self.current_language: str = DEFAULT_LANGUAGE_CODE
        self.current_language_name: str = get_language_name(DEFAULT_LANGUAGE_CODE)
        self.last_detected_dialect_info: str | None = None

        # TV Model Management
        self.recognized_tv_models: List[str] = [] # Stores all models mentioned in the session
        self.active_tv_model: str | None = None    # The model currently in focus for troubleshooting/media
        self.current_model_general_images: dict | None = None # General images for the active_tv_model

        # Troubleshooting State
        self.in_troubleshooting_flow: bool = False # General flag if any troubleshooting is active
        self.current_problem_description: str = "" # Problem related to active_tv_model

        # Inter-turn Expectation Flags (Manual State Flags)
        self.expecting_model_for_problem: str | None = None # Stores problem if bot asked for model
        self.expecting_confirmation_for: Dict[str, Any] | None = None # e.g., {"type": "new_problem", "details": {...}} or {"type": "elaboration", "last_topic": "..."}

        # UI History
        self.history_for_ui: list[dict] = []

        # Langchain Memory (for conversational context)
        self.memory = ConversationBufferWindowMemory(
            k=MAX_HISTORY_TURNS_LLM,
            memory_key="history", # Must match MessagesPlaceholder in ChatPromptTemplate
            return_messages=True  # Returns list of BaseMessage objects
        )

        # PDF Context
        self.pdf_context_text: str | None = None
        self.pdf_context_source_filename: str | None = None

        log.info(f"New ChatSession. Default lang: {self.current_language_name} ({self.current_language}), LLM Memory k={MAX_HISTORY_TURNS_LLM}")

    def add_to_history(self, role: str, content: str | None):
        if role not in ["user", "assistant", "system"]:
            log.warning(f"Invalid history role '{role}'. Valid roles are 'user', 'assistant', 'system'.")
            return
        if not content or not content.strip():
            log.debug(f"Skipping empty history entry for role '{role}'.")
            return

        timestamp = datetime.datetime.now().isoformat()
        content_stripped = content.strip()

        # Add to UI history
        self.history_for_ui.append({"role": role, "content": content_stripped, "timestamp": timestamp})
        if len(self.history_for_ui) > MAX_HISTORY_TURNS_UI * 2 + 5: # Keep UI history manageable
            self.history_for_ui = self.history_for_ui[-(MAX_HISTORY_TURNS_UI * 2 + 5):]

        # Add to Langchain memory
        if role == "user":
            self.memory.chat_memory.add_user_message(content_stripped)
        elif role == "assistant":
            self.memory.chat_memory.add_ai_message(content_stripped)
        # System messages can be added if they are crucial for LLM context
        # For example, a system message about a PDF being loaded could be:
        # elif role == "system" and "PDF context set" in content_stripped: # Example condition
        #    self.memory.chat_memory.add_message(SystemMessage(content=f"[System Note: {content_stripped}]"))


        log_content_preview = content_stripped[:60].replace('\n', ' ').replace('\r', '')
        log.debug(f"History add: Role={role}, Content='{log_content_preview}...' "
                  f"(UI items: {len(self.history_for_ui)}, "
                  f"LLM Mem messages: {len(self.memory.chat_memory.messages)})")

    def get_lc_memory_messages(self) -> List[BaseMessage]:
        """Returns the list of BaseMessage objects from Langchain memory."""
        return self.memory.chat_memory.messages

    def get_ui_history(self) -> list[dict]:
        """Returns history formatted for UI (list of dicts)."""
        return self.history_for_ui

    def clear_lc_memory(self):
        """Clears the Langchain conversational memory."""
        self.memory.clear()
        log.info("Langchain conversational memory cleared.")

    def set_language(self, lang_code: str, dialect_info: str | None = None):
        resolved_lang_code = lang_code if lang_code in SUPPORTED_LANGUAGES_MAP else DEFAULT_LANGUAGE_CODE
        resolved_lang_name = get_language_name(resolved_lang_code) # Use getter
        
        changed = False
        if self.current_language != resolved_lang_code:
            changed = True
        if self.last_detected_dialect_info != dialect_info: # Handles None vs some string
            changed = True
        
        if changed:
            log.info(f"Session lang updating from {self.current_language_name} (Code: {self.current_language}, Dialect: {self.last_detected_dialect_info or 'N/A'}) "
                     f"to {resolved_lang_name} (Code: {resolved_lang_code}, Dialect: {dialect_info or 'N/A'})")
            self.current_language = resolved_lang_code
            self.current_language_name = resolved_lang_name
            self.last_detected_dialect_info = dialect_info
        elif self.current_language_name != resolved_lang_name : # Ensure name is consistent if code was already same
            self.current_language_name = resolved_lang_name

    def add_recognized_model(self, model_name: str):
        if model_name and isinstance(model_name, str):
            model_upper = model_name.strip().upper()
            if model_upper and model_upper not in self.recognized_tv_models:
                self.recognized_tv_models.append(model_upper)
                log.info(f"Model '{model_upper}' added to recognized models. All: {self.recognized_tv_models}")
                if not self.active_tv_model: # If no active model, make this the active one
                    self.set_active_model(model_upper, "newly_recognized")

    def set_active_model(self, model_name: str | None, reason: str = "user_request"):
        old_active = self.active_tv_model
        if model_name is None:
            if self.active_tv_model:
                log.info(f"Active TV model '{self.active_tv_model}' cleared. Reason: {reason}. Recognized models: {self.recognized_tv_models}")
            self.active_tv_model = None
            self.current_problem_description = "" # Clear problem if active model is cleared
            self.current_model_general_images = None
            self.in_troubleshooting_flow = False # Also reset troubleshooting flow if active model is cleared
            return

        model_upper = model_name.strip().upper()
        if model_upper not in self.recognized_tv_models: # Ensure it's added to recognized list first
            self.add_recognized_model(model_upper) 

        if self.active_tv_model != model_upper:
            log.info(f"Active TV model changed from '{old_active}' to '{model_upper}'. Reason: {reason}. Recognized: {self.recognized_tv_models}")
            self.active_tv_model = model_upper
            self.current_problem_description = "" # Reset problem description when active model changes
            self.current_model_general_images = None # Reset images for new active model
            self.in_troubleshooting_flow = False # Reset flow state for the new model context
        elif not self.active_tv_model and model_upper: # Setting for the first time (already handled by add_recognized_model if it was first)
            log.info(f"Active TV model set to '{model_upper}'. Reason: {reason}. Recognized: {self.recognized_tv_models}")
            self.active_tv_model = model_upper
        # If model_upper is already active_tv_model, no significant change in active_tv_model itself.

    def start_troubleshooting_flow(self, problem_description: str, for_model: str | None = None):
        """Initiates or updates a troubleshooting flow for the active (or specified) model."""
        target_model_name = for_model or self.active_tv_model
        if not target_model_name:
            log.warning("Cannot start troubleshooting flow without an active or specified TV model.")
            return # Or raise an error, or set an expectation to get model

        target_model_upper = target_model_name.strip().upper()
        if self.active_tv_model != target_model_upper:
            self.set_active_model(target_model_upper, "troubleshooting_start")
            if not self.active_tv_model: # If set_active_model failed (should be rare)
                log.error(f"Failed to set active model to {target_model_upper} during troubleshooting start.")
                return


        self.in_troubleshooting_flow = True
        self.current_problem_description = problem_description.strip()
        self.clear_expectations() # Clear previous turn-specific expectations before starting new flow logic
        log.info(f"Troubleshooting flow started/updated for model '{self.active_tv_model}'. Problem: '{self.current_problem_description[:50]}...'")


    def end_session(self, reason="user_ended"):
        log.info(f"SESSION ENDED. Reason: {reason}. Was: ActiveModel='{self.active_tv_model}', Recognized={self.recognized_tv_models}, Problem='{self.current_problem_description[:30]}...'")
        self.recognized_tv_models = []
        self.active_tv_model = None
        self.current_model_general_images = None
        self.in_troubleshooting_flow = False
        self.current_problem_description = ""
        self.clear_pdf_context()
        self.clear_expectations()
        self.clear_lc_memory()

    def clear_active_problem(self):
        if self.current_problem_description:
            log.info(f"Problem '{self.current_problem_description[:50]}' cleared for model '{self.active_tv_model or 'N/A'}'. Model context may remain.")
        self.current_problem_description = ""
        # self.in_troubleshooting_flow might remain true if active_tv_model is still set
        # but we are no longer focused on a *specific* problem.
        self.clear_expectations(clear_model_expectation=False) # Keep model expectation if any

    def set_expectation(self, expectation_type: str, details: Dict[str, Any] | None = None, problem_context_for_model_request: str | None = None):
        self.clear_expectations() # Clear any old expectation first
        self.expecting_confirmation_for = {"type": expectation_type, "details": details or {}}
        if problem_context_for_model_request: # e.g. if asking for model for a specific problem
            self.expecting_model_for_problem = problem_context_for_model_request
        log.info(f"SETTING EXPECTATION: Type='{expectation_type}', Details='{str(details)[:100]}...', ProblemContextForModel='{str(problem_context_for_model_request)[:50]}'")

    def get_expectation(self) -> Dict[str, Any] | None:
        return self.expecting_confirmation_for

    def clear_expectations(self, clear_model_expectation_context=True):
        if self.expecting_confirmation_for:
            log.debug(f"Clearing expectation: {self.expecting_confirmation_for}")
            self.expecting_confirmation_for = None
        if clear_model_expectation_context and self.expecting_model_for_problem:
            log.debug(f"Clearing expecting_model_for_problem context: {self.expecting_model_for_problem}")
            self.expecting_model_for_problem = None

    def set_pdf_context(self, text: str, filename: str):
        self.pdf_context_text = text
        self.pdf_context_source_filename = filename
        log.info(f"PDF context set from '{filename}' (text length: {len(text)} chars).")
        # Optionally add to Langchain memory if LLM should always "remember" PDF loading event
        # self.memory.chat_memory.add_message(SystemMessage(content=f"[System Note: PDF document '{filename}' loaded.]"))

    def clear_pdf_context(self):
        cleared_from = self.pdf_context_source_filename
        if self.pdf_context_text or self.pdf_context_source_filename:
            log.info(f"PDF context (was from: '{cleared_from or 'N/A'}') cleared.")
            # if cleared_from:
            #    self.memory.chat_memory.add_message(SystemMessage(content=f"[System Note: PDF document '{cleared_from}' cleared.]"))
        self.pdf_context_text = None
        self.pdf_context_source_filename = None

    def get_pdf_context_for_llm(self, max_chars: int = 3500) -> str:
        if not self.pdf_context_text:
            return ""
        
        context = self.pdf_context_text
        filename_display = self.pdf_context_source_filename or "the loaded document"
        truncation_indicator = f"\n[...Content from '{filename_display}' truncated...]"
        
        if len(context) > max_chars:
            effective_max_chars = max_chars - len(truncation_indicator)
            if effective_max_chars <= 0: 
                context = context[:max_chars]
            else:
                context = context[:effective_max_chars] + truncation_indicator
            log.warning(f"PDF context for LLM (from '{filename_display}') was truncated to approx {max_chars} chars.")
        
        return f"\n\n--- Context from PDF: '{filename_display}' ---\n{context}\n--- End of PDF Context ---"

    def get_current_session_details(self) -> dict:
        """Returns a dictionary with current state details for logging or debugging."""
        return {
            "active_tv_model": self.active_tv_model,
            "recognized_tv_models": self.recognized_tv_models,
            "current_problem": self.current_problem_description,
            "in_troubleshooting_flow": self.in_troubleshooting_flow,
            "lang_code": self.current_language,
            "lang_name": self.current_language_name,
            "dialect_info": self.last_detected_dialect_info,
            "current_expectation": self.expecting_confirmation_for,
            "problem_awaiting_model_context": self.expecting_model_for_problem,
            "pdf_context_active": bool(self.pdf_context_text),
            "pdf_filename": self.pdf_context_source_filename,
            "general_images_for_active_model": bool(self.current_model_general_images),
            "lc_memory_message_count": len(self.memory.chat_memory.messages) if self.memory else 0
        }