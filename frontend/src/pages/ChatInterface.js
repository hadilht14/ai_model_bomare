// frontend/src/pages/ChatInterface.js
import React, { useState, useEffect, useCallback, useRef } from 'react';
import axios from 'axios';
import '../ChatInterface.css';
import Sidebar from '../components/Sidebar.js';
import ChatArea from '../components/ChatArea.js';
import InputBar from '../components/InputBar.js';
import TopBar from '../components/TopBar.js';
import { FiLoader } from 'react-icons/fi';

const API_BASE_URL = 'http://localhost:5000/api';
const COMPANY_LOGO_PATH_FRAGMENT = "/bomare_logo.png";
const COMPANY_NAME = "Bomare Company";

const ChatInterface = () => {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [chatHistory, setChatHistory] = useState([]);
  const [selectedMode, setSelectedMode] = useState('Chatbot');
  const [currentLanguage, setCurrentLanguage] = useState('en');
  const [isLoadingResponse, setIsLoadingResponse] = useState(false);
  const [isChatModuleLoading, setIsChatModuleLoading] = useState(true);

  const companyLogoPath = process.env.PUBLIC_URL + COMPANY_LOGO_PATH_FRAGMENT;
  const initAttempted = useRef(false);

  const isChatActive = (messages.length > 0 && messages.some(m => m.sender === 'user' || m.sender === 'bot')) || isLoadingResponse;

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);

  const fetchChatHistory = useCallback(async () => {
    console.log("FETCH_HISTORY: Called");
    try {
      const response = await axios.get(`${API_BASE_URL}/chat_history`);
      const fetchedHistory = response.data || [];
      console.log("FETCH_HISTORY: Success - Data:", fetchedHistory);
      setChatHistory(fetchedHistory);
      return fetchedHistory;
    } catch (error) {
      console.error("FETCH_HISTORY: Error fetching chat history:", error.response?.data || error.message);
      setChatHistory([]);
      return [];
    }
  }, []);

  const startNewChat = useCallback(async (isInitializing = false) => {
    console.log("START_NEW_CHAT: Called, isInitializing:", isInitializing);
    if (!isInitializing && isLoadingResponse) {
      console.log("START_NEW_CHAT: Aborting, already loading response for non-init call.");
      return null;
    }

    if (!isInitializing) setIsLoadingResponse(true);

    let newSessionId = null;
    try {
      const response = await axios.post(`${API_BASE_URL}/new_chat`);
      console.log("START_NEW_CHAT: API Response:", response.data);
      if (response.data && response.data.sessionId) {
        newSessionId = response.data.sessionId;
        setCurrentChatId(newSessionId);
        setMessages(response.data.messages || []);
        setCurrentLanguage(response.data.languageCode || 'en');
        if (!isInitializing || chatHistory.length === 0) {
            await fetchChatHistory();
        }
      } else {
        console.error("START_NEW_CHAT: Error - No sessionId in response", response.data);
        setMessages([{ sender: 'system', text: 'Error: Could not start a new chat session (missing ID).', timestamp: new Date().toISOString() }]);
        setCurrentChatId(null);
      }
    } catch (error) {
      console.error("START_NEW_CHAT: Axios catch error:", error.response?.data || error.message);
      setMessages([{ sender: 'system', text: 'Error starting new chat. Please try again.', timestamp: new Date().toISOString() }]);
      setCurrentChatId(null);
    } finally {
      if (!isInitializing) setIsLoadingResponse(false);
    }
    console.log("START_NEW_CHAT: Finished. Returned Session ID (if any):", newSessionId);
    return newSessionId;
  }, [fetchChatHistory, isLoadingResponse, chatHistory.length]);

  const loadChatSession = useCallback(async (sessionId, isInitializing = false) => {
    console.log("LOAD_CHAT_SESSION: Called for session:", sessionId, "isInitializing:", isInitializing);
    if (!isInitializing && isLoadingResponse) {
        console.log("LOAD_CHAT_SESSION: Aborting, already loading response for non-init call.");
        return null;
    }
    if (!isInitializing) setIsLoadingResponse(true);

    let loadedSessionId = null;
    try {
      const response = await axios.get(`${API_BASE_URL}/chat_session/${sessionId}`);
      console.log("LOAD_CHAT_SESSION: API Response:", response.data);
      if (response.data && response.data.sessionId) {
        loadedSessionId = response.data.sessionId;
        setCurrentChatId(loadedSessionId);
        setMessages(response.data.messages || []);
        if (response.data.languageCode && response.data.languageCode !== currentLanguage) {
            setCurrentLanguage(response.data.languageCode);
            console.log("LOAD_CHAT_SESSION: Language updated to:", response.data.languageCode);
        }
      } else {
        console.error("LOAD_CHAT_SESSION: Error - No sessionId in response or invalid data", response.data);
        loadedSessionId = await startNewChat(isInitializing);
      }
    } catch (error) {
      console.error(`LOAD_CHAT_SESSION: Error loading session ${sessionId} (axios catch):`, error.response?.data || error.message);
      if (error.response && error.response.status === 404) {
        setMessages([{ sender: 'system', text: 'Chat session not found. Starting a new chat.', timestamp: new Date().toISOString() }]);
        loadedSessionId = await startNewChat(isInitializing);
      } else {
        setMessages([{ sender: 'system', text: 'Error loading chat session. Please try again.', timestamp: new Date().toISOString() }]);
        setCurrentChatId(null);
      }
    } finally {
      if (!isInitializing) setIsLoadingResponse(false);
    }
    console.log("LOAD_CHAT_SESSION: Finished. Returned Session ID (if any):", loadedSessionId);
    return loadedSessionId;
  }, [startNewChat, isLoadingResponse, currentLanguage]);

  useEffect(() => {
    const initializeChatModule = async () => {
      if (initAttempted.current) {
        console.log("INIT_MODULE: Skipping redundant initialization (already attempted).");
        setIsChatModuleLoading(false);
        return;
      }
      initAttempted.current = true;
      console.log("INIT_MODULE: Starting automatic session initialization...");
      setIsChatModuleLoading(true);

      let sessionIdToUseInEffect = null;
      try {
        const history = await fetchChatHistory();
        if (history && history.length > 0 && history[0]?.id) {
          console.log("INIT_MODULE: History found, attempting to load session:", history[0].id);
          sessionIdToUseInEffect = await loadChatSession(history[0].id, true);
        }

        if (!sessionIdToUseInEffect) {
          console.log("INIT_MODULE: No session from history or load failed. Starting a new chat...");
          sessionIdToUseInEffect = await startNewChat(true);
        }

        if (!sessionIdToUseInEffect) {
            console.error("INIT_MODULE: CRITICAL - Failed to establish any session ID after all attempts.");
            // setCurrentChatId is already null or set by failed attempts
        } else {
            console.log("INIT_MODULE: Session established/loaded with ID:", sessionIdToUseInEffect);
            // State (currentChatId) was set inside loadChatSession or startNewChat
        }

      } catch (error) {
        console.error("INIT_MODULE: Critical error during initialization sequence:", error);
        if (!currentChatId && !sessionIdToUseInEffect) {
             console.warn("INIT_MODULE: Critical error led to no session. Forcing one last new chat attempt.");
             await startNewChat(true); // This will attempt to set currentChatId state
        }
      } finally {
        console.log("INIT_MODULE: Initialization sequence finished.");
        setIsChatModuleLoading(false);
      }
    };

    initializeChatModule();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Keep dependencies minimal for one-time effect

  const handleSendMessage = async (content, type = 'text') => {
    let chatIdForThisMessage = currentChatId; // Get current sessionId from state

    console.log(`SEND_MESSAGE: Initiated. currentChatId from state: ${chatIdForThisMessage}. Content type: ${type}. Content: ${type === 'file' ? content.name : String(content).substring(0,50)}`);

    if (!chatIdForThisMessage) {
      console.warn("SEND_MESSAGE: No active chat ID from state. Attempting to start a new chat before sending.");
      setIsLoadingResponse(true);
      const newId = await startNewChat(false);
      if (!newId) {
        alert("Critical Error: No active chat session and could not start a new one. Please refresh the page.");
        console.error("SEND_MESSAGE: Failed to start a new chat when currentChatId was null.");
        setIsLoadingResponse(false);
        return;
      }
      chatIdForThisMessage = newId; // CRUCIAL: Use the ID returned from startNewChat for *this* send
      console.log("SEND_MESSAGE: New chat started, using ID for this message:", chatIdForThisMessage);
    }

    const userMessageText = type === 'file' && content instanceof File ? `File: ${content.name}` : content;
    const userMessage = { sender: 'user', text: userMessageText, timestamp: new Date().toISOString(), type, originalContent: content };

    let updatedMessages = [...messages];
    updatedMessages = updatedMessages.filter(m => m.sender !== 'system' || !m.text.startsWith('Context:'));
    if (messages.filter(m => m.sender === 'user' || m.sender === 'bot').length === 0) {
      updatedMessages.push({ sender: 'system', text: `Context: Mode - ${selectedMode}, Language - ${currentLanguage.toUpperCase()}.`, timestamp: new Date().toISOString() });
    }
    updatedMessages.push(userMessage);
    setMessages(updatedMessages);
    setIsLoadingResponse(true);

    let dataForAxios;
    const endpoint = `${API_BASE_URL}/chat`;
    let axiosConfig = {};

    if (!chatIdForThisMessage) { // Should be extremely rare to hit this now
        console.error("SEND_MESSAGE: CRITICAL FAILURE - chatIdForThisMessage is STILL null just before API call. Aborting.");
        setMessages(prev => [...prev, { sender: 'bot', text: "Error: Session ID became null unexpectedly. Cannot send message.", timestamp: new Date().toISOString() }]);
        setIsLoadingResponse(false);
        return;
    }

    const commonPayload = {
        sessionId: chatIdForThisMessage,
        language: currentLanguage,
        mode: selectedMode
    };

    if (type === 'text') {
      const payloadObject = {
        ...commonPayload,
        message: content,
      };
      dataForAxios = JSON.stringify(payloadObject); // Explicitly stringify
      axiosConfig = {
        headers: {
          'Content-Type': 'application/json',
        },
      };
      console.log("SEND_MESSAGE: Preparing TEXT request. Stringified Payload being sent:", dataForAxios);
    } else if (type === 'file' && content instanceof File) {
      const formData = new FormData();
      formData.append('file', content);
      // Add common payload fields to FormData
      Object.keys(commonPayload).forEach(key => formData.append(key, commonPayload[key]));
      // If you need to send a message string along with the file from the input bar
      // const messageAssociatedWithFile = ... (get this from your input bar state)
      // formData.append('message', messageAssociatedWithFile || `Uploaded: ${content.name}`);
      formData.append('message', userMessageText); // Or use the filename as the message

      dataForAxios = formData;
      axiosConfig = { headers: {} }; // Axios handles Content-Type for FormData
      console.log("SEND_MESSAGE: Preparing FILE request. FormData being sent (see Network tab for details). SessionId in FormData:", chatIdForThisMessage);
    } else {
      console.warn("SEND_MESSAGE: Invalid message type or content:", content, type);
      setIsLoadingResponse(false);
      return;
    }

    try {
      // The console.log right before axios.post is already good
      const response = await axios.post(endpoint, dataForAxios, axiosConfig);
      console.log("SEND_MESSAGE: API Response Data:", response.data);
      if (response.data && typeof response.data.reply === 'string') {
        const botMessage = { sender: 'bot', text: response.data.reply, timestamp: new Date().toISOString() };
        setMessages(prev => [...prev, botMessage]);
        if (response.data.languageCode && response.data.languageCode !== currentLanguage) {
            setCurrentLanguage(response.data.languageCode);
            console.log("SEND_MESSAGE: Language updated by backend to:", response.data.languageCode);
        }
      } else {
        console.error("SEND_MESSAGE: Invalid API response structure from backend. 'reply' field missing or not a string.", response.data);
        setMessages(prev => [...prev, { sender: 'bot', text: "Received an unexpected or malformed response from the server.", timestamp: new Date().toISOString() }]);
      }

      const currentHistEntry = chatHistory.find(h => h.id === chatIdForThisMessage);
      if (currentHistEntry && (!currentHistEntry.title || currentHistEntry.title.startsWith("New Chat") || currentHistEntry.title.startsWith("Chat ") || currentHistEntry.title.startsWith("Session"))) {
        // Only fetch history if there was at least one user message to potentially update the title from
        if (updatedMessages.some(m => m.sender === 'user')) {
             await fetchChatHistory();
        }
      }
    } catch (error) {
      console.error("SEND_MESSAGE: API Error caught:", error);
      const errorReply = "Sorry, I couldn't process your message at this time. Please try again.";
      let backendErrorMsg = errorReply;
      if (error.response) {
        console.error("SEND_MESSAGE: Backend Error Response Data:", error.response.data);
        console.error("SEND_MESSAGE: Backend Error Response Status:", error.response.status);
        backendErrorMsg = error.response.data?.reply || error.response.data?.error || errorReply;
      } else if (error.request) {
        console.error('SEND_MESSAGE: No response received from server:', error.request);
        backendErrorMsg = "No response from the server. Please check your internet connection and try again.";
      } else {
        console.error('SEND_MESSAGE: Error setting up request:', error.message);
        backendErrorMsg = "An error occurred while sending your message. Please try again.";
      }
      setMessages(prev => [...prev, { sender: 'bot', text: backendErrorMsg, timestamp: new Date().toISOString() }]);
    } finally {
      setIsLoadingResponse(false);
    }
  };

  const handleModeChange = (newMode) => {
    setSelectedMode(newMode);
    if (isChatActive && messages.some(m => m.sender !== 'system')) {
      setMessages(prev => [...prev, { sender: 'system', text: `Switched to ${newMode} mode.`, timestamp: new Date().toISOString() }]);
    }
  };
  const handleLanguageChange = (lang) => {
    setCurrentLanguage(lang);
    if (isChatActive && messages.some(m => m.sender !== 'system')) {
      setMessages(prev => [...prev, { sender: 'system', text: `Language changed to ${lang.toUpperCase()}.`, timestamp: new Date().toISOString() }]);
    }
  };

  if (isChatModuleLoading) {
    return (
      <div className="chat-module-container chat-module-loading">
        <FiLoader size={48} style={{ animation: 'spin 1s linear infinite', marginBottom: '16px' }} />
        <span>Bomare Assistant is loading...</span>
      </div>
    );
  }

  return (
    <div className="chat-module-container">
      <Sidebar
        isOpen={isSidebarOpen}
        toggleSidebar={toggleSidebar}
        onNewChat={() => startNewChat(false)}
        chatHistory={chatHistory}
        isHistoryLoading={isChatModuleLoading}
        onSelectChat={(sessionId) => loadChatSession(sessionId, false)}
        currentChatId={currentChatId}
        currentLanguage={currentLanguage}
        onLanguageChange={handleLanguageChange}
      />
      <div className={`main-content ${isSidebarOpen ? 'sidebar-open' : ''} ${isChatActive || currentChatId ? 'input-bottom' : 'input-centered'}`}>
        <TopBar
          selectedMode={selectedMode}
          onModeChange={handleModeChange}
          onToggleSidebar={toggleSidebar}
          companyLogoPath={companyLogoPath}
          companyName={COMPANY_NAME}
        />
        <div className="chat-interaction-area">
          {!currentChatId && messages.length === 0 && !isChatModuleLoading ? (
            <div className="chat-area welcome-message">
              <h2>Hello, I'm Bomare Assistant</h2>
              <p>Your guide for TV troubleshooting and more. How can I assist you today?</p>
            </div>
          ) : (
            <ChatArea messages={messages} isLoading={isLoadingResponse} currentLanguage={currentLanguage} />
          )}
          <InputBar onSendMessage={handleSendMessage} isLoading={isLoadingResponse || (!currentChatId && isChatModuleLoading)} />
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;
