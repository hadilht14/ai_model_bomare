import React, { useState, useRef } from 'react';
import { FiSend, FiMic, FiPaperclip } from 'react-icons/fi';

const InputBar = ({ onSendMessage, isLoading }) => {
  const [inputValue, setInputValue] = useState('');
  const fileInputRef = useRef(null);

  const handleSend = () => {
    if (inputValue.trim()) {
      onSendMessage(inputValue, 'text');
      setInputValue('');
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      onSendMessage(file, 'file');
      if (event.target) { // Add a check for event.target
          event.target.value = null; // Reset file input
      }
    }
  };

  const triggerFileInput = () => {
    if (!isLoading && fileInputRef.current) fileInputRef.current.click();
  };

  const handleAudioInput = () => {
    if (!isLoading) alert("Audio input functionality to be implemented.");
  };

  return (
    <div className="input-bar-container">
      <div className="input-bar-wrapper">
        <input
            type="file"
            ref={fileInputRef}
            style={{ display: 'none' }}
            onChange={handleFileChange}
            disabled={isLoading}
            accept="image/*,application/pdf,.doc,.docx,.txt,audio/*,video/*"
        />
        <button onClick={triggerFileInput} title="Attach file" className="input-bar-button" disabled={isLoading}>
            <FiPaperclip />
        </button>
        <button onClick={handleAudioInput} title="Record audio" className="input-bar-button" disabled={isLoading}>
            <FiMic />
        </button>

        <input
          type="text"
          placeholder="Ask anything..."
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isLoading}
        />
        <button onClick={handleSend} title="Send message" className="input-bar-button send-button" disabled={isLoading || !inputValue.trim()}>
          <FiSend />
        </button>
      </div>
    </div>
  );
};

export default InputBar;