import React, { useState, useRef, useEffect } from 'react';
import {
  FiPlusSquare, FiSettings, FiGlobe, FiLoader, FiMessageSquare, FiArrowLeft
} from 'react-icons/fi';

const Sidebar = ({
  isOpen,
  toggleSidebar,
  onNewChat,
  chatHistory,
  isHistoryLoading,
  onSelectChat,
  currentChatId,
  currentLanguage,
  onLanguageChange,
}) => {
  const [isLangDropdownOpen, setIsLangDropdownOpen] = useState(false);
  const langDropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (langDropdownRef.current && !langDropdownRef.current.contains(event.target)) {
        setIsLangDropdownOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleLangSelect = (lang) => {
    onLanguageChange(lang);
    setIsLangDropdownOpen(false);
  };

  return (
    <aside className={`sidebar ${isOpen ? '' : 'closed'}`}>
      <div className="sidebar-header">
        <button onClick={toggleSidebar} className="sidebar-action-button close-sidebar-button" aria-label="Close menu">
          <FiArrowLeft />
        </button>
        <button onClick={onNewChat} className="sidebar-action-button new-chat-icon-button" aria-label="New Chat">
          <FiPlusSquare />
        </button>
      </div>

      <div className="sidebar-content">
        {isHistoryLoading ? (
          <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-dark)' }}>
            <FiLoader className="spin" size={24} style={{ marginRight: '8px', animation: 'spin 1s linear infinite' }} />
            Loading...
          </div>
        ) : chatHistory.length > 0 ? (
          <ul>
            {chatHistory.map((chat) => (
              <li
                key={chat.id}
                className={`history-item ${chat.id === currentChatId ? 'active' : ''}`}
                onClick={() => onSelectChat(chat.id)}
                title={chat.title}
              >
                <FiMessageSquare />
                <span>
                  {chat.title || `Session ${chat.id.slice(-5)}`}
                </span>
              </li>
            ))}
          </ul>
        ) : (
          <p style={{padding: '16px', fontSize: 'var(--fs-sm)', color: 'var(--text-dark)', textAlign: 'center' }}>
              No previous chats. <br/>Start a new conversation!
          </p>
        )}
      </div>

      <div className="sidebar-footer">
        <div className="language-control" ref={langDropdownRef}>
          <button onClick={() => setIsLangDropdownOpen(prev => !prev)} aria-label="Change language" className="footer-icon-button">
              <FiGlobe />
          </button>
          {isLangDropdownOpen && (
              <div className="language-dropdown">
              <button onClick={() => handleLangSelect('en')} className={currentLanguage === 'en' ? 'active' : ''}>English</button>
              <button onClick={() => handleLangSelect('fr')} className={currentLanguage === 'fr' ? 'active' : ''}>Français</button>
              <button onClick={() => handleLangSelect('ar')} className={currentLanguage === 'ar' ? 'active' : ''}>العربية</button>
              </div>
          )}
        </div>
        <button className="settings-button footer-icon-button" aria-label="Settings" onClick={() => alert('Settings page not implemented yet!')}>
            <FiSettings />
        </button>
      </div>
    </aside>
  );
};

export default Sidebar;