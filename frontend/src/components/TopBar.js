import React from 'react';
import { FiMenu } from 'react-icons/fi';

const TopBar = ({
  selectedMode,
  onModeChange,
  onToggleSidebar,
  companyLogoPath,
  companyName
}) => {
  return (
    <div className="top-bar">
      <button
        onClick={onToggleSidebar}
        className="hamburger-button"
        aria-label="Toggle menu"
      >
        <FiMenu />
      </button>

      <div className="top-bar-logo-centered">
        <img src={companyLogoPath} alt={`${companyName} Logo`} />
      </div>

      <div className="mode-switcher-container top-bar-right-item">
        <button
          className={selectedMode === 'Chatbot' ? 'active' : ''}
          onClick={() => onModeChange('Chatbot')}
          aria-pressed={selectedMode === 'Chatbot'}
        >
          Chatbot
        </button>
        <button
          className={selectedMode === 'Interactive Assistant' ? 'active' : ''}
          onClick={() => onModeChange('Interactive Assistant')}
          aria-pressed={selectedMode === 'Interactive Assistant'}
        >
          Assistant
        </button>
      </div>
    </div>
  );
};

export default TopBar;