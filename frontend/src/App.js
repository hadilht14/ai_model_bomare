import React from 'react';
import ChatInterface from './pages/ChatInterface';
import './index.css'; // Global styles/vars for the chat module (theme)

function App() {
  // This App component is now just a simple host for the ChatInterface.
  // When you integrate ChatInterface into another project, that project's App.js (or relevant component)
  // will import and render ChatInterface.
  // No complex routing or state management here.
  return (
    // Optionally, you can wrap ChatInterface in a div if you want to apply
    // some root styling specifically for the standalone running context.
    // For now, ChatInterface has its own root .chat-module-container div.
    <ChatInterface />
  );
}

export default App;