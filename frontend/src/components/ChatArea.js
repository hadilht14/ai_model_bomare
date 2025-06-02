// frontend/src/components/ChatArea.js
import React, { useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm'; // For tables, strikethrough, etc.

// Pass currentLanguage from ChatInterface.js as a prop
const ChatArea = ({ messages, isLoading, currentLanguage }) => {
  const endOfMessagesRef = useRef(null);

  useEffect(() => {
    endOfMessagesRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Custom renderer for images
  const renderers = {
    img: ({ src, alt, title }) => {
      let resolvedSrc = src;

      // Current understanding: backend sends src like "troubleshooting/image.png"
      // process.env.PUBLIC_URL is "" during development and for root production builds.
      // So, we need to ensure the path becomes "/troubleshooting/image.png"
      // which is relative to the public folder.

      if (src) {
        if (src.startsWith('http://') || src.startsWith('https://')) {
          // It's an absolute URL, use it as is.
          resolvedSrc = src;
        } else if (src.startsWith('/')) {
          // It already starts with a slash, assume it's correct relative to PUBLIC_URL
          // (e.g., if backend already prepended PUBLIC_URL or it's intentionally absolute)
          resolvedSrc = `${process.env.PUBLIC_URL}${src}`;
        } else {
          // It's a relative path like "troubleshooting/image.png".
          // Prepend PUBLIC_URL and a slash.
          resolvedSrc = `${process.env.PUBLIC_URL}/${src}`;
        }

        // Normalize multiple slashes that might occur (e.g., if PUBLIC_URL is "" and src starts with /)
        // but be careful not to mess up "http://" or "https://".
        if (resolvedSrc.startsWith('http://')) {
            resolvedSrc = 'http://' + resolvedSrc.substring(7).replace(/\/\//g, '/');
        } else if (resolvedSrc.startsWith('https://')) {
            resolvedSrc = 'https://' + resolvedSrc.substring(8).replace(/\/\//g, '/');
        } else {
            // For relative paths or paths starting with a single slash after PUBLIC_URL
            resolvedSrc = resolvedSrc.replace(/\/\//g, '/');
        }
      }
      
      console.log(
        "ChatArea Rendering image -> original src from markdown:", src, 
        "| process.env.PUBLIC_URL:", process.env.PUBLIC_URL,
        "| Attempted resolvedSrc for <img> tag:", resolvedSrc, 
        "| Alt text:", alt
      );
      
      return (
        <img 
          src={resolvedSrc} 
          alt={alt || 'Chatbot image'} // Provide a default alt text
          title={title} 
          style={{ 
            maxWidth: '100%', 
            height: 'auto', 
            borderRadius: '8px', 
            marginTop: '10px', 
            marginBottom: '10px', 
            display: 'block',
            border: '1px solid #444' // Added a light border for visibility during debugging
          }} 
          onError={(e) => { 
            e.target.style.display='none'; // Hide broken image icon
            const errorDiv = document.createElement('div');
            errorDiv.textContent = `Error: Image not found. Tried: ${resolvedSrc}. Alt: ${alt || 'N/A'}`;
            errorDiv.style.color = 'red';
            errorDiv.style.fontSize = 'small';
            errorDiv.style.padding = '5px';
            errorDiv.style.border = '1px dashed red';
            e.target.parentNode.insertBefore(errorDiv, e.target.nextSibling);
            console.error("ChatArea Image Error: Failed to load image.", {
                originalSrc: src,
                resolvedSrc: resolvedSrc,
                alt: alt,
                element: e.target
            });
          }}
        />
      );
    }
  };

  return (
    <div className="chat-area">
      {messages.map((msg, index) => {
        let messageLang = 'en'; 
        if (msg.sender === 'bot') {
          messageLang = currentLanguage || 'en'; 
        }
        const isRtl = messageLang === 'ar';

        return (
          <div
            key={`${msg.sender}-${msg.timestamp}-${index}-${msg.text ? msg.text.slice(0,10) : 'file'}`}
            className={`message-wrapper ${msg.sender} ${isRtl ? 'message-rtl' : 'message-ltr'}`}
          >
            <div className="message">
              {msg.type === 'file' && msg.sender === 'user' ? (
                `File: ${msg.text || (msg.originalContent && msg.originalContent.name) || 'Uploaded File'}`
              ) : msg.sender === 'bot' ? (
                <ReactMarkdown
                  components={renderers} 
                  remarkPlugins={[remarkGfm]} 
                >
                  {msg.text}
                </ReactMarkdown>
              ) : (
                msg.text
              )}
            </div>
          </div>
        );
      })}
      {isLoading && (
        <div className="loading-indicator">
          <div className="loading-indicator-dots">
            <span></span><span></span><span></span>
          </div>
        </div>
      )}
      <div ref={endOfMessagesRef} />
    </div>
  );
};

export default ChatArea;