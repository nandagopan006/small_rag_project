import React, { useState, useEffect, useRef } from "react";
import Header from "./components/Header";
import WelcomeScreen from "./components/WelcomeScreen";
import ChatMessage from "./components/ChatMessage";
import ChatInput from "./components/ChatInput";
import LoadingIndicator from "./components/LoadingIndicator";
import { askPolicyQuestion, fetchDocuments } from "./services/api";
import { AlertCircle, X } from "lucide-react";

export default function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);
  const [handbookInfo, setHandbookInfo] = useState(null);

  const threadEndRef = useRef(null);

  // Fetch handbook metadata on mount
  useEffect(() => {
    async function loadHandbookMetadata() {
      try {
        const docs = await fetchDocuments();
        if (docs && docs.length > 0) {
          const activeDoc = docs.find((d) => d.status === "processed") || docs[0];
          setHandbookInfo(activeDoc);
        }
      } catch {
        // Non-blocking fallback
      }
    }

    loadHandbookMetadata();
  }, []);

  // Auto-scroll to latest message in active conversation
  useEffect(() => {
    if (messages.length > 0 || isLoading) {
      threadEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, isLoading]);

  /**
   * Send a question to the Django RAG backend.
   *
   * @param {string} questionText - User's query.
   */
  async function handleSendMessage(questionText) {
    if (!questionText || isLoading) {
      return;
    }

    const timestamp = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });

    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: questionText,
      timestamp,
    };

    setMessages((prev) => [...prev, userMessage]);
    setErrorMessage(null);
    setIsLoading(true);

    try {
      const response = await askPolicyQuestion(questionText);

      const assistantMessage = {
        id: `assistant-${Date.now()}`,
        role: "assistant",
        content: response.answer,
        sources: response.sources || [],
        timestamp: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorText =
        err?.message ||
        "An unexpected error occurred while communicating with the college policy assistant.";
      setErrorMessage(errorText);

      const errorAssistantMessage = {
        id: `error-${Date.now()}`,
        role: "assistant",
        content:
          "I was unable to retrieve a policy answer due to a connection issue with the backend. Please verify the server is running and try again.",
        sources: [],
        timestamp: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      };

      setMessages((prev) => [...prev, errorAssistantMessage]);
    } finally {
      setIsLoading(false);
    }
  }

  /**
   * Reset conversation state back to single-screen welcome.
   */
  function handleClearChat() {
    setMessages([]);
    setErrorMessage(null);
  }

  const hasMessages = messages.length > 0;

  return (
    <div className={`app-root ${hasMessages ? "state-chat" : "state-welcome"}`}>
      {/* Minimal Navigation Header */}
      <Header
        messages={messages}
        onClearChat={handleClearChat}
        handbookInfo={handbookInfo}
      />

      {/* Main Content Area */}
      <main className="app-main">
        {/* State A: Minimal Single-Screen Welcome (No Scrolling) */}
        {!hasMessages && (
          <WelcomeScreen
            onSend={handleSendMessage}
            isLoading={isLoading}
            handbookInfo={handbookInfo}
          />
        )}

        {/* State B: Active Conversation Thread */}
        {hasMessages && (
          <div className="content-container">
            <div className="chat-thread-container" role="log" aria-live="polite">
              {errorMessage && (
                <div className="error-banner" role="alert">
                  <div className="error-banner-content">
                    <AlertCircle size={16} />
                    <span>{errorMessage}</span>
                  </div>
                  <button
                    type="button"
                    className="error-dismiss-btn"
                    onClick={() => setErrorMessage(null)}
                    aria-label="Dismiss error notification"
                  >
                    <X size={16} />
                  </button>
                </div>
              )}

              {messages.map((msg) => (
                <ChatMessage key={msg.id} message={msg} />
              ))}

              {isLoading && <LoadingIndicator />}

              <div ref={threadEndRef} />
            </div>
          </div>
        )}
      </main>

      {/* Bottom Composer Dock (Only visible when conversation is active) */}
      {hasMessages && (
        <div className="composer-dock">
          <div className="composer-inner">
            <ChatInput onSend={handleSendMessage} isLoading={isLoading} />
            <div className="composer-caption">
              Answers are grounded in the Northstar College Policy Handbook.
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
