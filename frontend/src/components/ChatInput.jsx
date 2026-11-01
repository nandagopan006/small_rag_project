import React, { useState, useRef, useEffect } from "react";
import { ArrowUp } from "lucide-react";

/**
 * Premium chat composer at the bottom of the conversation area.
 *
 * @param {Object} props
 * @param {Function} props.onSend - Submit handler receiving question string.
 * @param {boolean} props.isLoading - Whether a request is currently processing.
 */
export default function ChatInput({ onSend, isLoading }) {
  const [question, setQuestion] = useState("");
  const textareaRef = useRef(null);

  // Auto-resize textarea height as text grows
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        140
      )}px`;
    }
  }, [question]);

  function handleChange(event) {
    setQuestion(event.target.value);
  }

  function handleKeyDown(event) {
    // Enter without Shift submits question
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  }

  function handleSubmit() {
    const cleanText = question.trim();
    if (!cleanText || isLoading) {
      return;
    }

    onSend(cleanText);
    setQuestion("");

    // Reset textarea height
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  }

  const isSubmitDisabled = !question.trim() || isLoading;

  return (
    <div className="chat-composer-box">
      <textarea
        ref={textareaRef}
        className="composer-textarea"
        rows={1}
        value={question}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        placeholder="Ask anything about your college policies (e.g. attendance, exams, leave)..."
        aria-label="Ask a question about college policies"
        disabled={isLoading}
      />
      <button
        type="button"
        className="send-button"
        onClick={handleSubmit}
        disabled={isSubmitDisabled}
        aria-label="Send question"
        title={isLoading ? "Processing question..." : "Send question (Enter)"}
      >
        <ArrowUp size={20} strokeWidth={2.5} />
      </button>
    </div>
  );
}
