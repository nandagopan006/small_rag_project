import React from "react";
import { Compass, RotateCcw } from "lucide-react";

/**
 * Clean navigation header matching the minimal editorial aesthetic.
 *
 * @param {Object} props
 * @param {Array} props.messages - Active conversation messages.
 * @param {Function} props.onClearChat - Callback to reset conversation.
 * @param {Object} [props.handbookInfo] - Metadata about the loaded handbook.
 */
export default function Header({ messages, onClearChat, handbookInfo }) {
  const hasMessages = messages && messages.length > 0;

  return (
    <header className="app-header">
      <div className="header-container">
        {/* Brand identity */}
        <a href="/" className="brand-section" aria-label="Northstar Policy Assistant Home">
          <div className="brand-logo-mark" aria-hidden="true">
            <Compass size={20} strokeWidth={2.2} />
          </div>
          <div className="brand-text-group">
            <span className="brand-title">Northstar</span>
            <span className="brand-subtitle">College Policy Assistant</span>
          </div>
        </a>

        {/* Right header actions */}
        <div className="header-actions">
          {/* Clear conversation action (visible only when there are messages) */}
          {hasMessages && (
            <button
              type="button"
              className="btn-pill-ghost"
              onClick={onClearChat}
              title="Clear current conversation"
              aria-label="Clear current conversation"
            >
              <RotateCcw size={14} />
              <span>Clear Chat</span>
            </button>
          )}
        </div>
      </div>
    </header>
  );
}
