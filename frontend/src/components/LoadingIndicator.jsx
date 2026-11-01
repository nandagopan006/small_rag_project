import React from "react";
import { Compass } from "lucide-react";

/**
 * Animated minimal loading indicator matching the editorial theme.
 */
export default function LoadingIndicator() {
  return (
    <div className="message-bubble-wrapper assistant" aria-live="polite">
      <div className="loading-card">
        <Compass size={18} className="loading-compass-icon" aria-hidden="true" />
        <span className="loading-text">
          Searching the college handbook and synthesizing policy answer...
        </span>
      </div>
    </div>
  );
}
