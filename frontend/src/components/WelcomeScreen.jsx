import React from "react";
import { ArrowUpRight, Sparkles } from "lucide-react";
import ChatInput from "./ChatInput";

/**
 * Concise, frequently referenced student policy questions.
 */
const SUGGESTED_PILLS = [
  {
    label: "Minimum attendance for semester exams",
    query: "What is the minimum attendance percentage required for examination eligibility?",
  },
  {
    label: "Undergraduate library borrowing limits",
    query: "How many books can an undergraduate student borrow and for how long?",
  },
  {
    label: "Procedure for academic accommodations",
    query: "What is the procedure to request academic accommodations for students with disabilities?",
  },
  {
    label: "Penalties for academic integrity violations",
    query: "What penalties apply to plagiarism or cheating under the academic integrity code?",
  },
];

/**
 * Minimal, single-viewport welcome screen.
 * Everything fits naturally on one screen with zero scrolling.
 *
 * @param {Object} props
 * @param {Function} props.onSend - Handler to submit a question.
 * @param {boolean} props.isLoading - Whether request is in progress.
 * @param {Object} [props.handbookInfo] - Handbook metadata.
 */
export default function WelcomeScreen({ onSend, isLoading, handbookInfo }) {
  return (
    <div className="welcome-single-screen">
      <div className="welcome-center-box">
        {/* Subtle Eyebrow */}
        <div className="welcome-eyebrow">
          <Sparkles size={13} aria-hidden="true" />
          <span>Academic Year 2026–2027 · Policy Handbook</span>
        </div>

        {/* Editorial Headline */}
        <h1 className="welcome-heading">
          Clarity for every <span className="accent-highlight">college policy.</span>
        </h1>

        {/* Supporting description */}
        <p className="welcome-description">
          Ask questions about your college handbook and get clear answers backed
          by relevant policy references.
        </p>

        {/* Embedded Chat Composer */}
        <div className="welcome-composer-area">
          <ChatInput onSend={onSend} isLoading={isLoading} />
        </div>

        {/* Suggested Quick Question Pills */}
        <div className="suggested-pills-container">
          <span className="suggested-pills-label">Frequently Asked:</span>
          <div className="suggested-pills-list">
            {SUGGESTED_PILLS.map((item, idx) => (
              <button
                key={idx}
                type="button"
                className="suggested-pill-btn"
                onClick={() => onSend(item.query)}
                disabled={isLoading}
                title={item.query}
              >
                <span>{item.label}</span>
                <ArrowUpRight size={13} className="pill-arrow-icon" aria-hidden="true" />
              </button>
            ))}
          </div>
        </div>

        {/* Subtle Handbook Status Info */}
        <div className="welcome-status-footer">
          <span>{handbookInfo?.title || "Northstar College Handbook"}</span>
          <span className="footer-dot">·</span>
          <span>10 Pages</span>
          <span className="footer-dot">·</span>
          <span>72 Policy Chunks</span>
        </div>
      </div>
    </div>
  );
}
