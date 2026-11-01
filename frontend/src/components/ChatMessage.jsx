import React from "react";
import { Compass } from "lucide-react";

/**
 * Remove raw [Source N] tags, parenthesized citations like ( [Source 2], [Source 3] ),
 * and clean punctuation so the text reads naturally.
 *
 * @param {string} text - Raw input text from LLM.
 * @returns {string} Cleaned text ready for formatting.
 */
function cleanSourceTags(text) {
  if (!text) return "";

  let cleaned = text;

  // 1. Remove parenthesized sources e.g. ( [Source 2] , [Source 3] ) or ( Source 2 )
  cleaned = cleaned.replace(/\s*\(\s*(\[?Source\s*\d+\]?\s*[,;]?\s*)+\)\s*/gi, " ");

  // 2. Normalize "As stated in [Source 1]:" -> "According to the policy:"
  cleaned = cleaned.replace(/As stated in \[?Source\s*\d+\]?:/gi, "According to the policy:");

  // 3. Remove standalone [Source N] tags
  cleaned = cleaned.replace(/\s*\[Source\s*\d+\]\s*/gi, " ");

  // 4. Remove any empty leftover parentheses like ()
  cleaned = cleaned.replace(/\(\s*\)/g, "");

  // 5. Clean extra spaces before punctuation e.g. "course . However" -> "course. However"
  cleaned = cleaned.replace(/\s+([.,;:!?])/g, "$1");

  // 6. Normalize multiple spaces
  cleaned = cleaned.replace(/[ \t]+/g, " ");

  return cleaned.trim();
}

/**
 * Render inline markdown elements: **bold** and *italic*.
 *
 * @param {string} text - Cleaned line text.
 * @returns {Array} Rendered React elements.
 */
function renderFormattedLine(text) {
  if (!text) return null;

  // Match **bold** and *italic* tokens
  const regex = /(\*\*.*?\*\*|\*.*?\*)/g;
  const parts = text.split(regex);

  return parts.map((part, index) => {
    if (!part) return null;

    // Bold: **text**
    if (part.startsWith("**") && part.endsWith("**") && part.length >= 4) {
      return (
        <strong key={index} className="formatted-bold">
          {part.slice(2, -2)}
        </strong>
      );
    }

    // Italic: *text*
    if (part.startsWith("*") && part.endsWith("*") && part.length >= 2) {
      return (
        <em key={index} className="formatted-italic">
          {part.slice(1, -1)}
        </em>
      );
    }

    return part;
  });
}

/**
 * Render text blocks as paragraphs, lists, or blockquotes.
 *
 * @param {string} block - Contiguous text block.
 * @param {number} blockIndex - Key index.
 * @returns {JSX.Element}
 */
function renderBlock(block, blockIndex) {
  const cleanedBlock = cleanSourceTags(block);
  if (!cleanedBlock) return null;

  const lines = cleanedBlock.split("\n").filter((l) => l.trim().length > 0);
  if (lines.length === 0) return null;

  // Check if block is a bulleted list
  const isBulletList = lines.every(
    (line) =>
      line.trim().startsWith("- ") ||
      line.trim().startsWith("* ") ||
      line.trim().startsWith("• ")
  );

  if (isBulletList) {
    return (
      <ul key={blockIndex} className="formatted-list">
        {lines.map((line, lIdx) => {
          const itemText = line.trim().replace(/^[-*•]\s*/, "");
          return <li key={lIdx}>{renderFormattedLine(itemText)}</li>;
        })}
      </ul>
    );
  }

  // Check if block is a blockquote
  if (cleanedBlock.startsWith("> ")) {
    const quoteText = lines.map((l) => l.replace(/^>\s*/, "")).join(" ");
    return (
      <blockquote key={blockIndex} className="formatted-quote">
        {renderFormattedLine(quoteText)}
      </blockquote>
    );
  }

  // Standard paragraph with line breaks preserved
  return (
    <p key={blockIndex}>
      {lines.map((line, lIdx) => (
        <React.Fragment key={lIdx}>
          {renderFormattedLine(line)}
          {lIdx < lines.length - 1 && <br />}
        </React.Fragment>
      ))}
    </p>
  );
}

/**
 * Render a chat message in the conversation thread.
 * Clean, properly formatted typography with zero source tags or raw bracket clutter.
 *
 * @param {Object} props
 * @param {Object} props.message - Message data object.
 */
export default function ChatMessage({ message }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="message-bubble-wrapper user">
        <div className="user-message-bubble">{message.content}</div>
      </div>
    );
  }

  // Split response by double newlines into blocks
  const rawBlocks = message.content
    ? message.content.split(/\n\s*\n/).filter((b) => b.trim().length > 0)
    : [];

  return (
    <div className="message-bubble-wrapper assistant">
      <div className="assistant-message-card">
        {/* Assistant Header */}
        <div className="assistant-card-header">
          <div className="assistant-author-group">
            <div className="assistant-avatar-badge" aria-hidden="true">
              <Compass size={14} />
            </div>
            <span className="assistant-author-name">Northstar Policy Assistant</span>
          </div>
          {message.timestamp && (
            <span style={{ fontSize: "0.75rem", color: "var(--color-text-muted)" }}>
              {message.timestamp}
            </span>
          )}
        </div>

        {/* Clean, Formatted Answer Typography (No source brackets or tags) */}
        <div className="assistant-body-text">
          {rawBlocks.length > 0 ? (
            rawBlocks.map((block, idx) => renderBlock(block, idx))
          ) : (
            <p>{cleanSourceTags(message.content)}</p>
          )}
        </div>
      </div>
    </div>
  );
}
