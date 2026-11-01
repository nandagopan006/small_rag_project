import React from "react";
import { BookOpen } from "lucide-react";

/**
 * Display verified source references returned by the RAG backend.
 *
 * @param {Object} props
 * @param {Array} props.sources - List of source citation objects.
 */
export default function SourceReferences({ sources }) {
  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className="sources-container">
      <div className="sources-header">
        <span className="sources-title">
          <BookOpen size={13} />
          <span>Verified Policy Sources ({sources.length})</span>
        </span>
      </div>

      <div className="sources-list">
        {sources.map((src, index) => {
          const label = src.source_label || `Source ${index + 1}`;
          const page = src.page_number ? `Page ${src.page_number}` : null;
          const section = src.section_title || "General Regulations";

          return (
            <div key={src.chunk_id || index} className="source-item-card">
              <div className="source-item-top">
                <span className="source-label-pill">{label}</span>
                {page && <span className="source-page-tag">{page}</span>}
              </div>
              <div className="source-section-title" title={section}>
                {section}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
