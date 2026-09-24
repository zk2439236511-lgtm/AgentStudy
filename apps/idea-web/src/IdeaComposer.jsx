import { useState } from 'react';
import './IdeaComposer.css';

export default function IdeaComposer() {
  const [idea, setIdea] = useState('');
  const [loading, setLoading] = useState(false);

  function handleGenerate() {
    if (loading) return;
    if (!idea.trim()) return;
    setLoading(true);
    // 演示：还没有接后端，用定时器模拟一次 AI 生成的耗时请求
    setTimeout(() => {
      setLoading(false);
    }, 2500);
  }

  return (
    <section className="composer" data-testid="idea-composer">
      <div className="composer-row">
        <input
          className="composer-input"
          type="text"
          placeholder="输入一句灵感，例如：把口头答应同事的事变成提醒"
          value={idea}
          disabled={loading}
          data-testid="prompt-input"
          onChange={(e) => setIdea(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleGenerate()}
        />
        <button
          type="button"
          className={loading ? 'composer-btn is-loading' : 'composer-btn'}
          disabled={loading}
          data-testid="prompt-submit"
          onClick={handleGenerate}
        >
          {loading ? '生成中…' : '灵感生成'}
        </button>
      </div>

      {loading && (
        <div className="loading-overlay" data-testid="loading-overlay">
          <div className="spinner" data-testid="loading-spinner" role="status" aria-label="生成中" />
          <p className="loading-text">灵感正在路上…</p>
        </div>
      )}
    </section>
  );
}
