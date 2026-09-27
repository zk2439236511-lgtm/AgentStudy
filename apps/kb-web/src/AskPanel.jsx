import { useState } from 'react';

export default function AskPanel() {
  const [question, setQuestion] = useState('');
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  async function handleAsk() {
    if (!question.trim()) {
      setError('先输入一个问题');
      return;
    }

    setBusy(true);
    setError('');
    setResult(null);
    try {
      const response = await fetch('/kb/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: question.trim() }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
      setResult(data);
    } catch (cause) {
      setError(`问答失败：${cause.message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="kb-card">
      <h2 className="kb-card-title">2 · 提问</h2>
      <textarea
        className="kb-textarea"
        data-testid="kb-question"
        rows={3}
        value={question}
        placeholder="例如：What is multi-head attention?"
        disabled={busy}
        onChange={(event) => setQuestion(event.target.value)}
      />
      <button className="kb-button" data-testid="kb-ask" onClick={handleAsk} disabled={busy}>
        {busy ? '检索并生成中…' : '问知识库'}
      </button>

      {busy && <p className="kb-loading" data-testid="kb-loading">Loading…</p>}
      {error && <p className="kb-error" data-testid="kb-error">{error}</p>}

      {result && (
        <>
          <article className="kb-answer" data-testid="kb-answer">{result.answer}</article>
          <ul className="kb-sources" data-testid="kb-sources">
            {result.sources.map((source, index) => (
              <li key={index} className="kb-source">
                <details>
                  <summary>
                    来源：{source.filename} · 第 {source.page} 页 · 相关度 {source.relevance}
                  </summary>
                  <p className="kb-excerpt">{source.excerpt}</p>
                </details>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}
