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
          {result.refused && (
            <p className="kb-refusal" data-testid="kb-refusal">
              检索层判定证据不足，已拒答（没有调用模型）。下面列出的片段都不够近，只是线索。
            </p>
          )}
          <article
            className={result.refused ? 'kb-answer kb-answer-refused' : 'kb-answer'}
            data-testid="kb-answer"
          >
            {result.answer}
          </article>
          <ul className="kb-sources" data-testid="kb-sources">
            {result.sources.map((source, index) => (
              <li key={index} className="kb-source">
                <details>
                  <summary>
                    {/* 接口里的 page 是 PyPDFLoader 的 0-based 页号，给人看要 +1；txt/md 没有页 */}
                    来源：{source.filename} ·{' '}
                    {source.page === null ? '纯文本资料' : `第 ${source.page + 1} 页`} ·{' '}
                    距离 {source.distance}（越小越相关）
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
