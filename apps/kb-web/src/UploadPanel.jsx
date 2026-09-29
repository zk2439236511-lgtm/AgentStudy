import { useRef, useState } from 'react';

export default function UploadPanel({ onIngested }) {
  const inputRef = useRef(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  async function handleUpload() {
    const file = inputRef.current?.files?.[0];
    if (!file) {
      setError('请先选一个 PDF 文件');
      return;
    }

    const body = new FormData();
    body.append('file', file);

    setBusy(true);
    setMessage('');
    setError('');
    try {
      const response = await fetch('/kb/documents', { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);

      setMessage(
        data.empty
          ? `${data.filename} 没抽出任何文字（可能是扫描件），没有入库`
          : data.skipped
            ? `${data.filename} 内容没变，沿用已有索引（没花钱重新嵌入）`
            : `${data.filename} 已入库，新增 ${data.chunks_added} 个向量块`,
      );
      inputRef.current.value = '';
      onIngested();
    } catch (cause) {
      setError(`入库失败：${cause.message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="kb-card" data-testid="kb-upload">
      <h2 className="kb-card-title">1 · 上传资料</h2>
      <div className="kb-upload-row">
        <input ref={inputRef} type="file" accept="application/pdf" disabled={busy} />
        <button className="kb-button" onClick={handleUpload} disabled={busy}>
          {busy ? '解析入库中…' : '上传并入库'}
        </button>
      </div>
      {message && <p className="kb-message" data-testid="kb-upload-message">{message}</p>}
      {error && <p className="kb-error" data-testid="kb-upload-error">{error}</p>}
    </section>
  );
}
