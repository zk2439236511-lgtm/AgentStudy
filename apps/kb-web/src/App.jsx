import { useCallback, useEffect, useState } from 'react';
import AskPanel from './AskPanel.jsx';
import UploadPanel from './UploadPanel.jsx';
import './App.css';

export default function App() {
  const [status, setStatus] = useState(null);
  const [statusError, setStatusError] = useState('');

  const refreshStatus = useCallback(async () => {
    try {
      const response = await fetch('/kb/status');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setStatus(await response.json());
      setStatusError('');
    } catch (error) {
      setStatusError(`读不到索引状态：${error.message}（后端起来了吗？）`);
    }
  }, []);

  useEffect(() => {
    refreshStatus();
  }, [refreshStatus]);

  return (
    <main className="kb-page">
      <header className="kb-header">
        <h1 className="kb-title">我的知识库</h1>
        <p className="kb-subtitle">
          上传 PDF / txt / md → 切块向量化 → 提问时先检索原文，再让通义千问照着原文回答
        </p>
        <p className="kb-status" data-testid="kb-status">
          {statusError ||
            (status
              ? `已入库 ${status.files} 个文件 · ${status.vectors} 个向量块`
              : '正在读取索引状态…')}
        </p>
      </header>

      <UploadPanel onIngested={refreshStatus} />
      <AskPanel />
    </main>
  );
}
