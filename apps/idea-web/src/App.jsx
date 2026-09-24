import Navbar from './Navbar.jsx';
import IdeaComposer from './IdeaComposer.jsx';

const DEMO_CARDS = [
  '用 AI 给今天的待办排优先级',
  '把口头答应同事的事记成灵感',
  '让 AI 把一句话扩写成方案草稿',
  '周末给爸妈打个电话',
  '读完《重构》的第 3 章',
  '整理上周的会议记录',
];

export default function App() {
  return (
    <>
      <Navbar />
      <main className="demo-section">
        <h1 className="demo-title">欢迎回来</h1>
        <p className="demo-hint">输入一句灵感，点击「灵感生成」体验加载动画；再向下滚动，看毛玻璃导航栏如何模糊背后的内容。</p>
        <IdeaComposer />
        <div className="demo-grid">
          {DEMO_CARDS.map((text) => (
            <div className="demo-card" key={text}>
              {text}
            </div>
          ))}
        </div>
      </main>
    </>
  );
}
