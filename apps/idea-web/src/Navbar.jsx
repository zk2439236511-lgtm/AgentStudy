import { useState } from 'react';
import './Navbar.css';

export default function Navbar() {
  const [active, setActive] = useState('console');

  return (
    <header className="navbar" data-testid="navbar">
      <div className="navbar-inner">
        <a className="navbar-logo" href="#" data-testid="navbar-logo" aria-label="灵感引擎首页">
          <svg
            className="logo-mark"
            width="30"
            height="30"
            viewBox="0 0 32 32"
            fill="none"
            aria-hidden="true"
          >
            <defs>
              <linearGradient id="logoGradient" x1="0" y1="0" x2="32" y2="32">
                <stop offset="0%" stopColor="#38bdf8" />
                <stop offset="100%" stopColor="#818cf8" />
              </linearGradient>
            </defs>
            <rect width="32" height="32" rx="9" fill="url(#logoGradient)" opacity="0.18" />
            <path
              d="M16 5l2.6 8.4L27 16l-8.4 2.6L16 27l-2.6-8.4L5 16l8.4-2.6L16 5z"
              fill="url(#logoGradient)"
            />
          </svg>
          <span className="logo-text">灵感引擎</span>
          <span className="logo-sub">IdeaSpark</span>
        </a>

        <nav className="navbar-tabs" role="tablist" aria-label="主导航">
          <button
            type="button"
            role="tab"
            aria-selected={active === 'console'}
            className={active === 'console' ? 'nav-tab is-active' : 'nav-tab'}
            data-testid="nav-tab-console"
            onClick={() => setActive('console')}
          >
            控制台
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={active === 'docs'}
            className={active === 'docs' ? 'nav-tab is-active' : 'nav-tab'}
            data-testid="nav-tab-docs"
            onClick={() => setActive('docs')}
          >
            文档中心
          </button>
        </nav>
      </div>
    </header>
  );
}
