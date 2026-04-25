"use client";

/**
 * AdminGuard
 *
 * Wraps any admin page with a password prompt.
 * Password is stored in sessionStorage so it persists across navigations
 * within the same browser tab but resets when the tab is closed.
 *
 * Set NEXT_PUBLIC_ADMIN_PASSWORD in your .env.local
 * (defaults to "admin123" if not set).
 */

import { useState, useEffect, ReactNode } from "react";
import { Lock, Eye, EyeOff, ShieldCheck } from "lucide-react";

const ADMIN_PASSWORD =
  process.env.NEXT_PUBLIC_ADMIN_PASSWORD ?? "admin123";

const SESSION_KEY = "admin_auth_ok";

interface AdminGuardProps {
  children: ReactNode;
}

export function AdminGuard({ children }: AdminGuardProps) {
  const [unlocked, setUnlocked] = useState(false);
  const [input,    setInput]    = useState("");
  const [error,    setError]    = useState("");
  const [show,     setShow]     = useState(false);
  const [shaking,  setShaking]  = useState(false);

  // Check sessionStorage on first render
  useEffect(() => {
    if (sessionStorage.getItem(SESSION_KEY) === "true") {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setUnlocked(true);
    }
  }, []);

  function attempt(e: React.FormEvent) {
    e.preventDefault();
    if (input === ADMIN_PASSWORD) {
      sessionStorage.setItem(SESSION_KEY, "true");
      setUnlocked(true);
    } else {
      setError("Incorrect password");
      setInput("");
      setShaking(true);
      setTimeout(() => setShaking(false), 600);
    }
  }

  if (unlocked) return <>{children}</>;

  return (
    <div className="min-h-[70vh] flex items-center justify-center p-4">
      <div
        className={`w-full max-w-sm glass-panel rounded-2xl p-8 border border-slate-200 shadow-2xl
          transition-transform ${shaking ? "animate-shake" : ""}`}
      >
        {/* Icon */}
        <div className="flex justify-center mb-6">
          <div className="p-4 rounded-full bg-gradient-to-br from-indigo-500/20 to-purple-500/20 border border-indigo-500/30">
            <Lock className="w-8 h-8 text-indigo-400" />
          </div>
        </div>

        <h2 className="text-2xl font-extrabold text-center text-slate-800 mb-1">
          Admin Access
        </h2>
        <p className="text-slate-500 text-sm text-center mb-8">
          Enter the admin password to continue
        </p>

        <form onSubmit={attempt} className="space-y-4">
          <div className="relative">
            <input
              id="admin-password-input"
              autoFocus
              type={show ? "text" : "password"}
              value={input}
              onChange={(e) => { setInput(e.target.value); setError(""); }}
              placeholder="Password"
              className={`w-full px-6 py-4 pr-10 bg-slate-50/60 border rounded-2xl text-slate-800
                placeholder-slate-500 focus:outline-none focus:ring-2 transition
                ${error
                  ? "border-red-200 focus:ring-red-500/30"
                  : "border-slate-200 focus:ring-indigo-500/40"
                }`}
            />
            <button
              type="button"
              tabIndex={-1}
              onClick={() => setShow(!show)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-600 transition"
            >
              {show ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          </div>

          {error && (
            <p className="text-red-600 text-sm font-medium text-center">{error}</p>
          )}

          <button
            id="admin-login-btn"
            type="submit"
            className="w-full py-3 rounded-2xl bg-gradient-to-r from-indigo-600 to-purple-600
              hover:from-indigo-500 hover:to-purple-500 text-slate-800 font-bold transition
              shadow-lg shadow-indigo-500/20 flex items-center justify-center gap-2"
          >
            <ShieldCheck className="w-5 h-5" />
            Unlock
          </button>
        </form>
      </div>
    </div>
  );
}
