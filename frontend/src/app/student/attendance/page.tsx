"use client";

/**
 * My Attendance Page
 *
 * Lets a student enter their ID and view their full attendance history.
 * Unified dark glass UI matching the rest of the app.
 * Fixed: uses relative /api URL (routed through Next.js proxy to FastAPI).
 */

import { useState } from "react";
import { BookOpenCheck, Search, RefreshCw, Loader2 } from "lucide-react";

interface AttendanceRecord {
  id: number;
  date: string;
  time: string;
  status: string;
  subject_name?: string;
  recognition_confidence?: number;
}

function statusStyle(status: string) {
  switch (status?.toLowerCase()) {
    case "present": return "bg-emerald-100 text-[#10B981] border border-emerald-500/20";
    case "late":    return "bg-amber-500/10  text-amber-600  border border-amber-500/20";
    default:        return "bg-red-50    text-red-600    border border-red-500/20";
  }
}

export default function StudentAttendancePage() {
  const [studentId, setStudentId] = useState("");
  const [history,   setHistory]   = useState<AttendanceRecord[]>([]);
  const [loading,   setLoading]   = useState(false);
  const [error,     setError]     = useState("");
  const [fetched,   setFetched]   = useState(false);

  async function fetchAttendance(e?: React.FormEvent) {
    e?.preventDefault();
    if (!studentId.trim()) return;
    setLoading(true);
    setError("");
    setFetched(false);
    try {
      // Use relative URL — Next.js proxy forwards /api/* to FastAPI
      const res = await fetch(`/api/attendance/student/${studentId.trim()}`);
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail ?? `HTTP ${res.status}`);
      }
      const data = await res.json();
      setHistory(data.data ?? []);
      setFetched(true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Failed to fetch records");
    } finally {
      setLoading(false);
    }
  }

  const totalPresent = history.filter((r) => r.status?.toLowerCase() === "present").length;
  const totalLate    = history.filter((r) => r.status?.toLowerCase() === "late").length;
  const rate         = history.length > 0
    ? Math.round(((totalPresent + totalLate) / history.length) * 100)
    : 0;

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-3xl font-bold text-slate-800 flex items-center gap-3">
          <BookOpenCheck className="w-8 h-8 text-blue-600" />
          My Attendance
        </h1>
        <p className="text-slate-500 mt-1">View your full attendance history by subject</p>
      </div>

      {/* Search bar */}
      <form
        onSubmit={fetchAttendance}
        className="glass-panel rounded-2xl p-5 flex flex-col sm:flex-row gap-4 items-center"
      >
        <div className="relative flex-1 w-full">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 pointer-events-none" />
          <input
            id="student-id-input"
            type="text"
            value={studentId}
            onChange={(e) => setStudentId(e.target.value)}
            placeholder="Enter your Student ID  (e.g. CS-2026-001)"
            className="w-full pl-9 pr-4 py-2.5 bg-slate-50/60 border border-slate-200 rounded-2xl text-slate-800 placeholder-slate-500
              focus:outline-none focus:ring-2 focus:ring-blue-500/40 focus:border-blue-500 transition"
          />
        </div>
        <button
          id="fetch-attendance-btn"
          type="submit"
          disabled={loading || !studentId.trim()}
          className="flex items-center gap-2 px-6 py-2.5 bg-[#2563EB] hover:bg-[#2563EB] rounded-2xl font-semibold text-slate-800
            transition shadow-lg shadow-blue-500/20 disabled:opacity-50 whitespace-nowrap"
        >
          {loading
            ? <><Loader2 className="w-4 h-4 animate-spin" /> Loading…</>
            : <><RefreshCw className="w-4 h-4" /> Fetch Records</>
          }
        </button>
      </form>

      {/* Error */}
      {error && (
        <div className="glass-panel rounded-2xl px-5 py-4 border border-red-200 bg-red-500/5 text-red-600 text-sm">
          ⚠️ {error}
        </div>
      )}

      {/* Stats */}
      {fetched && history.length > 0 && (
        <div className="grid grid-cols-3 gap-4">
          {[
            { label: "Total Classes", value: history.length, color: "text-blue-600" },
            { label: "Present / Late", value: `${totalPresent} / ${totalLate}`, color: "text-[#10B981]" },
            { label: "Attendance Rate", value: `${rate}%`,
              color: rate >= 75 ? "text-[#10B981]" : rate >= 50 ? "text-amber-600" : "text-red-600" },
          ].map(({ label, value, color }) => (
            <div key={label} className="glass-panel rounded-2xl p-5 text-center">
              <div className={`text-3xl font-extrabold ${color} mb-1`}>{value}</div>
              <div className="text-slate-500 text-sm">{label}</div>
            </div>
          ))}
        </div>
      )}

      {/* Table */}
      {fetched && (
        <div className="glass-panel rounded-2xl overflow-hidden">
          {history.length === 0 ? (
            <div className="py-16 text-center">
              <BookOpenCheck className="w-12 h-12 text-slate-700 mx-auto mb-3" />
              <p className="text-slate-500 font-medium">No attendance records found</p>
              <p className="text-slate-600 text-sm mt-1">for student ID: {studentId}</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="bg-slate-50/60 text-slate-500 text-xs font-mono uppercase border-b border-slate-200">
                  <tr>
                    <th className="px-6 py-4 font-semibold">Date</th>
                    <th className="px-6 py-4 font-semibold">Time</th>
                    <th className="px-6 py-4 font-semibold">Subject</th>
                    <th className="px-6 py-4 font-semibold">Status</th>
                    <th className="px-6 py-4 font-semibold text-right">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-700/40">
                  {history.map((record, i) => (
                    <tr key={i} className="hover:bg-slate-50/30 transition-colors">
                      <td className="px-6 py-4 text-slate-900 font-medium">{record.date}</td>
                      <td className="px-6 py-4 text-slate-500 font-mono text-xs">{record.time}</td>
                      <td className="px-6 py-4 text-blue-300 font-medium">
                        {record.subject_name ?? "General"}
                      </td>
                      <td className="px-6 py-4">
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold ${statusStyle(record.status)}`}>
                          {record.status?.toUpperCase()}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-right text-slate-500 text-xs font-mono font-mono">
                        {record.recognition_confidence != null
                          ? `${(record.recognition_confidence * 100).toFixed(1)}%`
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {!fetched && !loading && !error && (
        <div className="glass-panel rounded-2xl py-16 text-center">
          <Search className="w-12 h-12 text-slate-700 mx-auto mb-3" />
          <p className="text-slate-500">Enter your Student ID and click Fetch Records</p>
        </div>
      )}
    </div>
  );
}
