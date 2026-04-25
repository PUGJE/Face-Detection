"use client";

/**
 * Admin Dashboard
 *
 * Two sections:
 * 1. Subject Summary Cards — attendance % per timetable slot
 * 2. Student × Subject Matrix — per-student attendance count for every subject
 */

import { useEffect, useState } from "react";
import { BookOpen, Users, RefreshCw, ChevronDown, ChevronUp } from "lucide-react";
import { AdminGuard } from "@/components/AdminGuard";


interface SubjectStat {
  timetable_id: number;
  subject_name: string;
  day_of_week: string;
  start_time: string;
  end_time: string;
  total_records: number;
  present_count: number;
  late_count: number;
}

interface MatrixSubject {
  id: number;
  subject_name: string;
  day_of_week: string;
  start_time: string;
  end_time: string;
}

interface MatrixStudent {
  student_id: string;
  student_name: string;
  attendance: Record<number, number>; // timetable_id -> count
}

interface MatrixData {
  subjects: MatrixSubject[];
  students: MatrixStudent[];
}

function countBadge(count: number): string {
  if (count === 0) return "bg-slate-50 text-slate-500";
  if (count === 1) return "bg-amber-100 text-amber-700";
  return "bg-emerald-100 text-[#10B981]";
}

export default function AdminDashboardPage() {
  const [stats,       setStats]       = useState<SubjectStat[]>([]);
  const [matrix,      setMatrix]      = useState<MatrixData | null>(null);
  const [loading,     setLoading]     = useState(true);
  const [matrixLoading, setMatrixLoading] = useState(true);
  const [error,       setError]       = useState("");
  const [showMatrix,  setShowMatrix]  = useState(true);
  const [sortCol,     setSortCol]     = useState<number | "name">("name");
  const [sortAsc,     setSortAsc]     = useState(true);

  useEffect(() => {
    fetchAll();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function fetchAll() {
    fetchSummary();
    fetchMatrix();
  }

  async function fetchSummary() {
    setLoading(true);
    setError("");
    try {
      const res  = await fetch(`/api/attendance/summary`);
      if (!res.ok) throw new Error("Failed to fetch statistics");
      const data = await res.json();
      setStats(data.data ?? []);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function fetchMatrix() {
    setMatrixLoading(true);
    try {
      const res  = await fetch(`/api/attendance/report/matrix`);
      if (!res.ok) throw new Error("Failed to fetch matrix");
      const data = await res.json();
      setMatrix(data.data ?? null);
    } catch {
      setMatrix(null);
    } finally {
      setMatrixLoading(false);
    }
  }

  // ------------------------------------------------------------------
  // Sorting
  // ------------------------------------------------------------------

  function toggleSort(col: number | "name") {
    if (sortCol === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(col);
      setSortAsc(false); // default: highest count first for subject cols
    }
  }

  function sortedStudents(students: MatrixStudent[]): MatrixStudent[] {
    return [...students].sort((a, b) => {
      let cmp: number;
      if (sortCol === "name") {
        cmp = a.student_name.localeCompare(b.student_name);
      } else {
        cmp = (a.attendance[sortCol] ?? 0) - (b.attendance[sortCol] ?? 0);
      }
      return sortAsc ? cmp : -cmp;
    });
  }

  function SortIcon({ col }: { col: number | "name" }) {
    if (sortCol !== col) return <span className="opacity-20">↕</span>;
    return sortAsc ? <ChevronUp className="inline w-3.5 h-3.5" /> : <ChevronDown className="inline w-3.5 h-3.5" />;
  }

  return (
    <AdminGuard>
      <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
        <div className="max-w-7xl mx-auto space-y-10">

        {/* Header */}
        <div className="flex items-center justify-between">
          <h1 className="text-4xl font-extrabold text-slate-800">
            Admin Dashboard
          </h1>
          <button
            onClick={fetchAll}
            className="flex items-center gap-2 px-4 py-2 bg-slate-50 hover:bg-slate-200 rounded-2xl text-slate-600 text-sm font-medium transition"
          >
            <RefreshCw className="w-4 h-4" />
            Refresh
          </button>
        </div>

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-600 p-4 rounded-2xl text-sm">
            {error}
          </div>
        )}

        {/* ── Section 1: Subject Summary Cards ──────────────────────── */}
        <section>
          <div className="flex items-center gap-3 mb-5">
            <BookOpen className="w-5 h-5 text-purple-600" />
            <h2 className="text-xl font-bold text-slate-800">Attendance by Subject</h2>
          </div>

          {loading ? (
            <div className="flex justify-center py-12">
              <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-purple-500" />
            </div>
          ) : stats.length === 0 ? (
            <div className="bg-white border border-slate-200 p-10 rounded-2xl text-center text-slate-500 text-sm">
              No data yet — create timetable slots and mark some attendance.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {stats.map((stat, i) => {
                const total          = stat.total_records;
                const attendanceRate = total > 0
                  ? Math.round(((stat.present_count + stat.late_count) / total) * 100)
                  : 0;

                return (
                  <div
                    key={i}
                    className="bg-white border border-slate-200 p-6 rounded-2xl flex flex-col relative overflow-hidden group hover:border-purple-200 transition-colors"
                  >
                    {/* BG icon */}
                    <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
                      <svg className="w-16 h-16 text-purple-600" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6l5.25 3.15-.92 1.44-6.33-3.8V7z" />
                      </svg>
                    </div>

                    <h3 className="text-xl font-bold text-slate-800 mb-1">{stat.subject_name}</h3>
                    <p className="text-purple-600 text-sm mb-5 flex items-center gap-2">
                      <span className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
                      {stat.day_of_week} ({stat.start_time.slice(0, 5)} – {stat.end_time.slice(0, 5)})
                    </p>

                    <div className="mt-auto">
                      <div className="flex justify-between items-end mb-2">
                        <div className="text-4xl font-black">{attendanceRate}%</div>
                        <div className="text-slate-500 text-xs font-mono text-right">
                          <div>{stat.present_count} Present</div>
                          <div>{stat.late_count} Late</div>
                        </div>
                      </div>
                      <div className="w-full bg-slate-100 h-2 rounded-full overflow-hidden border border-slate-200">
                        <div
                          className="bg-[#2563EB] h-full rounded-full transition-all duration-700"
                          style={{ width: `${attendanceRate}%` }}
                        />
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </section>

        {/* ── Section 2: Student × Subject Matrix ───────────────────── */}
        <section>
          <div className="flex items-center justify-between mb-5">
            <div className="flex items-center gap-3">
              <Users className="w-5 h-5 text-blue-600" />
              <h2 className="text-xl font-bold text-slate-800">Student × Subject Matrix</h2>
              {matrix && (
                <span className="text-xs text-slate-500 bg-slate-50 px-2 py-0.5 rounded-full">
                  {matrix.students.length} students · {matrix.subjects.length} subjects
                </span>
              )}
            </div>
            <button
              onClick={() => setShowMatrix(!showMatrix)}
              className="text-slate-500 hover:text-slate-800 transition text-sm flex items-center gap-1"
            >
              {showMatrix ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              {showMatrix ? "Collapse" : "Expand"}
            </button>
          </div>

          {showMatrix && (
            matrixLoading ? (
              <div className="flex justify-center py-12">
                <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500" />
              </div>
            ) : !matrix || (matrix.students.length === 0 || matrix.subjects.length === 0) ? (
              <div className="bg-white border border-slate-200 p-10 rounded-2xl text-center text-slate-500 text-sm">
                No data to display. Register students and create timetable slots first.
              </div>
            ) : (
              <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden">
                {/* Legend */}
                <div className="flex items-center gap-5 px-6 py-3 border-b border-slate-200 text-xs text-slate-500">
                  <span>Click a column header to sort</span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-3 rounded bg-emerald-50 inline-block" /> 2+ sessions
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-3 rounded bg-amber-500/20 inline-block" /> 1 session
                  </span>
                  <span className="flex items-center gap-1.5">
                    <span className="w-3 h-3 rounded bg-slate-50 inline-block" /> Absent
                  </span>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="text-slate-500 border-b border-slate-200">
                        {/* Student name col */}
                        <th
                          className="sticky left-0 z-10 bg-white px-6 py-4 text-left font-semibold cursor-pointer hover:text-slate-800 transition min-w-[160px]"
                          onClick={() => toggleSort("name")}
                        >
                          Student <SortIcon col="name" />
                        </th>
                        {/* One col per subject */}
                        {matrix.subjects.map((sub) => (
                          <th
                            key={sub.id}
                            className="px-6 py-4 text-center font-semibold cursor-pointer hover:text-slate-800 transition min-w-[110px]"
                            onClick={() => toggleSort(sub.id)}
                          >
                            <div className="text-xs text-slate-700 font-bold truncate max-w-[100px] mx-auto">
                              {sub.subject_name}
                            </div>
                            <div className="text-neutral-600 font-normal text-[10px] mt-0.5 truncate">
                              {sub.day_of_week.slice(0, 3)} {sub.start_time.slice(0, 5)}
                            </div>
                            <SortIcon col={sub.id} />
                          </th>
                        ))}
                        {/* Row total */}
                        <th className="px-6 py-4 text-center font-semibold text-slate-500 min-w-[70px]">
                          Total
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-neutral-800">
                      {sortedStudents(matrix.students).map((student, ri) => {
                        const rowTotal = Object.values(student.attendance).reduce((a, b) => a + b, 0);
                        return (
                          <tr
                            key={student.student_id}
                            className={`transition-colors hover:bg-slate-50/40 ${
                              ri % 2 === 0 ? "" : "bg-white/30"
                            }`}
                          >
                            {/* Name */}
                            <td className="sticky left-0 z-10 bg-white px-6 py-4 font-medium text-slate-800 group-hover:bg-slate-50">
                              <div className="font-semibold">{student.student_name}</div>
                              <div className="text-slate-500 text-xs font-mono font-mono">{student.student_id}</div>
                            </td>
                            {/* Count per subject */}
                            {matrix.subjects.map((sub) => {
                              const cnt = student.attendance[sub.id] ?? 0;
                              return (
                                <td key={sub.id} className="px-6 py-4 text-center">
                                  <span
                                    className={`inline-flex items-center justify-center w-8 h-8 rounded-2xl text-sm font-bold ${countBadge(cnt)}`}
                                  >
                                    {cnt}
                                  </span>
                                </td>
                              );
                            })}
                            {/* Row total */}
                            <td className="px-6 py-4 text-center">
                              <span className="text-sm font-bold text-purple-600">{rowTotal}</span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>

                    {/* Column totals footer */}
                    <tfoot>
                      <tr className="border-t border-slate-200 bg-white/60">
                        <td className="sticky left-0 z-10 bg-white/80 px-6 py-4 text-xs font-semibold text-slate-500 uppercase tracking-wider">
                          Total
                        </td>
                        {matrix.subjects.map((sub) => {
                          const colTotal = matrix.students.reduce(
                            (acc, s) => acc + (s.attendance[sub.id] ?? 0), 0
                          );
                          return (
                            <td key={sub.id} className="px-6 py-4 text-center">
                              <span className="text-sm font-bold text-slate-600">{colTotal}</span>
                            </td>
                          );
                        })}
                        <td className="px-6 py-4 text-center text-sm font-bold text-purple-300">
                          {matrix.students.reduce(
                            (acc, s) => acc + Object.values(s.attendance).reduce((a, b) => a + b, 0), 0
                          )}
                        </td>
                      </tr>
                    </tfoot>
                  </table>
                </div>
              </div>
            )
          )}
        </section>
        </div>
      </div>
    </AdminGuard>
  );
}
