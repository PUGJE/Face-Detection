"use client";

import { useState, useEffect } from "react";
import { Calendar, Plus, Pencil, Trash2, Save, X, Clock } from "lucide-react";
import { AdminGuard } from "@/components/AdminGuard";

const DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"];


interface Slot {
  id: number;
  subject_name: string;
  day_of_week: string;
  start_time: string;
  end_time: string;
  teacher_id?: string;
}

const emptyForm = (): Omit<Slot, "id"> => ({
  subject_name: "",
  day_of_week: "Monday",
  start_time: "09:00",
  end_time: "10:00",
  teacher_id: "",
});

// Normalise "HH:MM:SS" → "HH:MM" for <input type="time">
function toInputTime(t: string): string {
  return t?.slice(0, 5) ?? "";
}

export default function TimetableManagerPage() {
  const [slots,      setSlots]      = useState<Slot[]>([]);
  const [loading,    setLoading]    = useState(true);
  const [error,      setError]      = useState("");
  const [showForm,   setShowForm]   = useState(false);
  const [editingId,  setEditingId]  = useState<number | null>(null);
  const [form,       setForm]       = useState(emptyForm());
  const [saving,     setSaving]     = useState(false);
  const [deleteId,   setDeleteId]   = useState<number | null>(null);
  const [activeSlot, setActiveSlot] = useState<Slot | null>(null);

  useEffect(() => {
    fetchSlots();
    fetchActive();
  }, []);

  async function fetchSlots() {
    setLoading(true);
    setError("");
    try {
      const res  = await fetch(`/api/timetable`);
      const data = await res.json();
      setSlots(data.data ?? []);
    } catch {
      setError("Failed to load timetable slots.");
    } finally {
      setLoading(false);
    }
  }

  async function fetchActive() {
    try {
      const res  = await fetch(`/api/timetable/active`);
      const data = await res.json();
      setActiveSlot(data.active ? data.data : null);
    } catch {
      setActiveSlot(null);
    }
  }

  // ------------------------------------------------------------------
  // Create / Update
  // ------------------------------------------------------------------

  function openCreate() {
    setEditingId(null);
    setForm(emptyForm());
    setShowForm(true);
  }

  function openEdit(slot: Slot) {
    setEditingId(slot.id);
    setForm({
      subject_name: slot.subject_name,
      day_of_week:  slot.day_of_week,
      start_time:   toInputTime(slot.start_time),
      end_time:     toInputTime(slot.end_time),
      teacher_id:   slot.teacher_id ?? "",
    });
    setShowForm(true);
  }

  async function saveSlot(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    try {
      const url    = editingId ? `/api/timetable/${editingId}` : `/api/timetable`;
      const method = editingId ? "PUT" : "POST";
      const res    = await fetch(url, {
        method,
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail ?? "Save failed");
      }
      setShowForm(false);
      await fetchSlots();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSaving(false);
    }
  }

  // ------------------------------------------------------------------
  // Delete
  // ------------------------------------------------------------------

  async function confirmDelete() {
    if (!deleteId) return;
    try {
      await fetch(`/api/timetable/${deleteId}`, { method: "DELETE" });
      setDeleteId(null);
      await fetchSlots();
    } catch {
      setError("Failed to delete slot.");
    }
  }

  // ------------------------------------------------------------------
  // Render helpers
  // ------------------------------------------------------------------

  const dayOrder = (d: string) => DAYS.indexOf(d);
  const sortedSlots = [...slots].sort(
    (a, b) => dayOrder(a.day_of_week) - dayOrder(b.day_of_week) ||
              a.start_time.localeCompare(b.start_time)
  );

  return (
    <AdminGuard>
      <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">

        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="p-3 bg-indigo-500/20 rounded-2xl border border-indigo-500/20">
              <Calendar className="w-8 h-8 text-indigo-400" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-slate-800">Timetable Manager</h1>
              <p className="text-slate-500 text-sm mt-0.5">
                Manage class slots — the scanner uses these to gate face scans
              </p>
            </div>
          </div>
          <button
            id="btn-add-slot"
            onClick={openCreate}
            className="flex items-center gap-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 rounded-2xl font-semibold transition shadow-lg shadow-indigo-500/20"
          >
            <Plus className="w-5 h-5" />
            Add Slot
          </button>
        </div>

        {/* Active class chip */}
        {activeSlot && (
          <div className="glass-panel flex items-center gap-3 px-6 py-4 border border-emerald-200 bg-emerald-500/5 rounded-2xl text-emerald-300 text-sm">
            <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
            <span className="font-semibold">Active now:</span>
            <span>{activeSlot.subject_name} — {toInputTime(activeSlot.start_time)} to {toInputTime(activeSlot.end_time)}</span>
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="glass-panel border border-red-200 bg-red-500/5 text-red-600 px-6 py-4 rounded-2xl text-sm">
            {error}
          </div>
        )}

        {/* Create / Edit form */}
        {showForm && (
          <div className="glass-panel rounded-2xl p-6 border border-slate-600 shadow-2xl">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-bold text-slate-800">
                {editingId ? "Edit Slot" : "New Timetable Slot"}
              </h2>
              <button onClick={() => setShowForm(false)} className="text-slate-500 hover:text-slate-800 transition">
                <X className="w-5 h-5" />
              </button>
            </div>

            <form onSubmit={saveSlot} className="grid grid-cols-1 sm:grid-cols-2 gap-5">
              <div className="sm:col-span-2">
                <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">Subject Name</label>
                <input required value={form.subject_name}
                  onChange={(e) => setForm({ ...form, subject_name: e.target.value })}
                  placeholder="e.g. Deep Learning"
                  className="w-full px-4 py-2.5 bg-slate-50/60 border border-slate-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-indigo-500/40 text-slate-800 placeholder-slate-500 transition"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">Day</label>
                <select value={form.day_of_week} onChange={(e) => setForm({ ...form, day_of_week: e.target.value })}
                  className="w-full px-4 py-2.5 bg-slate-50/60 border border-slate-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-indigo-500/40 text-slate-800 transition">
                  {DAYS.map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">
                  Teacher ID <span className="text-slate-600 normal-case font-normal">(optional)</span>
                </label>
                <input value={form.teacher_id} onChange={(e) => setForm({ ...form, teacher_id: e.target.value })}
                  placeholder="T001"
                  className="w-full px-4 py-2.5 bg-slate-50/60 border border-slate-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-indigo-500/40 text-slate-800 placeholder-slate-500 transition"
                />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">Start Time</label>
                <input required type="time" value={form.start_time} onChange={(e) => setForm({ ...form, start_time: e.target.value })}
                  className="w-full px-4 py-2.5 bg-slate-50/60 border border-slate-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-indigo-500/40 text-slate-800 transition" />
              </div>
              <div>
                <label className="block text-xs font-semibold text-slate-500 uppercase tracking-wider mb-1.5">End Time</label>
                <input required type="time" value={form.end_time} onChange={(e) => setForm({ ...form, end_time: e.target.value })}
                  className="w-full px-4 py-2.5 bg-slate-50/60 border border-slate-200 rounded-2xl focus:outline-none focus:ring-2 focus:ring-indigo-500/40 text-slate-800 transition" />
              </div>
              <div className="sm:col-span-2 flex justify-end gap-3 pt-2">
                <button type="button" onClick={() => setShowForm(false)}
                  className="px-5 py-2.5 rounded-2xl bg-slate-50 hover:bg-slate-700 text-slate-600 font-medium transition">Cancel</button>
                <button id="btn-save-slot" type="submit" disabled={saving}
                  className="flex items-center gap-2 px-5 py-2.5 rounded-2xl bg-indigo-600 hover:bg-indigo-500 text-slate-800 font-semibold transition disabled:opacity-60">
                  <Save className="w-4 h-4" />
                  {saving ? "Saving…" : editingId ? "Update Slot" : "Create Slot"}
                </button>
              </div>
            </form>
          </div>
        )}

        {/* Delete confirm modal */}
        {deleteId !== null && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="glass-panel border border-slate-200 rounded-2xl p-8 max-w-sm w-full shadow-2xl">
              <Trash2 className="w-12 h-12 text-red-600 mx-auto mb-4" />
              <h3 className="text-xl font-bold text-center text-slate-800 mb-2">Delete Slot?</h3>
              <p className="text-slate-500 text-sm text-center mb-6">
                Attendance records will be preserved but unlinked from this subject.
              </p>
              <div className="flex gap-3">
                <button onClick={() => setDeleteId(null)}
                  className="flex-1 py-2.5 rounded-2xl bg-slate-50 hover:bg-slate-700 text-slate-600 font-medium transition">Cancel</button>
                <button id="btn-confirm-delete" onClick={confirmDelete}
                  className="flex-1 py-2.5 rounded-2xl bg-red-600 hover:bg-red-500 text-slate-800 font-semibold transition">Delete</button>
              </div>
            </div>
          </div>
        )}

        {/* Slots table */}
        {loading ? (
          <div className="flex justify-center py-16">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-indigo-500" />
          </div>
        ) : sortedSlots.length === 0 ? (
          <div className="glass-panel rounded-2xl p-16 text-center">
            <Calendar className="w-16 h-16 text-slate-700 mx-auto mb-4" />
            <p className="text-slate-500 font-medium">No timetable slots yet.</p>
            <p className="text-slate-600 text-sm mt-1">Click &quot;Add Slot&quot; to create your first class.</p>
          </div>
        ) : (
          <div className="glass-panel rounded-2xl overflow-hidden">
            <table className="w-full text-left">
              <thead>
                <tr className="border-b border-slate-200/60 text-slate-500 text-xs font-mono uppercase tracking-wider">
                  <th className="px-6 py-4 font-semibold">Subject</th>
                  <th className="px-6 py-4 font-semibold">Day</th>
                  <th className="px-6 py-4 font-semibold">Time</th>
                  <th className="px-6 py-4 font-semibold">Teacher</th>
                  <th className="px-6 py-4 font-semibold text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-700/40">
                {sortedSlots.map((slot) => {
                  const isNow = activeSlot?.id === slot.id;
                  return (
                    <tr key={slot.id}
                      className={`group transition-colors ${
                        isNow ? "bg-emerald-500/5 hover:bg-emerald-50" : "hover:bg-slate-50/30"
                      }`}>
                      <td className="px-6 py-4">
                        <div className="flex items-center gap-2">
                          {isNow && <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />}
                          <span className="font-semibold text-slate-800">{slot.subject_name}</span>
                        </div>
                      </td>
                      <td className="px-6 py-4 text-slate-600">{slot.day_of_week}</td>
                      <td className="px-6 py-4 text-slate-600">
                        <span className="flex items-center gap-1.5 font-mono text-sm">
                          <Clock className="w-3.5 h-3.5 text-slate-500" />
                          {toInputTime(slot.start_time)} – {toInputTime(slot.end_time)}
                        </span>
                      </td>
                      <td className="px-6 py-4 text-slate-500 text-sm">{slot.teacher_id || "—"}</td>
                      <td className="px-6 py-4 text-right">
                        <div className="flex items-center justify-end gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                          <button onClick={() => openEdit(slot)}
                            className="p-2 rounded-2xl bg-slate-50 hover:bg-indigo-500/20 hover:text-indigo-400 text-slate-500 transition" title="Edit">
                            <Pencil className="w-4 h-4" />
                          </button>
                          <button onClick={() => setDeleteId(slot.id)}
                            className="p-2 rounded-2xl bg-slate-50 hover:bg-red-50 hover:text-red-600 text-slate-500 transition" title="Delete">
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </AdminGuard>
  );
}
