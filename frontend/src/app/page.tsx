import { Users, CheckCircle, XCircle } from "lucide-react";

async function getStats() {
  // Use absolute URL since fetch in app router runs on server
  try {
    const res = await fetch("http://127.0.0.1:8000/api/stats", {
      cache: "no-store",
    });
    if (!res.ok) throw new Error("Failed to fetch stats");
    return res.json();
  } catch (error) {
    return { success: false, data: null };
  }
}

export default async function Home() {
  const statsRes = await getStats();
  const summary = statsRes?.data?.recognizer || { total_faces: 0, model_name: "Unavailable" };

  return (
    <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div>
        <h1 className="text-4xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-400">
          Dashboard Overview
        </h1>
        <p className="text-slate-400 mt-2 text-lg">
          Live statistics from the FaceID Core engine
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-panel rounded-2xl p-6 hover:-translate-y-1 transition-transform">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-blue-500/20 rounded-xl">
              <Users className="w-8 h-8 text-blue-400" />
            </div>
            <div>
              <p className="text-slate-400 text-sm font-medium">Registered Students</p>
              <h2 className="text-3xl font-bold text-white">{summary.total_faces}</h2>
            </div>
          </div>
        </div>

        <div className="glass-panel rounded-2xl p-6 hover:-translate-y-1 transition-transform">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-emerald-500/20 rounded-xl">
              <CheckCircle className="w-8 h-8 text-emerald-400" />
            </div>
            <div>
              <p className="text-slate-400 text-sm font-medium">System Status</p>
              <h2 className="text-xl font-bold text-white">{statsRes.success ? "Online" : "Offline"}</h2>
            </div>
          </div>
        </div>

        <div className="glass-panel rounded-2xl p-6 hover:-translate-y-1 transition-transform">
          <div className="flex items-center space-x-4">
            <div className="p-3 bg-purple-500/20 rounded-xl">
              <XCircle className="w-8 h-8 text-purple-400" />
            </div>
            <div>
              <p className="text-slate-400 text-sm font-medium">Active Engine</p>
              <h2 className="text-xl font-bold text-white uppercase">{summary.model_name}</h2>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
