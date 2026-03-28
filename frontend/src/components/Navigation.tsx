import Link from "next/link";
import { UserPlus, Activity, Home, ScanFace } from "lucide-react";

export function Navigation() {
  return (
    <nav className="glass-nav sticky top-0 z-50 w-full">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-3">
            <ScanFace className="w-8 h-8 text-blue-400" />
            <span className="font-bold text-xl tracking-tight text-white">FaceID Core</span>
          </div>
          <div className="flex space-x-8">
            <Link href="/" className="flex items-center space-x-2 text-slate-300 hover:text-white transition-colors">
              <Home className="w-5 h-5" />
              <span>Dashboard</span>
            </Link>
            <Link href="/students" className="flex items-center space-x-2 text-slate-300 hover:text-white transition-colors">
              <UserPlus className="w-5 h-5" />
              <span>Students</span>
            </Link>
            <Link href="/attendance" className="flex items-center space-x-2 text-slate-300 hover:text-white transition-colors">
              <Activity className="w-5 h-5" />
              <span>Live Attendance</span>
            </Link>
          </div>
        </div>
      </div>
    </nav>
  );
}
