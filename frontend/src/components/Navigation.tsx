"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  UserPlus, Activity, Home, ScanFace,
  Calendar, LayoutDashboard, BookOpenCheck,
} from "lucide-react";

const navItems = [
  { href: "/",                  label: "Dashboard",     icon: Home },
  { href: "/students",          label: "Students",      icon: UserPlus },
  { href: "/attendance",        label: "Live Scanner",  icon: Activity },
  { href: "/student/attendance",label: "My Attendance", icon: BookOpenCheck },
  { href: "/admin/dashboard",   label: "Admin",         icon: LayoutDashboard },
  { href: "/admin/timetable",   label: "Timetable",     icon: Calendar },
];

export function Navigation() {
  const pathname = usePathname();

  return (
    <nav className="glass-nav sticky top-0 z-50 w-full">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <div className="flex items-center space-x-3 shrink-0">
            <ScanFace className="w-8 h-8 text-blue-600" />
            <span className="font-bold text-xl tracking-tight text-slate-800">FaceID Core</span>
          </div>

          {/* Links */}
          <div className="flex items-center space-x-1 overflow-x-auto">
            {navItems.map(({ href, label, icon: Icon }) => {
              const active = pathname === href;
              return (
                <Link
                  key={href}
                  href={href}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded-2xl text-sm font-medium transition-colors whitespace-nowrap ${
                    active
                      ? "bg-blue-50 text-blue-600"
                      : "text-slate-500 hover:text-slate-900 hover:bg-slate-50"
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{label}</span>
                </Link>
              );
            })}
          </div>
        </div>
      </div>
    </nav>
  );
}
