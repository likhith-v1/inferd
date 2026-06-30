import { BarChart3, Boxes, Gauge, LayoutDashboard, RadioTower, X } from "lucide-react";
import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/streams", label: "Streams", icon: RadioTower },
  { to: "/spec-decode", label: "Spec decode", icon: Gauge },
  { to: "/memory", label: "Memory", icon: Boxes },
  { to: "/benchmarks", label: "Benchmarks", icon: BarChart3 }
];

export default function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <>
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <div className="brand-row">
          <span className="brand-mark" aria-hidden="true">i</span>
          <div>
            <strong>inferd</strong>
            <span>ENGINE</span>
          </div>
          <button className="icon-button close-sidebar" type="button" onClick={onClose} aria-label="Close navigation">
            <X size={18} />
          </button>
        </div>
        <nav className="nav-list" aria-label="Dashboard">
          {navItems.map(({ to, label, icon: Icon }) => (
            <NavLink key={to} to={to} end={to === "/"} onClick={onClose}>
              <Icon size={17} aria-hidden="true" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="capture-card">
          <span aria-hidden="true">?</span>
          <strong>Demo capture</strong>
          <p>Dashboard under load with live polling and benchmark overlays.</p>
          <span className="capture-link">docs/demo.md</span>
        </div>
      </aside>
      <button className={`scrim ${open ? "show" : ""}`} type="button" onClick={onClose} aria-label="Close navigation" />
    </>
  );
}
