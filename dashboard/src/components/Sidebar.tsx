import { AlignLeft, BarChart3, LayoutDashboard, PanelLeft, Server, X, Zap } from "lucide-react";
import { NavLink } from "react-router-dom";
import { useDashboard } from "../lib/dashboard";

interface NavItem {
  to: string;
  label: string;
  icon: typeof LayoutDashboard;
  badge?: number;
}

export default function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { generation } = useDashboard();
  const activeStreams = generation.active.length;

  const mainNav: NavItem[] = [
    { to: "/", label: "Overview", icon: LayoutDashboard },
    { to: "/streams", label: "Streams", icon: AlignLeft, badge: activeStreams || undefined },
    { to: "/spec-decode", label: "Spec decode", icon: Zap },
    { to: "/memory", label: "Memory", icon: Server }
  ];
  const analyticsNav: NavItem[] = [{ to: "/benchmarks", label: "Benchmarks", icon: BarChart3 }];

  const renderItem = ({ to, label, icon: Icon, badge }: NavItem) => (
    <NavLink key={to} to={to} end={to === "/"} onClick={onClose}>
      <Icon size={16} aria-hidden="true" />
      <span>{label}</span>
      {badge ? <span className="nav-badge">{badge}</span> : null}
    </NavLink>
  );

  return (
    <>
      <aside className={`sidebar ${open ? "open" : ""}`}>
        <div className="brand-row">
          <strong>inferd</strong>
          <PanelLeft size={17} aria-hidden="true" />
          <button className="icon-button close-sidebar" type="button" onClick={onClose} aria-label="Close navigation">
            <X size={16} />
          </button>
        </div>

        <div className="nav-section-label">MAIN NAVIGATION</div>
        <nav className="nav-list" aria-label="Main navigation">
          {mainNav.map(renderItem)}
        </nav>

        <div className="nav-section-label spaced">ANALYTICS &amp; INSIGHTS</div>
        <nav className="nav-list" aria-label="Analytics and insights">
          {analyticsNav.map(renderItem)}
        </nav>

        <div className="sidebar-spacer" />
      </aside>
      <button className={`scrim ${open ? "show" : ""}`} type="button" onClick={onClose} aria-label="Close navigation" />
    </>
  );
}
