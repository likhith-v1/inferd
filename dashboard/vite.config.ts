import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const target = env.VITE_INFERD_API || "http://localhost:8000";

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        "/metrics": { target, changeOrigin: true },
        "/healthz": { target, changeOrigin: true },
        "/generate": { target, changeOrigin: true }
      }
    }
  };
});
