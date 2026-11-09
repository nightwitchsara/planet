import { defineConfig } from 'vite';

// Cross-origin isolation lets the simulation worker share the world with the
// tile workers through SharedArrayBuffer instead of copying it to each one.
const isolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
};

export default defineConfig({
  worker: { format: 'es' },
  server: { port: 5317, strictPort: true, headers: isolation },
  preview: { headers: isolation },
  build: { target: 'es2022', sourcemap: true },
});
