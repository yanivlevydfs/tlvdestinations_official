/* Fly-TLV Service Worker — SAFE MODE */

self.addEventListener("install", () => {
  self.skipWaiting();
});

self.addEventListener("activate", () => {
  self.clients.claim();
});

// 🚫 Do NOT intercept fetch at all
// (navigation fallback can be added later safely)
