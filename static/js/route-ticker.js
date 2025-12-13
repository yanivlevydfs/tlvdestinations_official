// ============================================================================
// Fly-TLV Route Updates Ticker 2026 (Final Polished Version)
// ============================================================================

// -------------------------
// Format helper
// -------------------------
function formatDateUTC(dateStr) {
    try {
        return new Date(dateStr).toLocaleDateString("en-GB", {
            day: "2-digit",
            month: "short",
            year: "numeric"
        });
    } catch {
        return dateStr;
    }
}

// -------------------------
// Sentence builders
// -------------------------
function formatRemovedRoute(r, dateText) {
    return `
        <span class="ticker-item removed">
            <i class="bi bi-x-circle-fill"></i>
            <strong>${r.airline}</strong> is not operating flights to
            <strong>${r.city} (${r.iata})</strong>
            <span class="update-date">on ${dateText}</span>
        </span>
    `.trim();
}

function formatAddedRoute(r, dateText) {
    return `
        <span class="ticker-item added">
            <i class="bi bi-check-circle-fill"></i>
            <strong>${r.airline}</strong> is operating flights to
            <strong>${r.city} (${r.iata})</strong>
            <span class="update-date">on ${dateText}</span>
        </span>
    `.trim();
}

// -------------------------
// Main loader
// -------------------------
async function loadRouteTicker() {
    const card = document.getElementById("route-ticker");
    if (!card) return;

    const inner = card.querySelector(".ticker-inner");
    if (!inner) return;

    try {
        const res = await fetch("/api/destinations/diff", {
            headers: { "Accept": "application/json" },
            cache: "no-store"
        });

        if (!res.ok) {
            console.error("Ticker diff fetch failed:", res.status);
            card.classList.add("d-none");
            return;
        }

        const diff = await res.json();
        const added = diff.added || [];
        const removed = diff.removed || [];
        const dateText = formatDateUTC(diff.generated);

        let items = [];

        // Added routes
        for (const r of added) {
            items.push(formatAddedRoute(r, dateText));
        }

        // Removed routes
        for (const r of removed) {
            items.push(formatRemovedRoute(r, dateText));
        }

        // No changes → hide completely
        if (items.length === 0) {
            card.classList.add("d-none");
            return;
        }

        // Populate ticker
        inner.innerHTML = items.join("");

        // Show card
        card.classList.remove("d-none");

    } catch (err) {
        console.error("Ticker diff error:", err);
        card.classList.add("d-none");
    }
}

// -------------------------
// Init on DOM load
// -------------------------
document.addEventListener("DOMContentLoaded", function () {
    loadRouteTicker();

    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
        new bootstrap.Tooltip(el);
    });
});