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
function formatRemovedRoute(r, dateText, isHebrew) {
    return isHebrew
        ? `
        <span class="ticker-item removed">
            <i class="bi bi-x-circle-fill"></i>
            <strong>${r.airline}</strong>
            אינה מפעילה טיסות ל
            <strong>${r.city} (${r.iata})</strong>
            <span class="update-date">בתאריך ${dateText}</span>
        </span>
        `.trim()
        : `
        <span class="ticker-item removed">
            <i class="bi bi-x-circle-fill"></i>
            <strong>${r.airline}</strong> is not operating flights to
            <strong>${r.city} (${r.iata})</strong>
            <span class="update-date">on ${dateText}</span>
        </span>
        `.trim();
}


function formatAddedRoute(r, dateText, isHebrew) {
    return isHebrew
        ? `
        <span class="ticker-item added">
            <i class="bi bi-check-circle-fill"></i>
            <strong>${r.airline}</strong>
            מפעילה טיסות ל
            <strong>${r.city} (${r.iata})</strong>
            <span class="update-date">בתאריך ${dateText}</span>
        </span>
        `.trim()
        : `
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
	const isHebrew = document.documentElement.lang === "he" ||localStorage.getItem("fe-lang") === "he";
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
            items.push(formatAddedRoute(r, dateText, isHebrew));

        }

        // Removed routes
        for (const r of removed) {
			items.push(formatRemovedRoute(r, dateText, isHebrew));
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

    const isHebrew =
        document.documentElement.lang === "he" ||
        localStorage.getItem("fe-lang") === "he";

    document.querySelectorAll('[data-bs-toggle="tooltip"]').forEach(el => {
        el.setAttribute(
            "title",
            isHebrew
                ? "מבוסס על לוח הטיסות שפורסם להיום. זמינות היעדים עשויה להשתנות בין תאריכים."
                : "Based on today’s published flight schedule. Routes may appear or disappear on different dates."
        );
        new bootstrap.Tooltip(el);
    });
});
