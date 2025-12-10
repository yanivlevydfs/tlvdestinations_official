document.addEventListener("DOMContentLoaded", () => {

    // -----------------------------
    // POPULAR DESTINATIONS
    // -----------------------------
    fetch("/api/analytics/top?limit=10")
        .then(r => r.json())
        .then(data => {
            const container = document.getElementById("popular-widget");

            if (!container) return;

            if (!data.length) {
                container.innerHTML = "<div class='text-muted'>No data available.</div>";
                return;
            }

            container.innerHTML = `
                <ul class="list-group">
                    ${data.map(row => `
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            <a href="/destinations/${row.iata}" class="fw-bold text-decoration-none">
                                ${row.city} (${row.iata})
                            </a>
                            <span class="badge bg-primary rounded-pill">${row.total_clicks}</span>
                        </li>
                    `).join("")}
                </ul>
            `;
        });


    // -----------------------------
    // TRENDING DESTINATIONS
    // -----------------------------
    fetch("/api/analytics/trending?limit=10")
        .then(r => r.json())
        .then(data => {
            const container = document.getElementById("trending-widget");

            if (!container) return;

            if (!data.length) {
                container.innerHTML = "<div class='text-muted'>No trending destinations today.</div>";
                return;
            }

            container.innerHTML = `
                <ul class="list-group">
                    ${data.map(row => {
                        let badgeClass = "badge-change-flat";
                        let arrow = "→";

                        if (row.change > 0) {
                            badgeClass = "badge-change-up";
                            arrow = "↑";
                        } else if (row.change < 0) {
                            badgeClass = "badge-change-down";
                            arrow = "↓";
                        }

                        return `
                        <li class="list-group-item d-flex justify-content-between align-items-center">
                            <a href="/destinations/${row.iata}" class="fw-bold text-decoration-none">
                                ${row.city} (${row.iata})
                            </a>
                            <span class="badge ${badgeClass} rounded-pill">
                                ${arrow} ${row.change}
                            </span>
                        </li>
                        `;
                    }).join("")}
                </ul>
            `;
        });

});
