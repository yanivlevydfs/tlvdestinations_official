// ads-and-modal.js
document.addEventListener("DOMContentLoaded", function () {
  const modal = document.getElementById("mapModal");
  const iframe = document.getElementById("map-frame");

  // AdSense slot IDs
  const adSlots = [
    "3014594736",
    "6530920497",
    "1374247994",
    "9061166326",
    "9683944869",
    "9767705348",
    "5792553893",
    "9109229790"
  ];

  adSlots.forEach(slotId => {
    renderAd("ad-slot-" + slotId, slotId);
  });

  // Bootstrap modal postMessage
  if (modal && iframe) {
    modal.addEventListener("shown.bs.modal", function () {
      iframe.contentWindow.postMessage("modal-shown", "*");
    });
  }
});

function renderAd(containerId, slotId, format = "auto") {
  const container = document.getElementById(containerId);
  if (!container) return;

  if (container.offsetWidth === 0) {
    setTimeout(() => renderAd(containerId, slotId, format), 500);
    return;
  }

  container.innerHTML = ""; // clear if re-rendering

  const ins = document.createElement("ins");
  ins.className = "adsbygoogle";
  ins.style.display = "block";
  ins.setAttribute("data-ad-client", "ca-pub-7964221655161384");
  ins.setAttribute("data-ad-slot", slotId);
  ins.setAttribute("data-ad-format", format);
  ins.setAttribute("data-full-width-responsive", "true");

  container.appendChild(ins);

  try {
    (adsbygoogle = window.adsbygoogle || []).push({});
  } catch (e) {
    console.warn("AdSense render failed", e);
  }
}

// ----------------------------------------------------
// Dynamic Direct Flight Deals injection
// ----------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  const select = document.getElementById("deals-city-select");
  const container = document.getElementById("deals-script-container");
  if (!select || !container) return;

  select.addEventListener("change", () => {
    const iata = select.value;
    const locale = document.documentElement.lang === "he" ? "he" : "en";

    // Clear previous script
    container.innerHTML = "";

    // Create new ad script with dynamic destination
    const script = document.createElement("script");
    script.async = true;
    script.charset = "utf-8";
    script.src =
      `https://tpembd.com/content?currency=ils&trs=461563&shmarker=674907.2323443` +
      `&destination=${iata}` +
      `&target_host=www.aviasales.com%2Fsearch` +
      `&locale=${locale}` +
      `&limit=1&powered_by=true&width=360` +
      `&primary=%230085FF&promo_id=4044&campaign_id=100`;

    container.appendChild(script);
  });
});
