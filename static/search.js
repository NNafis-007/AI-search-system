// static/search.js

// this will be called when the button is clicked
async function sendSearch() {
    const query = document.getElementById("searchInput").value.trim();
    if (!query) return;
  
    const resp = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, limit: 10 })
    });
    if (!resp.ok) {
      console.error("Search failed:", resp.statusText);
      return;
    }
  
    const { results } = await resp.json();
    renderResults(results);
  }
  
  function renderResults(results) {
    const container = document.getElementById("results");
    container.innerHTML = "";
  
    if (results.length === 0) {
      container.innerHTML = '<p class="no-results">No products found.</p>';
      return;
    }
  
    results.forEach(({ product_id, text }) => {
      const card = document.createElement("div");
      card.className = "product-card";
      card.innerHTML = `
        <h2>${product_id}</h2>
        <p>${text}</p>
      `;
      container.appendChild(card);
    });
  }
  
  // wait until the DOM is ready, then wire up the button
  document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("searchButton")
            .addEventListener("click", sendSearch);
  });
  