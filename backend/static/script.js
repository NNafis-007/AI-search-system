// static/script.js

// sendSearch() is called from the button’s onclick
async function sendSearch() {
    console.log("Clicked");
    const query = document.getElementById("inputText").value.trim();
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
  