// static/search.js

let lastQuery = "";  // store the user's latest search

// Called when the button is clicked
async function sendSearch() {
  const query = document.getElementById("searchInput").value.trim();
  if (!query) return;

  lastQuery = query;  // remember it for later

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

// Render the list of product cards
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

    // When the user clicks the card, store the (query, positive) pair
    card.addEventListener("click", () => {
      storeQueryPositive(lastQuery, text);
    });

    container.appendChild(card);
  });
}

// Call your store endpoint
async function storeQueryPositive(user_query, positive) {
  try {
    const resp = await fetch("/api/store/query_positive", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_query, positive })
    });
    if (!resp.ok) {
      console.error("Failed to store query-positive:", resp.statusText);
    } else {
      console.log("Stored:", { user_query, positive });
    }
  } catch (err) {
    console.error("Error storing query-positive:", err);
  }
}

// Wire up the search button once the DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("searchButton")
          .addEventListener("click", sendSearch);
});
