async function sendText() {
    const input = document.getElementById("inputText").value;
    const response = await fetch("/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input })
    });
    const data = await response.json();
    document.getElementById("outputText").value = data.result;
}
