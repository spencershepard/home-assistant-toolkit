import express from "express";
import bodyParser from "body-parser";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// In-memory storage for current content
let currentContent = {
  body_html: `<div class="card"><h2>Welcome!</h2><p>Your agent will paint here.</p></div>`,
  script_js: "",
  timestamp: Date.now()
};

// Middleware
app.use(bodyParser.json({ limit: "2mb" }));
app.use(express.static(path.join(__dirname, "public"))); // optional static assets

// Endpoint: GET /current
app.get("/current", (req, res) => {
  res.json(currentContent);
});

// Endpoint: POST /render
// Accepts { body_html: "<div>...</div>", script_js: "console.log('hi');" }
app.post("/render", (req, res) => {
  const { body_html, script_js } = req.body;
  if (typeof body_html !== "string" || (script_js && typeof script_js !== "string")) {
    return res.status(400).json({ error: "Invalid input" });
  }
  currentContent = { 
    body_html, 
    script_js: script_js || "",
    timestamp: Date.now()
  };
  console.log("Content updated!");
  res.json({ status: "ok" });
});

// Endpoint: GET / → serves the base template
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

// Start server
app.listen(PORT, () => {
  console.log(`Canvas web server running on http://localhost:${PORT}`);
});
