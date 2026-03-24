const express = require("express");
const http = require("http");
const path = require("path");
const { WebSocketServer, WebSocket } = require("ws");

const PORT = process.env.PORT || 8080;

const app = express();
app.use(express.static(path.join(__dirname, "public")));

const server = http.createServer(app);
const wss = new WebSocketServer({ server, path: "/ws" });

let lastPayload = {
    type: "direction",
    active: false,
    center_deg: 0,
    width_deg: 0,
    confidence: 0,
    mode: "idle",
    ts: Date.now() / 1000,
};

function broadcastToViewers(payload) {
    const raw = JSON.stringify(payload);

    for (const client of wss.clients) {
        if (client.readyState === WebSocket.OPEN && client.role === "viewer") {
            client.send(raw);
        }
    }
}

wss.on("connection", (ws, req) => {
    const url = new URL(req.url, `http://${req.headers.host}`);
    const role = url.searchParams.get("role") || "viewer";
    ws.role = role;

    console.log(`WS connected: ${role}`);

    if (role === "viewer") {
        ws.send(JSON.stringify(lastPayload));
    }

    ws.on("message", (message) => {
        if (ws.role !== "producer") return;

        try {
            const parsed = JSON.parse(message.toString());
            lastPayload = {
                ...parsed,
                server_ts_ms: Date.now(),
            };
            broadcastToViewers(lastPayload);
        } catch (err) {
            console.error("Invalid producer payload:", err.message);
        }
    });

    ws.on("close", () => {
        console.log(`WS closed: ${role}`);
    });
});

server.listen(PORT, "0.0.0.0", () => {
    console.log(`Viewer UI: http://<your-laptop-ip>:${PORT}`);
    console.log(`Producer WS: ws://<your-laptop-ip>:${PORT}/ws?role=producer`);
});