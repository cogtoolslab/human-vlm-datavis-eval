#!/usr/bin/env node

/**
 * Module dependencies.
 */

import app from "./App";
import Debug from "debug";
import http from "http";
import https from "https"
import fs from "fs"

const debug = Debug("tecl-backend:server");

/**
 * Get port from environment and store in Express.
 */

const port = normalizePort(process.env.PORT || "3000");
app.set("port", port);

/**
 * Create HTTP server.
 */



// Create HTTPS server

const {PRIVATE_KEY_PATH, CERT_PATH, CHAIN_PATH} = process.env
const server = PRIVATE_KEY_PATH && CERT_PATH && CHAIN_PATH ? https.createServer({
  key: fs.readFileSync(PRIVATE_KEY_PATH),
  cert: fs.readFileSync(CERT_PATH),
  ca: fs.readFileSync(CHAIN_PATH)
}, app) : http.createServer(app);




/**
 * Listen on provided port, on all network interfaces.
 */

server.listen(port);
server.on("error", onError);
server.on("listening", onListening);

/**
 * Normalize a port into a number, string, or false.
 */

function normalizePort(val: string) {
  const port = parseInt(val, 10);

  if (isNaN(port)) {
    // named pipe
    return val;
  }

  if (port >= 0) {
    // port number
    return port;
  }

  return false;
}

/**
 * Event listener for HTTP server "error" event.
 */

function onError(error: NodeJS.ErrnoException) {
  if (error.syscall !== "listen") {
    throw error;
  }

  const bind = typeof port === "string" ? "Pipe " + port : "Port " + port;

  // handle specific listen errors with friendly messages
  switch (error.code) {
    case "EACCES":
      console.error(bind + " requires elevated privileges");
      process.exit(1);

    case "EADDRINUSE":
      console.error(bind + " is already in use");
      process.exit(1);

    default:
      throw error;
  }
}

/**
 * Event listener for HTTP server "listening" event.
 */

function onListening() {
  const addr = server.address();

  if (addr) {
    const bind =
      typeof addr === "string" ? "pipe " + addr : "port " + addr.port;
    debug("Listening on " + bind);
  }
}
