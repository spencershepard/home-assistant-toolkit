#!/usr/bin/env node

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { WebSocketServer } from 'ws';
import { StreamableHTTPServerTransport } from '@modelcontextprotocol/sdk/server/streamableHttp.js';
import express from 'express';
import { v4 as uuidv4 } from 'uuid';
import {
  CallToolRequestSchema,
  ErrorCode,
  ListToolsRequestSchema,
  McpError,
} from '@modelcontextprotocol/sdk/types.js';
import http from 'http';

// Parse command line arguments
const args = process.argv.slice(2);
const wsPort = args.includes('--port') ? parseInt(args[args.indexOf('--port') + 1]) : 4444
const wsHost = args.includes('--host') ? args[args.indexOf('--host') + 1] : '0.0.0.0';
const useWebSocket = args.includes('--ws');

class CanvasMCPServer {
  constructor() {
    this.server = new Server(
      {
        name: 'canvas-mcp-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          tools: {},
        },
      }
    );

    this.setupToolHandlers();
    
    // Error handling
    this.server.onerror = (error) => console.error('[MCP Error]', error);
    process.on('SIGINT', async () => {
      await this.server.close();
      process.exit(0);
    });

    // Session management for Streamable HTTP
    this.transports = {};
    this.pendingTransports = {};
    this.app = express();
    this.app.use(express.json());
    this.setupHttpEndpoints();
  }
  // Helper to create/connect transport for a session
  async createAndConnectTransport(sessionId) {
    if (this.pendingTransports[sessionId] || this.transports[sessionId]) {
      return this.pendingTransports[sessionId] || this.transports[sessionId];
    }
    const transport = new StreamableHTTPServerTransport({
      enableJsonResponse: true,
      eventSourceEnabled: true,
      onsessioninitialized: (actualId) => {
        delete this.pendingTransports[actualId];
      }
    });
    transport.sessionId = sessionId;
    transport.onclose = () => {
      if (this.transports[sessionId]) {
        delete this.transports[sessionId];
      }
    };
    this.pendingTransports[sessionId] = transport;
    this.transports[sessionId] = transport;
    try {
      await this.server.connect(transport);
    } catch (error) {
      delete this.pendingTransports[sessionId];
      delete this.transports[sessionId];
      throw error;
    }
    return transport;
  }
  setupHttpEndpoints() {
    // Health check endpoint
    this.app.get('/health', (req, res) => {
        res.status(200).json({ status: 'ok' });
    });
    // POST /mcp endpoint for Streamable HTTP
    this.app.post('/mcp', async (req, res) => {
      const body = req.body;
      const rpcId = (body && body.id !== undefined) ? body.id : null;
      // Session ID from header
      const clientSessionIdHeader = req.headers['mcp-session-id'];
      const actualClientSessionId = Array.isArray(clientSessionIdHeader)
        ? clientSessionIdHeader[0]
        : clientSessionIdHeader;
      let transport;
      let effectiveSessionId;
      const isInitRequest = body && body.method === 'initialize';
      if (isInitRequest) {
        effectiveSessionId = uuidv4();
        transport = await this.createAndConnectTransport(effectiveSessionId);
        res.setHeader('Mcp-Session-Id', effectiveSessionId);
      } else if (actualClientSessionId && this.transports[actualClientSessionId]) {
        transport = this.transports[actualClientSessionId];
        effectiveSessionId = actualClientSessionId;
      } else {
        return res.status(400).json({
          jsonrpc: '2.0',
          error: { code: -32003, message: 'Bad Request: No valid session ID for non-initialize request.' },
          id: rpcId
        });
      }
      req.headers['mcp-session-id'] = effectiveSessionId;
      res.setHeader('Mcp-Session-Id', effectiveSessionId);
      try {
        await transport.handleRequest(req, res, body);
      } catch (err) {
        if (!res.headersSent) {
          res.status(500).json({
            jsonrpc: '2.0',
            error: { code: -32603, message: 'Internal server error during MCP request handling' },
            id: rpcId
          });
        }
      }
    });
    // DELETE /mcp for session cleanup
    this.app.delete('/mcp', async (req, res) => {
      const sessionId = req.headers['mcp-session-id'];
      if (this.transports[sessionId]) {
        delete this.transports[sessionId];
        res.status(204).end();
      } else {
        res.status(404).json({ error: 'Session not found' });
      }
    });
  }

  // Helper method to make HTTP requests
  async makeRequest(path, method = 'GET', data = null, retries = 3) {
    for (let attempt = 0; attempt < retries; attempt++) {
      try {
        return await new Promise((resolve, reject) => {
          const postData = data ? JSON.stringify(data) : null;
          
          const options = {
            hostname: 'localhost',
            port: 3000,
            path: path,
            method: method,
            timeout: 5000, // 5 second timeout
            headers: {
              'Content-Type': 'application/json',
              ...(postData && { 'Content-Length': Buffer.byteLength(postData) })
            }
          };

          const req = http.request(options, (res) => {
            let body = '';
            res.on('data', (chunk) => body += chunk);
            res.on('end', () => {
              try {
                const result = JSON.parse(body);
                resolve({ ok: res.statusCode >= 200 && res.statusCode < 300, status: res.statusCode, data: result });
              } catch (e) {
                resolve({ ok: res.statusCode >= 200 && res.statusCode < 300, status: res.statusCode, data: body });
              }
            });
          });

          req.on('error', reject);
          req.on('timeout', () => {
            req.destroy();
            reject(new Error('Request timeout'));
          });
          
          if (postData) {
            req.write(postData);
          }
          
          req.end();
        });
      } catch (error) {
        if (attempt === retries - 1) {
          throw new Error(`Failed after ${retries} attempts: ${error.message}. Make sure the web server is running on port 3000.`);
        }
        // Wait before retry
        await new Promise(resolve => setTimeout(resolve, 1000 * (attempt + 1)));
      }
    }
  }

  setupToolHandlers() {
    // List available tools
    this.server.setRequestHandler(ListToolsRequestSchema, async () => ({
      tools: [
        {
          name: 'render_canvas',
          description: 
            'IMPORTANT: Always use JavaScript window.open() instead of iframes for displaying web pages.' +
            'Renders the provided HTML content onto the canvas. Use this when the user asks for a visual representation.' +
            'You can also manipulate the DOM, add event listeners, or run any client-side code.',
          inputSchema: {
            type: 'object',
            properties: {
              html: {
                type: 'string',
                description: 'HTML content to render in the canvas. Can include any valid HTML markup.',
              },
              javascript: {
                type: 'string',
                description: 'Optional JavaScript code to execute. Use this for DOM manipulation, opening URLs, or any client-side functionality.',
              },
            },
            required: ['html'],
          },
        },
        {
          name: 'get_current_canvas',
          description: 'Get the current content displayed on the canvas. Returns both HTML and JavaScript content.',
          inputSchema: {
            type: 'object',
            properties: {},
          },
        },
      ],
    }));

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      try {
        if (request.params.name === 'render_canvas') {
          const { html, javascript = '' } = request.params.arguments;
          
          const response = await this.makeRequest('/render', 'POST', {
            body_html: html,
            script_js: javascript,
          });

          if (!response.ok) {
            throw new McpError(ErrorCode.InternalError, `HTTP ${response.status}`);
          }

          return {
            content: [
              {
                type: 'text',
                text: `Canvas updated successfully! View at http://localhost:3000\nStatus: ${response.data.status}`,
              },
            ],
          };
        }

        if (request.params.name === 'get_current_canvas') {
          const response = await this.makeRequest('/current');
          
          if (!response.ok) {
            throw new McpError(ErrorCode.InternalError, `HTTP ${response.status}`);
          }

          const current = response.data;
          return {
            content: [
              {
                type: 'text',
                text: `Current canvas content:\nHTML: ${current.body_html}\nJavaScript: ${current.script_js || '(none)'}`,
              },
            ],
          };
        }

        throw new McpError(ErrorCode.MethodNotFound, `Unknown tool: ${request.params.name}`);
      } catch (error) {
        if (error instanceof McpError) {
          throw error;
        }
        throw new McpError(ErrorCode.InternalError, `Tool execution failed: ${error.message}`);
      }
    });
  }

  async run() {
    // Start Express HTTP server for Streamable HTTP
    const port = 3001;
    this.app.listen(port, () => {
      console.error(`Canvas MCP server running on HTTP (Streamable) at http://localhost:${port}/mcp`);
    });
    // Also keep stdio for local MCP
    const transport = new StdioServerTransport();
    await this.server.connect(transport);
    console.error('Canvas MCP server running on stdio');
  }
}

const server = new CanvasMCPServer();
server.run().catch(console.error);
