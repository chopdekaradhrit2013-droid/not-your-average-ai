import asyncio
import os
import sys
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from openai import OpenAI
from pydantic import BaseModel

from app.main import (
    API_KEY,
    BASE_URL,
    execute_tool,
    load_skills,
    poll_lsp_diagnostics,
    substitute_skill_body,
)

try:
    from app.lsp_client import LSPClient
except ImportError:
    LSPClient = None

app = FastAPI(title="Not Your Average AI - Claude Code Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str
    api_key: str | None = None
    base_url: str | None = None


class ToolExecutionLog(BaseModel):
    tool: str
    args: Dict[str, Any]
    output: str


class ChatResponse(BaseModel):
    response: str
    tool_calls: List[ToolExecutionLog]


def run_agent_web(prompt: str, custom_api_key: str | None = None, custom_base_url: str | None = None):
    api_key = custom_api_key or os.getenv("OPENROUTER_API_KEY") or API_KEY
    base_url = custom_base_url or os.getenv("OPENROUTER_BASE_URL") or BASE_URL

    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="OPENROUTER_API_KEY is not configured. Please set it in Railway environment variables or pass it in the request.",
        )

    client = OpenAI(api_key=api_key, base_url=base_url)

    lsp_client = None
    if LSPClient:
        try:
            lsp_client = LSPClient.from_environment()
        except Exception:
            pass

    skills = load_skills()
    remaining = prompt.strip()
    invoked_skills = []

    while remaining:
        parts = remaining.split(None, 1)
        token = parts[0]
        if token.startswith("/") and token[1:] in skills:
            sname = token[1:]
            invoked_skills.append(sname)
            remaining = parts[1].strip() if len(parts) > 1 else ""
            if skills[sname].get("context") == "fork":
                break
        else:
            break

    raw_args = remaining

    tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a file",
                "parameters": {
                    "type": "object",
                    "required": ["file_path", "content"],
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path of the file to write to",
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file",
                        },
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Execute a shell command",
                "parameters": {
                    "type": "object",
                    "required": ["command"],
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute",
                        },
                    },
                },
            },
        },
    ]

    if lsp_client:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "LSP",
                    "description": "Returns the file, line, and character where a symbol is defined, or finds all references to a symbol.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "operation": {
                                "type": "string",
                                "enum": ["goToDefinition", "findReferences"],
                            },
                            "filePath": {
                                "type": "string",
                                "description": "Absolute path, or a path relative to the agent's current working directory.",
                            },
                            "line": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "1-based line number of the symbol.",
                            },
                            "character": {
                                "type": "integer",
                                "minimum": 1,
                                "description": "1-based character offset in UTF-16 code units (same as character count for ASCII).",
                            },
                        },
                        "required": ["operation", "filePath", "line", "character"],
                    },
                },
            }
        )

    if skills and not invoked_skills:
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": "Skill",
                    "description": "Load a skill's instructions into the conversation",
                    "parameters": {
                        "type": "object",
                        "required": ["name"],
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "The name of the skill to use",
                            },
                            "args": {
                                "type": "string",
                                "description": "Optional arguments for the skill",
                            },
                        },
                    },
                },
            }
        )

    messages = []
    system_parts = []
    if skills:
        skills_list = []
        seen = set()
        for skill in skills.values():
            if skill["name"] in seen:
                continue
            seen.add(skill["name"])
            skills_list.append(f"- {skill['name']}: {skill['description']}")
        skills_text = "You have access to the following skills:\n\n" + "\n".join(skills_list)
        if not invoked_skills:
            skills_text += (
                "\n\nIf a skill matches the user's request, call the Skill tool with its name and follow the instructions it returns."
            )
        system_parts.append(skills_text)

    if lsp_client:
        lsp_text = (
            "When migrating or refactoring code across multiple files, you must edit only one file per turn.\n"
            "Edit the definition or API file first, and do NOT edit caller or dependent files in the same turn.\n"
            "After writing a file, wait for language server diagnostics (<new-diagnostics>) in the tool result to see type errors, and use those diagnostics to repair caller files in subsequent turns.\n"
            "After all Write edits, when repeating findReferences at the updated definition, use goToDefinition first (e.g. from a caller) or inspect line numbers to find the exact 1-based line and character where the definition is now located, and call findReferences at that exact definition position."
        )
        system_parts.append(lsp_text)

    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    if invoked_skills:
        for skill_name in invoked_skills:
            skill = skills[skill_name]
            body = substitute_skill_body(skill["body"], raw_args)
            content = (
                f"Skill: {skill['name']} (located at {skill['dir']})\n"
                f"Paths in the instructions below are relative to that folder.\n\n"
                f"{body}"
            )
            messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": prompt})

    tool_execution_logs: List[ToolExecutionLog] = []

    # Agent Loop
    max_turns = 25
    turn_count = 0

    while turn_count < max_turns:
        turn_count += 1
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=tools,
        )

        if not chat.choices:
            raise RuntimeError("No choices in LLM response")

        message = chat.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content or "", tool_execution_logs

        written_files = set()
        for tool_call in message.tool_calls:
            try:
                import json
                tool_args = json.loads(tool_call.function.arguments)
            except Exception:
                tool_args = {}

            result = execute_tool(
                tool_call,
                client=client,
                tools=tools,
                skills=skills,
                lsp_client=lsp_client,
                written_files=written_files,
            )
            tool_execution_logs.append(
                ToolExecutionLog(
                    tool=tool_call.function.name,
                    args=tool_args,
                    output=str(result)[:500] + ("..." if len(str(result)) > 500 else ""),
                )
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        if lsp_client and messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "tool":
            new_diags = poll_lsp_diagnostics(lsp_client)
            if new_diags:
                if len(new_diags) == 1:
                    diag_text = (
                        f"\n\n<new-diagnostics>The following new diagnostic issues were detected: {new_diags[0]}</new-diagnostics>"
                    )
                else:
                    diag_text = (
                        "\n\n<new-diagnostics>The following new diagnostic issues were detected:\n"
                        + "\n".join(new_diags)
                        + "\n</new-diagnostics>"
                    )
                messages[-1]["content"] += diag_text

    return "Agent reached maximum turns.", tool_execution_logs


@app.get("/health")
def health():
    return {"status": "healthy", "service": "not-your-average-ai"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    try:
        response_text, tool_logs = await asyncio.to_thread(
            run_agent_web, req.prompt, req.api_key, req.base_url
        )
        return ChatResponse(response=response_text, tool_calls=tool_logs)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Not Your Average AI - Autonomous Coding Assistant</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
  <style>
    pre code { background: #1e293b; color: #f8fafc; border-radius: 6px; padding: 12px; display: block; overflow-x: auto; }
    .tool-badge { font-family: monospace; font-size: 0.75rem; }
  </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen flex flex-col font-sans">
  <!-- Header -->
  <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur sticky top-0 z-10 px-6 py-4 flex items-center justify-between">
    <div class="flex items-center space-x-3">
      <div class="h-9 w-9 rounded-lg bg-gradient-to-tr from-indigo-500 to-cyan-400 flex items-center justify-center font-bold text-lg text-white shadow-lg shadow-indigo-500/20">
        AI
      </div>
      <div>
        <h1 class="font-bold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent">
          Not Your Average AI
        </h1>
        <p class="text-xs text-slate-400">Autonomous Coding Agent &bull; 100% CodeCrafters Challenge</p>
      </div>
    </div>
    <div class="flex items-center space-x-3 text-sm">
      <a href="https://github.com/chopdekaradhrit2013-droid/not-your-average-ai" target="_blank" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition flex items-center space-x-2 border border-slate-700">
        <i class="fa-brands fa-github"></i>
        <span>GitHub</span>
      </a>
      <span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
        <span class="h-1.5 w-1.5 rounded-full bg-emerald-400 mr-1.5 animate-pulse"></span>
        Ready
      </span>
    </div>
  </header>

  <!-- Main Container -->
  <main class="flex-1 max-w-4xl w-full mx-auto p-4 md:p-6 flex flex-col">
    <!-- Messages View -->
    <div id="chat-box" class="flex-1 space-y-6 overflow-y-auto mb-4 pb-12">
      <!-- Welcome Card -->
      <div class="p-6 rounded-2xl bg-gradient-to-b from-slate-900 to-slate-900/50 border border-slate-800 shadow-xl">
        <h2 class="text-xl font-semibold text-white mb-2">👋 Welcome to your Autonomous Coding Agent!</h2>
        <p class="text-slate-400 text-sm leading-relaxed mb-4">
          This agent can read files, write code, run shell commands, execute custom skills, and diagnose type errors using the Language Server Protocol.
        </p>
        <div class="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Try asking:</div>
        <div class="flex flex-wrap gap-2">
          <button onclick="setPrompt('Calculate 37 * 4 and explain prime factorization')" class="px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-xs text-slate-300 border border-slate-700 transition">
            🧮 Math & Reasoning
          </button>
          <button onclick="setPrompt('List files in the current directory and check git status')" class="px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-xs text-slate-300 border border-slate-700 transition">
            📂 Inspect Workspace
          </button>
          <button onclick="setPrompt('Create a python script hello.py that prints a greeting and run it with python3')" class="px-3 py-1.5 rounded-lg bg-slate-800/80 hover:bg-slate-700 text-xs text-slate-300 border border-slate-700 transition">
            ✍️ Write & Execute Code
          </button>
        </div>
      </div>
    </div>

    <!-- Input Form -->
    <div class="sticky bottom-4 bg-slate-900/90 backdrop-blur border border-slate-800 rounded-2xl p-2 shadow-2xl">
      <form id="chat-form" onsubmit="handleSubmit(event)" class="flex flex-col space-y-2">
        <textarea
          id="prompt-input"
          rows="2"
          placeholder="Give the agent a task (e.g. 'Create a script that calculates fibonacci numbers and test it')..."
          class="w-full bg-transparent px-4 py-2 text-sm text-slate-100 placeholder-slate-500 focus:outline-none resize-none"
          onkeydown="if(event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); handleSubmit(event); }"
        ></textarea>
        <div class="flex items-center justify-between px-2 pt-1 border-t border-slate-800/60">
          <div class="text-[11px] text-slate-500 flex items-center space-x-1">
            <span>Powered by Anthropic Claude &bull; Autonomous Tools Enabled</span>
          </div>
          <button
            type="submit"
            id="send-btn"
            class="px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs transition flex items-center space-x-2 shadow-lg shadow-indigo-600/30 disabled:opacity-50"
          >
            <span>Run Agent</span>
            <i class="fa-solid fa-arrow-up"></i>
          </button>
        </div>
      </form>
    </div>
  </main>

  <script>
    function setPrompt(text) {
      document.getElementById('prompt-input').value = text;
      document.getElementById('prompt-input').focus();
    }

    async function handleSubmit(e) {
      if (e) e.preventDefault();
      const input = document.getElementById('prompt-input');
      const prompt = input.value.trim();
      if (!prompt) return;

      const chatBox = document.getElementById('chat-box');
      const sendBtn = document.getElementById('send-btn');

      // Append user message
      const userBubble = document.createElement('div');
      userBubble.className = "flex justify-end";
      userBubble.innerHTML = `
        <div class="bg-indigo-600 text-white px-4 py-2.5 rounded-2xl rounded-br-sm max-w-xl text-sm shadow-md">
          ${prompt.replace(/</g, "&lt;").replace(/>/g, "&gt;")}
        </div>
      `;
      chatBox.appendChild(userBubble);
      input.value = "";

      // Append Loading / Agent message
      const agentBubble = document.createElement('div');
      agentBubble.className = "flex flex-col space-y-3 max-w-3xl";
      agentBubble.innerHTML = `
        <div class="flex items-center space-x-2 text-xs text-indigo-400 font-medium">
          <i class="fa-solid fa-spinner fa-spin"></i>
          <span>Agent thinking & executing tools...</span>
        </div>
      `;
      chatBox.appendChild(agentBubble);
      chatBox.scrollTop = chatBox.scrollHeight;

      sendBtn.disabled = true;

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt: prompt })
        });

        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.detail || 'Failed to execute agent prompt');
        }

        let toolsHtml = "";
        if (data.tool_calls && data.tool_calls.length > 0) {
          toolsHtml = `
            <div class="space-y-1.5 my-2">
              <div class="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center space-x-1">
                <i class="fa-solid fa-screwdriver-wrench"></i>
                <span>Executed ${data.tool_calls.length} tool call(s)</span>
              </div>
              <div class="space-y-1">
                ${data.tool_calls.map(tc => `
                  <details class="bg-slate-900 border border-slate-800 rounded-lg p-2 text-xs">
                    <summary class="cursor-pointer font-mono font-medium text-indigo-300 hover:text-indigo-200 flex items-center justify-between">
                      <span>[${tc.tool}] ${JSON.stringify(tc.args).slice(0, 70)}...</span>
                      <span class="text-[10px] text-slate-500">output</span>
                    </summary>
                    <pre class="mt-2 text-[11px] text-slate-300 overflow-x-auto">${tc.output.replace(/</g, "&lt;").replace(/>/g, "&gt;")}</pre>
                  </details>
                `).join('')}
              </div>
            </div>
          `;
        }

        agentBubble.innerHTML = `
          <div class="bg-slate-900 border border-slate-800 px-5 py-4 rounded-2xl rounded-bl-sm text-sm text-slate-200 shadow-lg space-y-3">
            ${toolsHtml}
            <div class="prose prose-invert max-w-none text-sm leading-relaxed">
              ${marked.parse(data.response || "No response")}
            </div>
          </div>
        `;
      } catch (err) {
        agentBubble.innerHTML = `
          <div class="bg-red-950/40 border border-red-800/60 px-4 py-3 rounded-2xl text-xs text-red-300 shadow">
            <strong>Error:</strong> ${err.message}
          </div>
        `;
      } finally {
        sendBtn.disabled = false;
        chatBox.scrollTop = chatBox.scrollHeight;
      }
    }
  </script>
</body>
</html>
"""
