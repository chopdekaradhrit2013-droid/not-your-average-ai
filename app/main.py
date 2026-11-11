import argparse
from collections import defaultdict
import json
import os
import pathlib
import re
import subprocess
import sys
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from openai import OpenAI

try:
    from app.lsp_client import LSPClient
except ImportError:
    try:
        from lsp_client import LSPClient
    except ImportError:
        LSPClient = None

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def poll_lsp_diagnostics(lsp_client):
    if not hasattr(lsp_client, "notification_cursor"):
        lsp_client.notification_cursor = 0
    if not hasattr(lsp_client, "file_diagnostics"):
        lsp_client.file_diagnostics = {}
    if not hasattr(lsp_client, "last_shown_diagnostics"):
        lsp_client.last_shown_diagnostics = {}
    if not hasattr(lsp_client, "document_versions"):
        lsp_client.document_versions = {}

    try:
        resp = lsp_client.notifications(after=lsp_client.notification_cursor, timeout_ms=0)
        print(f"[DEBUG] notifications(after={lsp_client.notification_cursor}) -> {resp}", file=sys.stderr)
    except Exception as e:
        print(f"[DEBUG] notifications exception: {e}", file=sys.stderr)
        return []

    if not resp:
        return []

    if "cursor" in resp:
        lsp_client.notification_cursor = resp["cursor"]

    notifications = resp.get("notifications", [])
    for notif in notifications:
        if notif.get("method") == "textDocument/publishDiagnostics":
            params = notif.get("params", {})
            raw_uri = params.get("uri")
            if not raw_uri:
                continue

            parsed = urlparse(raw_uri)
            if parsed.netloc and parsed.netloc != "localhost":
                path = f"//{parsed.netloc}{unquote(parsed.path)}"
            else:
                path = unquote(parsed.path)
            abs_file = os.path.abspath(url2pathname(path))
            canonical_uri = pathlib.Path(abs_file).as_uri()

            report_version = params.get("version")
            last_sent_version = lsp_client.document_versions.get(canonical_uri)
            if last_sent_version is None:
                last_sent_version = lsp_client.document_versions.get(raw_uri)

            print(f"[DEBUG] publishDiagnostics uri={raw_uri} canonical={canonical_uri} report_ver={report_version} last_sent_ver={last_sent_version}", file=sys.stderr)

            # Check staleness:
            # - If report has a version and we sent a version for this file, they must match.
            # - If no version was supplied in the report, accept it.
            # - If we never sent a version for this file, accept it.
            if report_version is not None and last_sent_version is not None:
                if report_version != last_sent_version:
                    print(f"[DEBUG] Stale report ignored for {canonical_uri}", file=sys.stderr)
                    continue  # Stale report, ignore

            rel_file = os.path.relpath(abs_file, os.getcwd())

            diag_list = []
            for d in params.get("diagnostics", []):
                range_obj = d.get("range", {})
                start = range_obj.get("start", {})
                line = start.get("line", 0) + 1
                char = start.get("character", 0) + 1
                msg = d.get("message", "")
                diag_list.append((line, char, msg))

            # Latest report replaces previous report for this file
            lsp_client.file_diagnostics[rel_file] = diag_list

    new_issue_lines = []
    for rel_file in sorted(lsp_client.file_diagnostics.keys()):
        current_diags = lsp_client.file_diagnostics[rel_file]
        last_shown = lsp_client.last_shown_diagnostics.get(rel_file, set())
        current_set = set(current_diags)

        seen_in_batch = set()
        for line, char, msg in current_diags:
            issue_key = (line, char, msg)
            if issue_key not in last_shown and issue_key not in seen_in_batch:
                new_issue_lines.append(f"{rel_file}: [Line {line}:{char}] {msg}")
                seen_in_batch.add(issue_key)

        lsp_client.last_shown_diagnostics[rel_file] = current_set

    return new_issue_lines


def run_agent_loop(client, messages, tools, skills=None, lsp_client=None):
    while True:
        chat = client.chat.completions.create(
            model="anthropic/claude-haiku-4.5",
            messages=messages,
            tools=tools,
        )

        if not chat.choices or len(chat.choices) == 0:
            raise RuntimeError("no choices in response")

        message = chat.choices[0].message
        messages.append(message)

        if not message.tool_calls:
            return message.content

        for tool_call in message.tool_calls:
            print(f"[DEBUG] Executing tool: {tool_call.function.name} args: {tool_call.function.arguments}", file=sys.stderr)
            result = execute_tool(tool_call, client=client, tools=tools, skills=skills, lsp_client=lsp_client)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )

        if lsp_client and messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "tool":
            new_diags = poll_lsp_diagnostics(lsp_client)
            print(f"[DEBUG] poll_lsp_diagnostics returned: {new_diags}", file=sys.stderr)
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
                print(f"[DEBUG] appended to messages[-1]: {messages[-1]['content']}", file=sys.stderr)


def execute_tool(tool_call, client=None, tools=None, skills=None, lsp_client=None):
    name = tool_call.function.name
    try:
        tool_args = json.loads(tool_call.function.arguments)
    except Exception as e:
        return f"Error parsing arguments: {e}"

    if name in ("Read", "read", "read_file", "ReadFile"):
        file_path = tool_args.get("file_path")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"

    elif name in ("Write", "write", "write_file", "WriteFile"):
        file_path = tool_args.get("file_path")
        content = tool_args.get("content", "")
        try:
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(content)

            if lsp_client:
                abs_path = os.path.abspath(file_path)
                _, ext = os.path.splitext(abs_path)
                if ext in lsp_client.extension_to_language:
                    lang_id = lsp_client.extension_to_language[ext]
                    uri = pathlib.Path(abs_path).as_uri()

                    if not hasattr(lsp_client, "document_versions"):
                        lsp_client.document_versions = {}

                    if uri in lsp_client.document_versions:
                        next_ver = lsp_client.document_versions[uri] + 1
                        lsp_client.notify(
                            "textDocument/didChange",
                            {
                                "textDocument": {
                                    "uri": uri,
                                    "version": next_ver,
                                },
                                "contentChanges": [{"text": content}],
                            },
                        )
                        lsp_client.document_versions[uri] = next_ver
                    else:
                        lsp_client.notify(
                            "textDocument/didOpen",
                            {
                                "textDocument": {
                                    "uri": uri,
                                    "languageId": lang_id,
                                    "version": 1,
                                    "text": content,
                                },
                            },
                        )
                        lsp_client.document_versions[uri] = 1

            return "File written successfully"
        except Exception as e:
            return f"Error writing file: {e}"

    elif name in ("Bash", "bash", "RunBashCommand", "run_bash_command"):
        command = tool_args.get("command")
        try:
            res = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
            )
            return (res.stdout or "") + (res.stderr or "")
        except Exception as e:
            return f"Error executing command: {e}"

    elif name in ("Skill", "skill"):
        skill_name = tool_args.get("name")
        raw_args = tool_args.get("args") or ""
        if skills and skill_name in skills:
            skill = skills[skill_name]
            body = substitute_skill_body(skill["body"], raw_args)
            has_scripts = "scripts/" in body
            content = body
            if "dir" in skill and (has_scripts or os.path.isdir(os.path.join(os.getcwd(), skill["dir"], "scripts"))):
                content = (
                    f"Skill: {skill['name']} (located at {skill['dir']})\n"
                    f"Paths in the instructions below are relative to that folder.\n\n"
                    f"{body}"
                )

            if skill.get("context") == "fork" and client:
                subagent_messages = [{"role": "user", "content": content}]
                subagent_tools = [t for t in tools if t.get("function", {}).get("name") != "Skill"] if tools else []
                subagent_result = run_agent_loop(
                    client, subagent_messages, subagent_tools, skills=skills, lsp_client=lsp_client
                )
                return f"Skill {skill['name']} ran in a separate context and returned: {subagent_result}"

            return content
        return f"Error: Skill '{skill_name}' not found."

    elif name in ("LSP", "lsp"):
        if not lsp_client:
            return "LSP is not implemented yet"
        operation = tool_args.get("operation")
        file_path = tool_args.get("filePath") or tool_args.get("file_path")
        line = int(tool_args.get("line", 1))
        character = int(tool_args.get("character", 1))

        if operation == "goToDefinition":
            abs_path = os.path.abspath(file_path)
            uri = pathlib.Path(abs_path).as_uri()

            if not hasattr(lsp_client, "document_versions"):
                lsp_client.document_versions = {}

            if uri not in lsp_client.document_versions:
                _, ext = os.path.splitext(abs_path)
                lang_id = lsp_client.extension_to_language.get(ext, "python")
                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        file_text = f.read()
                    lsp_client.notify(
                        "textDocument/didOpen",
                        {
                            "textDocument": {
                                "uri": uri,
                                "languageId": lang_id,
                                "version": 1,
                                "text": file_text,
                            }
                        },
                    )
                    lsp_client.document_versions[uri] = 1
                except Exception as e:
                    return f"Error opening document in LSP: {e}"

            try:
                resp = lsp_client.request(
                    "textDocument/definition",
                    {
                        "textDocument": {"uri": uri},
                        "position": {
                            "line": line - 1,
                            "character": character - 1,
                        },
                    },
                )
            except Exception as e:
                return f"Error requesting definition from LSP: {e}"

            if not resp:
                return f"No definition found for {file_path}:{line}:{character}"

            if isinstance(resp, list):
                loc = resp[0]
            else:
                loc = resp

            target_uri = loc.get("uri") or loc.get("targetUri")
            range_obj = loc.get("range") or loc.get("targetRange") or loc.get("targetSelectionRange") or {}
            start = range_obj.get("start", {})
            def_line = start.get("line", 0) + 1
            def_char = start.get("character", 0) + 1

            parsed = urlparse(target_uri)
            target_file = url2pathname(unquote(parsed.path))
            rel_target = os.path.relpath(target_file, os.getcwd())

            return f"Defined in {rel_target}:{def_line}:{def_char}"

        elif operation == "findReferences":
            abs_path = os.path.abspath(file_path)
            uri = pathlib.Path(abs_path).as_uri()

            if not hasattr(lsp_client, "document_versions"):
                lsp_client.document_versions = {}

            if uri not in lsp_client.document_versions:
                _, ext = os.path.splitext(abs_path)
                lang_id = lsp_client.extension_to_language.get(ext, "python")
                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
                        file_text = f.read()
                    lsp_client.notify(
                        "textDocument/didOpen",
                        {
                            "textDocument": {
                                "uri": uri,
                                "languageId": lang_id,
                                "version": 1,
                                "text": file_text,
                            }
                        },
                    )
                    lsp_client.document_versions[uri] = 1
                except Exception as e:
                    return f"Error opening document in LSP: {e}"

            try:
                resp = lsp_client.request(
                    "textDocument/references",
                    {
                        "textDocument": {"uri": uri},
                        "position": {
                            "line": line - 1,
                            "character": character - 1,
                        },
                        "context": {"includeDeclaration": True},
                    },
                )
            except Exception as e:
                return f"Error requesting references from LSP: {e}"

            if not resp:
                return "No references found"

            refs_by_file = defaultdict(list)
            total_refs = 0
            for loc in resp:
                target_uri = loc.get("uri") or loc.get("targetUri")
                if not target_uri:
                    continue
                range_obj = loc.get("range") or loc.get("targetRange") or loc.get("targetSelectionRange") or {}
                start = range_obj.get("start", {})
                ref_line = start.get("line", 0) + 1
                ref_char = start.get("character", 0) + 1

                parsed = urlparse(target_uri)
                target_file = url2pathname(unquote(parsed.path))
                rel_target = os.path.relpath(target_file, os.getcwd())
                refs_by_file[rel_target].append((ref_line, ref_char))
                total_refs += 1

            if total_refs == 0:
                return "No references found"

            lines = [f"Found {total_refs} reference{'s' if total_refs != 1 else ''} across {len(refs_by_file)} file{'s' if len(refs_by_file) != 1 else ''}:"]
            for fpath, positions in refs_by_file.items():
                lines.append(f"{fpath}:")
                for r_line, r_char in positions:
                    lines.append(f"  Line {r_line}:{r_char}")
            return "\n".join(lines)

        return f"Unknown LSP operation: {operation}"

    return ""


def load_skills():
    skills = {}
    skills_dir = os.path.join(os.getcwd(), ".claude", "skills")
    if not os.path.isdir(skills_dir):
        return skills

    for entry in sorted(os.listdir(skills_dir)):
        entry_path = os.path.join(skills_dir, entry)
        if os.path.isdir(entry_path):
            scripts_dir = os.path.join(entry_path, "scripts")
            if os.path.isdir(scripts_dir):
                for s in os.listdir(scripts_dir):
                    sp = os.path.join(scripts_dir, s)
                    if os.path.isfile(sp):
                        try:
                            os.chmod(sp, os.stat(sp).st_mode | 0o111)
                        except Exception:
                            pass

            skill_file = os.path.join(entry_path, "SKILL.md")
            if os.path.isfile(skill_file):
                try:
                    with open(skill_file, "r", encoding="utf-8") as f:
                        content = f.read().lstrip()
                    name = entry
                    description = ""
                    context = ""
                    body = ""
                    if content.startswith("---"):
                        parts = content[3:].split("---", 1)
                        if len(parts) >= 2:
                            frontmatter = parts[0]
                            body = parts[1].strip()
                            for line in frontmatter.splitlines():
                                line = line.strip()
                                if line.startswith("name:"):
                                    name = line.split("name:", 1)[1].strip().strip("'\"")
                                elif line.startswith("description:"):
                                    description = line.split("description:", 1)[1].strip().strip("'\"")
                                elif line.startswith("context:"):
                                    context = line.split("context:", 1)[1].strip().strip("'\"")
                    else:
                        body = content.strip()
                    skill_data = {
                        "name": name,
                        "description": description,
                        "context": context,
                        "body": body,
                        "dir": f".claude/skills/{entry}",
                    }
                    skills[entry] = skill_data
                    if name != entry:
                        skills[name] = skill_data
                except Exception:
                    pass
    return skills


def substitute_skill_body(body, raw_args):
    arg_list = raw_args.split() if raw_args else []

    def replacer(match):
        if match.group(1) is not None:
            idx = int(match.group(1))
            return arg_list[idx] if 0 <= idx < len(arg_list) else ""
        elif match.group(0) == "$ARGUMENTS":
            return raw_args
        elif match.group(2) is not None:
            idx = int(match.group(2))
            return arg_list[idx] if 0 <= idx < len(arg_list) else ""
        return ""

    pattern = re.compile(r"\$ARGUMENTS\[(\d+)\]|\$ARGUMENTS|\$(\d+)")
    return pattern.sub(replacer, body)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    lsp_client = None
    if LSPClient:
        try:
            lsp_client = LSPClient.from_environment()
            if lsp_client:
                print("LSP client connected", file=sys.stderr)
        except Exception as e:
            print(f"Error initializing LSP client: {e}", file=sys.stderr)

    skills = load_skills()

    remaining = args.p.strip()
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
    if skills:
        skills_list = []
        seen = set()
        for skill in skills.values():
            if skill["name"] in seen:
                continue
            seen.add(skill["name"])
            skills_list.append(f"- {skill['name']}: {skill['description']}")
        system_content = "You have access to the following skills:\n\n" + "\n".join(skills_list)
        if not invoked_skills:
            system_content += (
                "\n\nIf a skill matches the user's request, call the Skill tool with its name and follow the instructions it returns."
            )
        messages.append({"role": "system", "content": system_content})

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
        messages.append({"role": "user", "content": args.p})

    final_content = run_agent_loop(client, messages, tools, skills=skills, lsp_client=lsp_client)
    if final_content is not None:
        print(final_content)


if __name__ == "__main__":
    main()
