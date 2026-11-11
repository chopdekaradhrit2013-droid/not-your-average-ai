import argparse
import json
import os
import re
import subprocess
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def execute_tool(tool_call, skills=None):
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
            if "dir" in skill:
                return (
                    f"Skill: {skill['name']} (located at {skill['dir']})\n"
                    f"Paths in the instructions below are relative to that folder.\n\n"
                    f"{body}"
                )
            return body
        return f"Error: Skill '{skill_name}' not found."

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
                    body = ""
                    if content.startswith("---"):
                        parts = content[3:].split("---", 1)
                        if len(parts) >= 2:
                            frontmatter = parts[0]
                            body = parts[1].strip()
                            for line in frontmatter.splitlines():
                                line = line.strip()
                                if line.startswith("name:"):
                                    name = line.split("name:", 1)[1].strip()
                                elif line.startswith("description:"):
                                    description = line.split("description:", 1)[1].strip()
                    else:
                        body = content.strip()
                    skill_data = {
                        "name": name,
                        "description": description,
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
        },
    ]

    skills = load_skills()

    messages = []
    if skills:
        skills_list = []
        seen = set()
        for skill in skills.values():
            if skill["name"] in seen:
                continue
            seen.add(skill["name"])
            skills_list.append(f"- {skill['name']}: {skill['description']}")
        system_content = (
            "You have access to the following skills:\n\n"
            + "\n".join(skills_list)
            + "\n\nIf a skill matches the user's request, call the Skill tool with its name and follow the instructions it returns."
        )
        messages.append({"role": "system", "content": system_content})

    remaining = args.p.strip()
    invoked_skills = []

    while remaining:
        parts = remaining.split(None, 1)
        token = parts[0]
        if token.startswith("/") and token[1:] in skills:
            invoked_skills.append(token[1:])
            remaining = parts[1].strip() if len(parts) > 1 else ""
        else:
            break

    raw_args = remaining

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
            if message.content:
                print(message.content)
            break

        for tool_call in message.tool_calls:
            result = execute_tool(tool_call, skills=skills)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                }
            )


if __name__ == "__main__":
    main()
