#!/usr/bin/env python3
import os
import uuid
import json
import requests
import gradio as gr


def build_messages(history: list, message: str, system_prompt: str):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    for item in history:
        if not item:
            continue
        user = item[0] if len(item) > 0 else ""
        assistant = item[1] if len(item) > 1 else ""
        if user:
            messages.append({"role": "user", "content": user})
        if assistant:
            messages.append({"role": "assistant", "content": assistant})
    if message:
        messages.append({"role": "user", "content": message})
    return messages


def stream_generate(
    message: str,
    history: list,
    session_id: str,
    api_base: str,
    api_key: str,
    system_prompt: str,
    enable_think: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
):
    if not session_id:
        session_id = str(uuid.uuid4())

    messages = build_messages(history, message, system_prompt)
    payload = {
        "messages": messages,
        "stream": True,
        "max_tokens": int(max_new_tokens),
        "max_completion_tokens": int(max_new_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "thinking": "on" if enable_think else "off",
    }

    url = api_base.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    assistant = ""
    thinking = ""
    in_think = False
    pending = ""
    thinking_open = False
    thinking_seen = False

    try:
        with requests.post(url, headers=headers, json=payload, stream=True) as resp:
            resp.encoding = "utf-8"
            if resp.status_code != 200:
                assistant = f"[HTTP {resp.status_code}] {resp.text.strip()}"
                yield history + [(message, assistant)], session_id, thinking
                return

            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if not line.startswith("data: "):
                    continue
                data = line[len("data: ") :].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                delta_content = delta.get("content") or ""
                if not delta_content:
                    continue

                if enable_think:
                    buf = pending + delta_content
                    pending = ""
                    while buf:
                        if in_think:
                            end_idx = buf.find("</think>")
                            if end_idx != -1:
                                thinking += buf[:end_idx]
                                buf = buf[end_idx + len("</think>") :]
                                in_think = False
                                thinking_open = False
                            else:
                                keep = min(len(buf), len("</think>") - 1)
                                if keep > 0:
                                    thinking += buf[:-keep]
                                    pending = buf[-keep:]
                                else:
                                    pending = buf
                                buf = ""
                        else:
                            start_idx = buf.find("<think>")
                            if start_idx != -1:
                                assistant += buf[:start_idx]
                                buf = buf[start_idx + len("<think>") :]
                                in_think = True
                                thinking_open = True
                                thinking_seen = True
                            else:
                                keep = min(len(buf), len("<think>") - 1)
                                if keep > 0:
                                    assistant += buf[:-keep]
                                    pending = buf[-keep:]
                                else:
                                    pending = buf
                                buf = ""
                else:
                    assistant += delta_content
                if enable_think:
                    thinking_update = gr.update(value=thinking)
                    panel_update = gr.update(open=thinking_open)
                    container_update = gr.update(visible=thinking_seen or thinking_open)
                else:
                    thinking_update = gr.update(value="")
                    panel_update = gr.update(open=False)
                    container_update = gr.update(visible=False)
                yield (
                    history + [(message, assistant)],
                    session_id,
                    thinking_update,
                    panel_update,
                    container_update,
                )
    except requests.RequestException as err:
        assistant = f"[HTTP error] {err}"
        thinking_update = gr.update(value="")
        panel_update = gr.update(open=False)
        container_update = gr.update(visible=False)
        yield (
            history + [(message, assistant)],
            session_id,
            thinking_update,
            panel_update,
            container_update,
        )


def build_ui():
    with gr.Blocks(title="fermi-infer OpenAI Demo") as demo:
        gr.Markdown("# fermi-infer OpenAI Demo")

        session_state = gr.State("")

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Row():
                    session_box = gr.Textbox(label="Session ID", value="", interactive=False)
                chatbot = gr.Chatbot(label="Conversation", height=520)
                with gr.Column(visible=False) as thinking_container:
                    with gr.Accordion("Thinking (collapsed after <think>)", open=False) as thinking_panel:
                        thinking_box = gr.Textbox(
                            label="Thinking (streamed)",
                            value="",
                            lines=6,
                            interactive=False,
                        )
                msg = gr.Textbox(label="Message", placeholder="Say something...")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
                    new_session = gr.Button("New Session")

            with gr.Column(scale=1, min_width=240):
                with gr.Accordion("Settings", open=False):
                    api_base = gr.Textbox(label="OpenAI Base URL", value="http://127.0.0.1:8000")
                    api_key = gr.Textbox(label="API Key (optional)", value="", type="password")
                    system_prompt = gr.Textbox(
                        label="System Prompt",
                        value="你是一个乐于助人的助手。",
                        lines=3,
                    )
                    enable_think = gr.Checkbox(label="Enable thinking (<think>)", value=False)
                    max_new_tokens = gr.Slider(1, 9056, value=256, step=1, label="max_tokens")
                    temperature = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="temperature")
                    top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="top_p")

        def _submit(
            message,
            history,
            session_id,
            api_base,
            api_key,
            system_prompt,
            enable_think,
            max_new_tokens,
            temperature,
            top_p,
        ):
            for update in stream_generate(
                message,
                history,
                session_id,
                api_base,
                api_key,
                system_prompt,
                enable_think,
                int(max_new_tokens),
                float(temperature),
                float(top_p),
            ):
                yield update

        msg.submit(
            _submit,
            inputs=[
                msg,
                chatbot,
                session_state,
                api_base,
                api_key,
                system_prompt,
                enable_think,
                max_new_tokens,
                temperature,
                top_p,
            ],
            outputs=[chatbot, session_state, thinking_box, thinking_panel, thinking_container],
            api_name=False,
        )
        send.click(
            _submit,
            inputs=[
                msg,
                chatbot,
                session_state,
                api_base,
                api_key,
                system_prompt,
                enable_think,
                max_new_tokens,
                temperature,
                top_p,
            ],
            outputs=[chatbot, session_state, thinking_box, thinking_panel, thinking_container],
            api_name=False,
        )

        def _sync_session_box(session_id):
            return session_id or ""

        session_state.change(
            _sync_session_box, inputs=[session_state], outputs=[session_box], api_name=False
        )

        def _clear():
            return (
                [],
                "",
                gr.update(value=""),
                gr.update(open=False),
                gr.update(visible=False),
            )

        clear.click(
            _clear,
            outputs=[chatbot, session_state, thinking_box, thinking_panel, thinking_container],
            api_name=False,
        )

        def _new_session():
            return (
                [],
                "",
                gr.update(value=""),
                gr.update(open=False),
                gr.update(visible=False),
            )

        new_session.click(
            _new_session,
            outputs=[chatbot, session_state, thinking_box, thinking_panel, thinking_container],
            api_name=False,
        )

    return demo


def main():
    demo = build_ui()
    demo.queue()
    demo.launch(show_api=False)


if __name__ == "__main__":
    main()
