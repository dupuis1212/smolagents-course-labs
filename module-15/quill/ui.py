"""Quill's web app — wrapping the agent in a ``GradioUI`` (Module 13, T12.1/T12.2).

Twelve modules in, Quill only runs as a script (``uv run python -m quill "..."``). That is fine
for *you*; it is useless for the colleague on the sales team who has the dataset and the question
but will not clone a repo, write a ``.env`` and type a Python command. This module turns Quill into
a **web app**: a chat where anyone uploads a CSV and asks a question.

**Three lines to a web app — ``GradioUI``.** ``GradioUI`` is smolagents' built-in Gradio interface
for a ``MultiStepAgent`` (it is built on ``gradio.ChatInterface``; it needs the ``[gradio]`` extra,
``gradio>=5.14.0`` as of smolagents 1.26.0). Its constructor is::

    GradioUI(agent, file_upload_folder=None, reset_agent_memory=False)

- ``agent`` — the agent to wrap. For Quill this is ALWAYS the output of ``build_quill()`` (the
  frozen construction entry point, ``quill/agent.py``). ``ui.py`` NEVER rebuilds the agent itself
  (06 §2 hard rule) — it imports and calls ``build_quill`` so the web Quill is byte-for-byte the
  Quill the tests exercise (manager + ``web_researcher`` + ``RetrieverTool``).
- ``file_upload_folder`` — the directory uploaded files land in. **If ``None``, upload is
  DISABLED.** Quill is a data analyst, so we pass ``"uploads"`` to UNLOCK CSV upload: the dataset
  arrives through the browser, not only by a hard-coded path.
- ``reset_agent_memory`` (default ``False``) — when ``True`` the agent's memory is wiped at the
  START of every interaction. We keep the default ``False`` so the chat REMEMBERS across turns
  (see below).

**Why the chat remembers: ``reset=False`` under the hood (T12.2 / T6.7).** With
``reset_agent_memory=False`` the UI runs each new message as ``agent.run(message, reset=False)`` —
the opposite of ``agent.run``'s own default (``reset=True``, which wipes memory). That is what makes
a chat a *chat*: Quill keeps the uploaded DataFrame and the previous question in memory, so "now
break that down by region" works without re-uploading. ⚠️ Common misconception: ``reset=False`` does
NOT make Quill re-think from scratch — it is the INVERSE, it KEEPS the context (which is why memory,
and cost, grow each turn → prune as in Module 6).

**The CSV upload trap (T12.1, the concrete point of this module).** ``GradioUI.upload_file``
defaults ``allowed_file_types=[".pdf", ".docx", ".txt"]`` — **``.csv`` is NOT in that list** (as of
smolagents 1.26.0), so a vanilla ``GradioUI`` REFUSES a CSV upload. For a data analyst that is
backwards. ``QuillGradioUI`` below widens the allow-list to ``[".csv", ".parquet", ".xlsx"]`` so
the dataset Quill exists to analyse can actually be uploaded. Once uploaded, Gradio writes the file
into ``file_upload_folder`` and appends its PATH to the conversation, so Quill's frozen
``load_dataset(path)`` (M3) can read it — the upload deposits a FILE whose path becomes available;
it does not inject a DataFrame.

We do NOT call ``.launch()`` at import time — only inside ``launch_ui`` / ``__main__``. ``.launch``
defaults ``share=True`` (a temporary public ``*.gradio.live`` tunnel); locally you usually want
``share=False`` (no public tunnel), which is this module's default.

Production note (T7, C7): a public agent is a threat vector. In prod Quill should run in a remote
sandbox (Approach 2 — Module 15), behind auth/rate limits, never in ``local``. The stop button on a
custom app (``stream_to_gradio`` + ``agent.interrupt()``, see ``build_custom_app``) is a minimal
guard. None of that is wired here beyond the hook — this module ships the UI; hardening is M15.
"""
from __future__ import annotations

from smolagents import GradioUI

from .agent import build_quill

# The file extensions Quill's uploader accepts. The smolagents GradioUI default is
# [".pdf", ".docx", ".txt"] (as of 1.26.0) — which has NO .csv — so a data analyst's UI would
# refuse its own dataset. We widen it to the tabular formats load_dataset (M3) understands plus
# .xlsx. This is the concrete, differentiating point of T12.1.
QUILL_ALLOWED_FILE_TYPES = [".csv", ".parquet", ".xlsx"]

# Where uploaded files land. GradioUI requires a NON-None folder to enable upload at all; we use a
# repo-local "uploads/" (shipped with a .gitkeep). load_dataset reads the deposited file by path.
DEFAULT_UPLOAD_FOLDER = "uploads"

__all__ = [
    "QUILL_ALLOWED_FILE_TYPES",
    "DEFAULT_UPLOAD_FOLDER",
    "QuillGradioUI",
    "make_ui",
    "launch_ui",
    "build_custom_app",
]


class QuillGradioUI(GradioUI):
    """A ``GradioUI`` that accepts CSV/Parquet/XLSX uploads (T12.1).

    The ONLY behavioural change from the stock ``GradioUI`` is the upload allow-list: the parent's
    ``upload_file`` defaults ``allowed_file_types=[".pdf",".docx",".txt"]``, which excludes ``.csv``.
    We override ``upload_file`` to pass Quill's tabular allow-list, so the dataset can be uploaded.
    Everything else (the ChatInterface, the ``reset=False`` continuation, the streaming) is the
    parent's — we wrap the agent, we do not reimplement the UI.
    """

    def upload_file(self, file, file_uploads_log, allowed_file_types=None):
        """Validate + store an upload, defaulting the allow-list to Quill's tabular formats.

        Mirrors ``GradioUI.upload_file`` but flips the default from the docs/txt list to
        ``QUILL_ALLOWED_FILE_TYPES`` (``.csv``/``.parquet``/``.xlsx``). A caller can still pass its
        own ``allowed_file_types`` to narrow/widen it. The stored file's PATH then becomes available
        to the agent, which reads it with ``load_dataset`` (the M3 frozen tool).
        """
        if allowed_file_types is None:
            allowed_file_types = QUILL_ALLOWED_FILE_TYPES
        return super().upload_file(file, file_uploads_log, allowed_file_types=allowed_file_types)


def make_ui(
    agent=None,
    *,
    file_upload_folder: str = DEFAULT_UPLOAD_FOLDER,
    reset_agent_memory: bool = False,
):
    """Build (do NOT launch) a ``QuillGradioUI`` wrapping ``agent`` (T12.1).

    This is the builder the offline tests use: it constructs the UI object and returns it WITHOUT
    starting a server, so the wiring (which agent, which upload folder, the ``reset_agent_memory``
    knob, the widened CSV allow-list) is assertable with no network and no ``.launch()``.

    Args:
        agent: the ``MultiStepAgent`` to wrap. ``None`` (the default) means "build the real Quill"
            via ``build_quill()`` — the ONLY place Quill is constructed (06 §2). Tests pass their
            own fake-model ``build_quill(model=fake_model([...]))`` so no LLM is called.
        file_upload_folder: where uploaded files land; must be non-``None`` to enable upload.
            Defaults to ``"uploads"``. Pass ``None`` to disable upload entirely (the stock default).
        reset_agent_memory: ``False`` (the default) keeps memory across turns — the UI runs
            ``agent.run(..., reset=False)`` so the chat remembers the uploaded dataset and prior
            questions (T12.2 / T6.7). ``True`` wipes memory each turn (one-shot per message).

    Returns:
        A ``QuillGradioUI`` instance, ready for ``.launch(share=...)``.
    """
    if agent is None:
        agent = build_quill()
    return QuillGradioUI(
        agent,
        file_upload_folder=file_upload_folder,
        reset_agent_memory=reset_agent_memory,
    )


def launch_ui(share: bool = False) -> None:
    """Build the real Quill web app and LAUNCH it (the ``python -m quill.ui`` entry point).

    Wraps ``build_quill()`` (manager + ``web_researcher`` + ``RetrieverTool``, fully wired) in a
    ``QuillGradioUI`` with CSV upload and multi-turn memory, then serves it. ``share`` defaults to
    ``False`` (no public tunnel) — pass ``True`` for a temporary ``*.gradio.live`` demo link.

    NOT called at import time (only by ``__main__``), so importing ``quill.ui`` never opens a port.
    """
    make_ui().launch(share=share)


def build_custom_app(agent, *, max_turns_label: str = "Quill"):
    """OPTIONAL custom Gradio app with a STOP button on ``agent.interrupt()`` (T12.2, "Try it").

    The stock ``GradioUI`` is clé-en-main but opaque. This shows the lower-level path the brief's
    "Try it yourself" asks for: drive the agent with the public ``stream_to_gradio`` helper (it runs
    ``agent.run(..., stream=True)`` under the hood and yields Gradio ``ChatMessage``s as steps land,
    accumulating ``ChatMessageStreamDelta``s and handling the ``FinalAnswerStep``), and wire a Stop
    button to ``agent.interrupt()`` — which stops the agent **at the end of its current step**, then
    raises (it does NOT abort mid-LLM-call or mid-code-execution; that is consistent with the "1 step
    = 1 LLM call → 1 action → 1 execution" model of Module 2). A stop button is a minimal guard for a
    public agent (C7).

    This builds and returns a ``gradio.Blocks`` WITHOUT launching it (the caller calls ``.launch``),
    so it stays importable/testable offline. ``gradio`` is imported lazily so importing ``quill.ui``
    does not require it at module load beyond ``GradioUI``'s own dependency.

    Args:
        agent: the ``MultiStepAgent`` to drive (pass ``build_quill()`` — never rebuild it here).
        max_turns_label: the chat title.

    Returns:
        A ``gradio.Blocks`` app (call ``.launch(share=...)`` on it).
    """
    import gradio as gr
    from smolagents import stream_to_gradio

    # Gradio 5.x requires Chatbot(type="messages"); Gradio 6 removed that parameter (mirrors
    # smolagents' own GradioUI.create_app version guard, as of smolagents 1.26.0).
    type_messages_kwarg = {"type": "messages"} if gr.__version__.startswith("5") else {}

    def _respond(message, history):
        # stream_to_gradio runs agent.run(message, stream=True) and yields ChatMessages as steps
        # complete; we accumulate them so the chat shows the trajectory live (T6.9 streaming).
        messages = []
        for msg in stream_to_gradio(agent, task=message, reset_agent_memory=False):
            messages.append(msg)
            yield messages

    with gr.Blocks(title=max_turns_label) as app:
        chatbot = gr.Chatbot(label=max_turns_label, **type_messages_kwarg)
        msg = gr.Textbox(label="Ask Quill", placeholder="Chart monthly revenue and tell me which category grew fastest")
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            stop = gr.Button("Stop", variant="stop")
        # Send streams the agent's steps into the chatbot.
        send.click(_respond, inputs=[msg, chatbot], outputs=chatbot)
        msg.submit(_respond, inputs=[msg, chatbot], outputs=chatbot)
        # Stop wires the Stop button to agent.interrupt() — stops at the end of the current step,
        # then raises (T1.14). The minimal public-agent guard.
        stop.click(lambda: agent.interrupt())
    return app


def main() -> None:
    """``python -m quill.ui`` — launch the Quill web app locally (no public tunnel)."""
    import sys

    share = "--share" in sys.argv[1:]
    launch_ui(share=share)


if __name__ == "__main__":
    main()
