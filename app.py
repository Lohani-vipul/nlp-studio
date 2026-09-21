import torch
import gradio as gr
from transformers import pipeline

DEVICE = 0 if torch.cuda.is_available() else -1

CACHE = {}
TASKS = {
    "Sentiment Analysis": {
        "task": "sentiment-analysis",
        "model": "distilbert-base-uncased-finetuned-sst-2-english",
    },
    "Text Generation": {
        "task": "text-generation",
        "model": "distilgpt2",
    },
    "Translation (EN → FR)": {
        "task": "translation_en_to_fr",
        "model": "Helsinki-NLP/opus-mt-en-fr",
    },
    "Summarization": {
        "task": "summarization",
        "model": "sshleifer/distilbart-cnn-12-6",
    },
    "Named Entity Recognition": {
        "task": "ner",
        "model": "dslim/bert-base-NER",
        "extra": {"aggregation_strategy": "simple"},
    },
}

NER_COLORS = {
    "PER": "#c4b5fd",
    "ORG": "#93c5fd",
    "LOC": "#6ee7b7",
    "MISC": "#fcd34d",
}


def get_pipe(label: str):
    if label not in CACHE:
        spec = TASKS[label]
        CACHE[label] = pipeline(
            spec["task"],
            model=spec["model"],
            device=DEVICE,
            **spec.get("extra", {}),
        )
    return CACHE[label]


def highlight_ner(text: str, ents: list) -> str:
    spans = sorted(
        [e for e in ents if e.get("start") is not None and e.get("end") is not None],
        key=lambda e: e["start"],
        reverse=True,
    )
    out = text
    for e in spans:
        color = NER_COLORS.get(e.get("entity_group", "MISC"), "#e5e7eb")
        piece = out[e["start"] : e["end"]]
        badge = (
            f'<mark style="background:{color};color:#0b0f19;padding:1px 4px;'
            f'border-radius:4px;font-weight:600">{piece}'
            f'<span style="font-size:10px;opacity:.75;margin-left:4px">'
            f'{e.get("entity_group", "")} {e.get("score", 0):.0%}</span></mark>'
        )
        out = out[: e["start"]] + badge + out[e["end"] :]
    return f'<p style="line-height:1.8;font-size:16px">{out}</p>'


def run(text: str, label: str):
    text = (text or "").strip()
    if not text:
        return "<p>Type something first.</p>"

    pipe = get_pipe(label)

    if label == "Sentiment Analysis":
        r = pipe(text)[0]
        tone = "#34d399" if r["label"] == "POSITIVE" else "#f87171"
        return (
            f'<div style="padding:16px;border-radius:12px;background:#111827">'
            f'<div style="font-size:13px;letter-spacing:.12em;color:#9ca3af">SENTIMENT</div>'
            f'<div style="font-size:28px;font-weight:700;color:{tone};margin:6px 0">{r["label"]}</div>'
            f'<div style="height:8px;background:#1f2937;border-radius:99px;overflow:hidden">'
            f'<div style="width:{r["score"]*100:.1f}%;height:100%;background:{tone}"></div></div>'
            f'<div style="margin-top:8px;color:#d1d5db">confidence {r["score"]:.1%}</div></div>'
        )

    if label == "Text Generation":
        r = pipe(
            text,
            max_new_tokens=80,
            do_sample=True,
            temperature=0.9,
            truncation=True,
        )[0]
        return (
            f'<p style="font-size:16px;line-height:1.7;white-space:pre-wrap">'
            f'{r["generated_text"]}</p>'
        )

    if label == "Translation (EN → FR)":
        r = pipe(text, max_length=256)[0]
        return (
            f'<div style="font-size:13px;letter-spacing:.12em;color:#9ca3af">FRENCH</div>'
            f'<p style="font-size:20px;line-height:1.6">{r["translation_text"]}</p>'
        )

    if label == "Summarization":
        r = pipe(text, max_length=120, min_length=20, do_sample=False)[0]
        return (
            f'<div style="font-size:13px;letter-spacing:.12em;color:#9ca3af">SUMMARY</div>'
            f'<p style="font-size:18px;line-height:1.7">{r["summary_text"]}</p>'
        )

    ents = pipe(text)
    if not ents:
        return "<p>No named entities found.</p>"
    chips = "".join(
        f'<span style="display:inline-block;margin:4px 6px 0 0;padding:4px 8px;'
        f'border-radius:999px;background:{NER_COLORS.get(e.get("entity_group"), "#e5e7eb")};'
        f'color:#0b0f19;font-size:12px">{e.get("word")} · {e.get("entity_group")} '
        f'{e.get("score", 0):.0%}</span>'
        for e in ents
    )
    return highlight_ner(text, ents) + f'<div style="margin-top:12px">{chips}</div>'


EXAMPLES = [
    ["I waited two hours and the food was still cold. Never coming back.", "Sentiment Analysis"],
    ["Once upon a time in a quiet coastal town,", "Text Generation"],
    ["Hugging Face makes it easy to ship transformer models to production.", "Translation (EN → FR)"],
    [
        "Transformers have become the default architecture for NLP. "
        "They use self-attention so the model can weigh every token against every other token, "
        "which is why they handle long-range dependencies better than RNNs. "
        "Pretrained checkpoints let you fine-tune on small datasets and still get strong results.",
        "Summarization",
    ],
    [
        "Sundar Pichai announced Gemini at Google I/O in Mountain View, California.",
        "Named Entity Recognition",
    ],
]

CSS = """
.gradio-container { max-width: 880px !important; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="NLP Studio",
    theme=gr.themes.Soft(primary_hue="violet", neutral_hue="slate"),
    css=CSS,
) as demo:
    gr.Markdown(
        """
        # NLP Studio
        Hugging Face **pipelines** — sentiment, generation, translation, summarization, NER.
        The first run of each task downloads that model, then it stays cached.
        """
    )
    text = gr.Textbox(
        label="Input",
        placeholder="Paste a sentence, paragraph, or prompt…",
        lines=6,
    )
    task = gr.Radio(list(TASKS), value="Sentiment Analysis", label="Task")
    run_btn = gr.Button("Run pipeline", variant="primary")
    result = gr.HTML()
    gr.Examples(EXAMPLES, inputs=[text, task], label="Try an example")
    run_btn.click(run, inputs=[text, task], outputs=result)
    text.submit(run, inputs=[text, task], outputs=result)

if __name__ == "__main__":
    demo.launch()
