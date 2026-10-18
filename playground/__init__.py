"""
Interactive layer over the harvest-playground toolkit.

Everything UI-independent lives in plain modules here (mmlu, scorers,
benchmark, capture_studio, arena, models) so it can be tested and
scripted without Gradio; `app.py` is a thin Gradio front end on top.
"""
