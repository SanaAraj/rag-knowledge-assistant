# RAG Knowledge Assistant demo - regenerate with:  vhs docs/demo.tape
# Run from the repo root with the venv created (uv venv), GROQ_API_KEY in .env,
# and data/ already indexed (python main.py --ingest).
Output docs/demo.gif

Set Shell "bash"
Set FontSize 17
Set Width 1240
Set Height 900
Set Padding 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 22ms
Set WindowBar Colorful

# ---- hidden setup: venv, offline+quiet model load, clean prompt ----
Hide
Type "source .venv/bin/activate"
Enter
Type "export HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
Enter
Type 'python(){ command python "$@" 2>/dev/null; }'
Enter
Type "export PS1='$ '"
Enter
Type "clear"
Enter
Show

Type "# Two PDFs (machine learning + neural networks) are already indexed."
Enter
Type "# Start an interactive session that cites its sources:"
Enter
Type "python main.py --chat"
Enter
Sleep 10s

Type "What is supervised learning?"
Sleep 300ms
Enter
Sleep 5s

Type "How does backpropagation work?"
Sleep 300ms
Enter
Sleep 6s

Type "What is the capital of France?"
Sleep 300ms
Enter
Sleep 4s

Enter
Sleep 1.2s
