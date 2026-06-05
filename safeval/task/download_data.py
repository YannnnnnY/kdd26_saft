from datasets import load_dataset

gsm8k = load_dataset('gsm8k')
math = load_dataset('HuggingFaceH4/MATH-500')
hexphi= load_dataset('Y-AI-R/HEx-PHI')
advbench= load_dataset('walledai/AdvBench')
beavertails= load_dataset('PKU-Alignment/BeaverTails', name="default")