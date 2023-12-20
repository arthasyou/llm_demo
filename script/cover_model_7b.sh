#!/bin/zsh

echo "cover model to llama.cpp..."
python3 ~/src/llama.cpp/convert.py --outfile ~/models/llamacpp/zy7B-f16.bin --outtype f16 /Users/you/src/llm_demo/outputs/zypt_7B
~/src/llama.cpp/quantize  ~/models/llamacpp/zy13B-f16.bin ~/models/llamacpp/zy13B-q4.bin q4_0
mv ~/models/llamacpp/zy7B-q4.bin ~/text-generation-webui/models/
rm -rf ~/models/llamacpp/zy7B-f16.bin

echo "All scripts executed successfully."
