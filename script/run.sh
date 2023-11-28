#!/bin/zsh

# 假设这些 Python 脚本位于当前目录下，或者您可以在下面的命令中指定完整路径

echo "Running prepare_data.py..."
python ../src/prepare_data.py

echo "Running load_data.py..."
python ../src/load_data.py

echo "Running lora.py..."
python ../src/lora.py

echo "cover model to llama.cpp..."
python3 ~/src/llama.cpp convert.py --outfile ~/models/llamacpp/zy-f16.bin --outtype f16 /Users/you/src/llm_demo/outputs/zysft
~/src/llama.cpp/quantize  ~/models/llamacpp/zy-f16.bin ~/models/llamacpp/zysft-q4.bin q4_0
mv ~/models/llamacpp/zysft-q4.bin ~/text-generation-webui/models/

echo "All scripts executed successfully."
