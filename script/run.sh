#!/bin/zsh

# 假设这些 Python 脚本位于当前目录下，或者您可以在下面的命令中指定完整路径

echo "Running prepare_data.py..."
python ../src/prepare_data.py

echo "Running load_data.py..."
python ../src/load_data.py

echo "Running lora.py..."
python ../src/lora.py

echo "All scripts executed successfully."
