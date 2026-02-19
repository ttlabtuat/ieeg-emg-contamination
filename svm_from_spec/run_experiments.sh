#!/bin/bash

# 実験を実行するPythonスクリプトのパス
PYTHON_SCRIPT="src/run_experiment.py"

# 参加者YAMLファイルのベースディレクトリ
JS_YAML_DIR="data/komeiji_js_yamls"

# 実験用のYAMLファイル
EXP_YAML_FILE="conf/conf_2600/spec280_350.yaml"

JS_NAME=("js01" "js02" "js04" "js05" "js07" "js08" "js11" "js13" "js14" "js15" "js16")
echo "Running experiments for all subjects..."
for subject in "${JS_NAME[@]}"; do
    echo "Processing $subject..."
    echo $PYTHON_SCRIPT --js_yaml "$JS_YAML_DIR/$subject.yaml" --exp_yaml "$EXP_YAML_FILE"
    python $PYTHON_SCRIPT --js_yaml "$JS_YAML_DIR/$subject.yaml" --exp_yaml "$EXP_YAML_FILE"
done