#!/bin/bash

# junten_bidsフォルダ内のすべてのsubjectからevents.tsvファイルをコピーするスクリプト

# コピー先ディレクトリを作成
mkdir -p copied_events

echo "events.tsvファイルのコピーを開始します..."

# すべてのsubjectフォルダをループ
for subject_dir in junten_bids/sub-js*/; do
    if [ -d "$subject_dir" ]; then
        subject_name=$(basename "$subject_dir")
        echo "処理中: $subject_name"
        
        # ieegフォルダ内のevents.tsvファイルを検索してコピー
        if [ -d "${subject_dir}ieeg" ]; then
            find "${subject_dir}ieeg" -name "*_events.tsv" -exec cp {} copied_events/ \;
            echo "  ${subject_name}のevents.tsvファイルをコピーしました"
        else
            echo "  警告: ${subject_name}のieegフォルダが見つかりません"
        fi
    fi
done

echo "完了: copied_eventsフォルダにすべてのevents.tsvファイルをコピーしました"
echo "コピーされたファイル数: $(ls copied_events/*.tsv 2>/dev/null | wc -l)"
