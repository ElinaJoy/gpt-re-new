import os
import json
from tqdm import tqdm
from amrlib import load_stog_model
import penman

# 数据路径
DATA_FILES = ["./dataset/ace05/train.json", "./dataset/ace05/test.json"]
CACHE_DIR = "cache/amr_graphs/ace05"
os.makedirs(CACHE_DIR, exist_ok=True)

# 加载 AMR 解析器
print("Loading AMR parser...")
try:
    stog = load_stog_model(model_dir="model_stog/myModels/model_parse_xfm_bart_base-v0_1_0")
    print("AMR parser loaded successfully.")
except:
    stog = None
    print("Failed to load AMR parser.")
    exit()

def parse_to_amr_batch(sentences):
    try:
        graphs = stog.parse_sents(sentences)
        return [penman.decode(g) if g else None for g in graphs]
    except:
        return [None] * len(sentences)

def preprocess_amr_for_file(file_path):
    print(f"\nProcessing {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    file_path = file_path.split("/")[-1]  # 获取文件名
    sentences = [" ".join(item["sentences"][0]) for item in data]
    batch_size = 16

    for i in tqdm(range(0, len(sentences), batch_size), desc=f"Parsing {file_path}", ncols=100):
        batch_sentences = sentences[i:i+batch_size]
        graphs = parse_to_amr_batch(batch_sentences)

        for j, g in enumerate(graphs):
            idx = i + j
            cache_path = os.path.join(CACHE_DIR, f"{file_path.replace('.json','')}_{idx}.amr")
            if g:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write(penman.encode(g))
            else:
                # 标记解析失败的句子
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write("")

for path in DATA_FILES:
    preprocess_amr_for_file(path)

print("\n✅ AMR缓存预处理完成，结果保存在:", CACHE_DIR)
