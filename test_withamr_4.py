import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm  # 导入 tqdm 进度条
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data, Batch as GeoBatch
from torch_geometric.nn import GCNConv, global_mean_pool
import amrlib
import penman
from torch.utils.data import DataLoader, Dataset

# 加了优化机制和缓存机制
# 无llm 只用了amr图加最基础的图对比学习
# ---------- 初始化 ----------
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("using device:", device)
print("Loading BERT model...")
bert_tokenizer = BertTokenizer.from_pretrained("new_models/bert-base-uncased")
bert_model = BertModel.from_pretrained("new_models/bert-base-uncased").to(device)
bert_model.eval()  # 设置为评估模式

print("Loading AMR parser...")
try:
    stog = amrlib.load_stog_model(model_dir="model_stog/myModels/model_parse_xfm_bart_base-v0_1_0")
    print("AMR parser loaded successfully.")
except:
    stog = None
    print("Failed to load AMR parser.")

RELATIONS = [
    "Message-Topic", "Cause-Effect", "Component-Whole", "Entity-Destination",
    "Entity-Origin", "Product-Producer", "Member-Collection",
    "Content-Container", "Instrument-Agency", "Other"
]

graph_model = None
node_embedding_cache = {}

def init_graph_model():
    """初始化图模型"""
    global graph_model
    if graph_model is None:
        graph_model = GraphContrastiveModel(input_dim=768, hidden_dim=128).to(device)
        graph_model.eval()
        # print("Graph model initialized.")
    return graph_model

# ---------- 数据集类 ----------
class RelationDataset(Dataset):
    """自定义数据集类，支持从磁盘加载 AMR 缓存"""
    def __init__(self, data, source="train", max_len=16, cache_dir="cache/amr_graphs"):
        self.data = data
        self.source = source  # 'train' or 'test'
        self.max_len = max_len
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.data)

    def get_amr_graph(self, sentence, idx):
        """从磁盘读取 AMR 缓存，如果不存在则返回 None"""
        cache_path = os.path.join(self.cache_dir, f"{self.source}_{idx}.amr")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    return penman.decode(content) if content.strip() else None
            except Exception as e:
                print(f"[AMR Load Error] idx={idx}: {e}")
        return None

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = " ".join(item["sentences"][0])
        ent_pairs = item["ner"]
        gold_relation = item["relations"][0][0][4] if item["relations"] else "Other"

        amr_graph = self.get_amr_graph(sentence, idx)
        graph_data = amr_to_graph(amr_graph)

        if graph_data:
            amr_vec, _ = get_graph_embeddings(graph_data)
            amr_vec = amr_vec.squeeze()
        else:
            amr_vec = torch.zeros(128)

        tokens = item["sentences"][0]
        (e1s, e1e, _), (e2s, e2e, _) = ent_pairs[0]
        ent1 = " ".join(tokens[e1s:e1e+1])
        ent2 = " ".join(tokens[e2s:e2e+1])

        with torch.no_grad():
            v1 = bert_model(**bert_tokenizer(ent1, return_tensors='pt').to(device))[0].mean(1).squeeze().cpu()
            v2 = bert_model(**bert_tokenizer(ent2, return_tensors='pt').to(device))[0].mean(1).squeeze().cpu()

        combined = torch.cat([v1, v2, amr_vec], dim=0)
        gold_relation_idx = RELATIONS.index(gold_relation)

        return combined, gold_relation_idx



# ---------- 图编码模块 ----------
class GraphContrastiveModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128):
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.gcn1(x, edge_index))
        x = F.relu(self.gcn2(x, edge_index))
        return global_mean_pool(x, batch)  # shape: [batch_size, hidden_dim]


def augment_graph(graph_data):
    # print("进入augment_graph函数")
    edge_mask = torch.rand(graph_data.edge_index.size(1)) > 0.2
    aug_edge_index = graph_data.edge_index[:, edge_mask]
    node_mask = torch.rand(graph_data.x.size(0)) > 0.15
    aug_x = graph_data.x.clone()
    aug_x[~node_mask] = 0
    # print("结束augment_graph函数")
    return Data(x=aug_x, edge_index=aug_edge_index)


def get_graph_embeddings(graph_list):
    global graph_model
    if graph_model is None:
        graph_model = init_graph_model()
    # print("进入get_graph_embeddings函数")
    """确保传入的是 Data 对象而不是元组"""
    # model = GraphContrastiveModel(input_dim=768, hidden_dim=128).to(device)         
    # 视图增强：保证每个图都是Data对象
    views1 = [augment_graph(g) for g in graph_list]
    views2 = [augment_graph(g) for g in graph_list]
    
    # 使用GeoBatch对多个图进行批处理
    batch1 = GeoBatch.from_data_list(views1).to(device)
    batch2 = GeoBatch.from_data_list(views2).to(device)
    
    emb1 = graph_model(batch1.x, batch1.edge_index, batch1.batch)
    emb2 = graph_model(batch2.x, batch2.edge_index, batch2.batch)
    
    sim = F.cosine_similarity(emb1, emb2, dim=1)
    avg_emb = (emb1 + emb2) / 2
    contrastive_losses = (1 - sim).tolist()
    # print("结束get_graph_embeddings函数")
    return avg_emb.cpu(), contrastive_losses



# ---------- AMR解析 ----------
def parse_to_amr(sentence):
    # print("进入parse_to_amr函数")
    """将句子解析为AMR图"""
    if stog is None:
        print("[AMR] Parser not available.")
        return None
    try:
        graphs = stog.parse_sents([sentence])
        # print(f"[AMR] Parsed sentence to AMR:\n{graphs[0]}")
        return penman.decode(graphs[0]) if graphs else None
    except:
        print("[AMR] Parsing failed.")
        # print("结束parse_to_amr函数")
        return None


def parse_to_amr_batch(sentences):
    # print("进入parse_to_amr_batch函数")
    """批量解析多个句子"""
    if stog is None:
        print("[AMR] Parser not available.")
        return [None] * len(sentences)
    try:
        graphs = stog.parse_sents(sentences)
        # print("[AMR] Parsed AMR batch:")
        # for g in graphs:
        #     print(g)
        # print("结束parse_to_amr_batch函数")
        return [penman.decode(g) if g else None for g in graphs]
    except:
        print("[AMR] Batch parsing failed.")
        # print("结束parse_to_amr_batch函数")
        return [None] * len(sentences)


def amr_to_graph(amr_graph):
    """将AMR图转换为GCN图"""
    if amr_graph is None:
        return None

    global node_embedding_cache
    node_labels = []
    node_map = {}
    
    # 1. 收集节点标签和索引映射
    for idx, (var, _, concept) in enumerate(amr_graph.instances()):
        node_map[var] = idx
        node_labels.append(concept)

    # 2. 批量处理未缓存的标签（修复KeyError的关键）
    unique_labels = list(set(node_labels))
    uncached_labels = [label for label in unique_labels if label not in node_embedding_cache]
    
    if uncached_labels:
        # 批量处理提高效率
        inputs = bert_tokenizer(
            uncached_labels, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=16
        ).to(device)
        
        with torch.no_grad():
            outputs = bert_model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu()
        
        # 更新缓存
        for i, label in enumerate(uncached_labels):
            node_embedding_cache[label] = embeddings[i]  # 直接存储张量

    # 3. 安全构建节点向量（防止残留KeyError）
    node_vecs = []
    for label in node_labels:
        if label in node_embedding_cache:
            node_vecs.append(node_embedding_cache[label])
        else:  # 兜底机制
            print(f"警告: 标签 '{label}' 未缓存，生成零向量")
            node_embedding_cache[label] = torch.zeros(768)  # BERT维度768
            node_vecs.append(node_embedding_cache[label])
    
    x = torch.stack(node_vecs)  # 直接堆叠张量

    # 4. 构建边
    edge_list = [
        [node_map[src], node_map[tgt]]
        for src, _, tgt in amr_graph.edges()
        if src in node_map and tgt in node_map
    ]

    if not edge_list:
        return None

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return [Data(x=x, edge_index=edge_index)]



# ---------- 关系分类模型 ----------
class RelationClassifier(nn.Module):
    def __init__(self, input_dim=768*2+128, hidden_dim=256, num_classes=10):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.fc(x)


# ---------- 训练与验证函数 ----------
def train(model, train_loader, optimizer, criterion):
    print("进入train函数")
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in tqdm(train_loader, desc="Training", ncols=100):
        optimizer.zero_grad()
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    print("结束train函数")
    return avg_loss, accuracy


def evaluate(model, val_loader):
    print("进入evaluate函数")
    model.eval()
    correct = 0
    total = 0
    for data in tqdm(val_loader, desc="Evaluating", ncols=100):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / total
    print("结束evaluate函数")
    return accuracy


# ---------- 主训练与评估流程 ----------
def main():
    # 加载数据
    with open("train.json", "r", encoding="utf-8") as f:
        train_data = [json.loads(line) for line in f]
    with open("test.json", "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    train_dataset = RelationDataset(train_data,source="train")
    test_dataset = RelationDataset(test_data,source="test")

    if graph_model is None:
        init_graph_model()  # 确保图模型已初始化
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = RelationClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    epochs = 4 
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_acc = evaluate(model, test_loader)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}")

    # 保存训练好的模型
    torch.save(model.state_dict(), "relation_classifier_model.pth")


if __name__ == "__main__":
    main()