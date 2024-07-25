import torch
import torch.nn as nn
import torch.nn.functional as F

text = open("input.txt").read()
chars = sorted(list(set(text)))
itos = {i:s for i,s in enumerate(chars)}
stoi = {s:i for i,s in enumerate(chars)}
vocab_size = len(chars)

encode = lambda s : [stoi[c] for c in s]
decode = lambda s : "".join([itos[c] for c in s])

data = encode(text)
n = int(len(data) * 0.9)
train_data = data[:n]
var_data = data[n:]

device = "cuda" if torch.cuda.is_available() else "cpu"
block_size = 256  # 最长上下文长度
batch_size = 64
vocab_size = len(chars)
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

def get_batch(split):
    data = train_data if split == "train" else var_data
    ix = torch.randint(0,len(data) - block_size, (batch_size,))
    x = [data[i:i + block_size] for i in ix]
    y = [data[i+1:i + block_size + 1] for i in ix]
    x = torch.tensor(x).to(device)
    y = torch.tensor(y).to(device)
    return x,y

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.head_size_sqrt = head_size**-0.5
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        weight = k @ q.transpose(-2,-1) * self.head_size_sqrt
        weight = weight.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        weight = weight.softmax(-1)
        weight = self.dropout(weight)
        out = weight @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(num_head*head_size, num_head*head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, fan_in, fan_out):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(fan_in, 4 * fan_out),
            nn.ReLU(),
            nn.Linear(4 * fan_out, fan_out),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        out = self.nn(x)
        return out

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, head_size)
        self.fforward = FeedForward(n_embd,n_embd)
        self.layer_nom1 = nn.LayerNorm(n_embd)
        self.layer_nom2 = nn.LayerNorm(n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x + self.sa_heads(self.layer_nom1(x))
        out = x + self.fforward(self.layer_nom2(x))
        out = self.dropout(out)
        return out

class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)]
        )
        self.layer_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)   # B,T,C  4 * 8 * 65  batch, block, n_embed
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # T * C
        x_emb = tok_emb + pos_emb
        x = self.blocks(x_emb)
        x = self.layer_norm(x)
        logits = self.lm_head(x)

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)  # logits的第二个维度为通道数 C
        return logits, loss

    def generate(self, idx, max_generate_tokens):
        for _ in range(max_generate_tokens):
            idx_use = idx[:, -block_size:]
            logits, loss = self(idx_use)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=-1)
        return idx

model = BigramLanguageModel().to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        loss_s = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            loss_s[k] = loss
        out[split] = loss_s.mean()
    model.train()
    return out

# 训练
optim = torch.optim.Adam(model.parameters(), lr=0.01)
batch_size = 64
eval_iter = 50
max_iter = 1001
eval_intval = 100
for _ in range(max_iter):
    if iter % eval_intval == 0:
        losses = estimate_loss()
        print(iter, losses)
    iter += 1
        
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optim.zero_grad()
    loss.backward()
    optim.step()