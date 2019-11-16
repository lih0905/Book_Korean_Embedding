import torch
import torch.nn as nn
import torch.optim as optim
import time

VEC_DIM = 100 # 임베딩 차원
HIDDEN_DIM = 100 # 은닉 차원
N = 4 # n-gram의 갯수
BATCH_SIZE = 64 # 배치 사이즈
epoch = 10

dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 훈련 데이터 지정
txt_path = '../data/tokenized/korquad_mecab.txt'
corpus = [sent.strip().split() for sent in open(txt_path, 'r', encoding='utf-8').readlines()]

# 단어장 생성
vocab = set()
for i in range(len(corpus)):
    vocab.update(corpus[i])

# 단어를 인덱스화
word_to_id = dict()
for word in vocab:
    word_to_id[word]=len(word_to_id)

# n-gram 데이터 생성
def sent_to_ngram(n, sent):
    ngrams = []
    for i in range(len(sent) - n + 1):
        ngram = []
        for j in range(n):
            ngram.append(sent[i + j])
        ngrams.append(ngram)
    return ngrams

train_data = []
for i in range(len(corpus)):
    train_data += sent_to_ngram(N, corpus[i])

def sent_to_vec(sent):
    vec = []
    for word in sent:
        vec.append(word_to_id[word])
    return vec

train_input = []
train_target = []
for data in train_data:
    train_input.append(sent_to_vec(data[:-1]))
    train_target.append(word_to_id[data[-1]])

train_input_ts = torch.tensor(train_input)
train_target_ts = torch.tensor(train_target)

# 데이터 로더 객체 생성
train = torch.utils.data.TensorDataset(train_input_ts, train_target_ts)
input_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE)

# 손실 함수는 크로스 엔트로피
loss_f = nn.CrossEntropyLoss()

# 모델 클래스 생성
class NPLM(nn.Module):
    def __init__(self, n, vec_dim, hidden_dim):
        super(NPLM, self).__init__()
        self.emb = nn.Embedding(len(vocab), vec_dim)
        self.lin1 = nn.Linear(vec_dim * (n - 1), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, len(vocab), bias=False)
        self.lin3 = nn.Linear(vec_dim * (n - 1), len(vocab))

    def forward(self, x):
        # x : [BATCH_SIZE, N-1]
        # self.emb(x) : [BATCH_SIZE, N-1, VEC_DIM]
        x = self.emb(x).view(len(x), -1)
        y = torch.tanh(self.lin1(x))
        z = self.lin3(x) + self.lin2(y)
        return z

model = NPLM(N, VEC_DIM, HIDDEN_DIM)
model.to(dev)
optimizer = optim.Adam(model.parameters(), lr=0.01)

start = time.time()

loss_list = []

check = 5000
# 훈련 프로세스 시작
for epo in range(epoch):
    loss_sum = 0
    for i, (x, y) in enumerate(input_loader):
        x, y = x.to(dev), y.to(dev)
        optimizer.zero_grad()
        output = model(x)

        loss = loss_f(output, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        if i % check == 0:
            elap = int(time.time() - start)
            loss_list.append(loss_sum / check)
            print('Epoch : {}, Iteration : {}, Loss : {:.2f}, Elapsed time : {:.0f}h {:.0f}m {:0f}s'.format(\
                epo, i, loss_sum / check, elap // 3600, (elap % 3600) // 60, (elap % 3600) % 60))
            loss_sum = 0

save_path = 'NPLM_{}epoch.pt'.format(epoch)
torch.save(model.state_dict(), save_path)

model = NPLM(N, VEC_DIM, HIDDEN_DIM)
model.load_state_dict(torch.load(save_path))
model.eval()

def similar_words(model, word, k=10):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    word_id = torch.tensor([word_to_id[word]])
    word_vec = model.emb(word_id)
    word_mat = next(iter(model.emb.parameters())).detach()

    cos_mat = cos(word_vec, word_mat)
    sim, indices = torch.topk(cos_mat, k + 1)

    word_list = []
    for i in indices:
        if i != word_id:
            word_list.append(id_to_word[i])
    return word_list, sim[1:].detach()