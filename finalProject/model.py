#############################################################################
### Търсене и извличане на информация. Приложение на дълбоко машинно обучение
### Стоян Михов
### Зимен семестър 2022/2023
#############################################################################
###
### Невронен машинен превод
###
#############################################################################

import torch

class MultiHeadAttentionLayer(torch.nn.Module):
    def __init__(self, hidden_size, heads, device):
        super().__init__()
                
        self.hidden_size = hidden_size
        self.heads = heads
        self.head_size = hidden_size // heads
        
        self.fc_q = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_k = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc_v = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        
        self.fc_out = torch.nn.Linear(hidden_size, hidden_size)
        
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_size])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        # query = [batch size, query len, hid dim]
        # key = [batch size, key len, hid dim]
        # value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)  # [batch size, query len, hid dim]
        K = self.fc_k(key)    # [batch size, key len, hid dim]  
        V = self.fc_v(value)  # [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.heads, self.head_size).permute(0, 2, 1, 3)  # [batch size, n heads, query len, head dim]
        K = K.view(batch_size, -1, self.heads, self.head_size).permute(0, 2, 3, 1)  # [batch size, n heads, head dim, key len]  
        V = V.view(batch_size, -1, self.heads, self.head_size).permute(0, 2, 1, 3)  # [batch size, n heads, value len, head dim]
                
        attn_score = torch.matmul(Q, K) / self.scale  # [batch size, n heads, query len, key len]
        
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -float('inf'))
        
        attention = torch.nn.functional.softmax(attn_score, dim = 3)  # [batch size, n heads, query len, key len]

        #no attention dropout       
        x = torch.matmul(attention, V)  # [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).flatten(2,3)  # [batch size, query len, n heads, head dim] -> [batch size, query len, head dim * n heads]

                
        x = self.fc_out(x)  # [batch size, query len, hid dim]
        
        return x

class Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, layers, heads, pf_size, dropout, device, max_length = 1000):
        super().__init__()

        self.device = device
        
        self.word_embedding = torch.nn.Embedding(input_size, hidden_size)
        self.pos_embedding = torch.nn.Embedding(max_length, hidden_size)
        
        self.layers = torch.nn.ModuleList([EncoderLayer(hidden_size, heads, pf_size, dropout, device) for _ in range(layers)])
        
        self.dropout = torch.nn.Dropout(dropout)
        
        
    def forward(self, src, src_mask):
        
        # src = [batch size, src len]
        # src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)   # [batch size, src len]
        
        src = self.dropout((self.word_embedding(src)) + self.pos_embedding(pos))  # [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask) # [batch size, src len, hid dim]
            
        return src

class EncoderLayer(torch.nn.Module):
    def __init__(self, hidden_size, heads, pf_size, dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = torch.nn.LayerNorm(hidden_size)
        self.ff_layer_norm = torch.nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttentionLayer(hidden_size, heads, device)
        self.positionwise_feedforward = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, pf_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(pf_size, hidden_size), 
        )
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, src, src_mask):
        
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, 1, 1, src len] 
                
        _src = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))  # [batch size, src len, hid dim]

        _src = self.positionwise_feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))  # [batch size, src len, hid dim]
        
        return src

class Decoder(torch.nn.Module):
    def __init__(self, output_size, hidden_size, layers, heads, pf_size, dropout, device, max_length = 1000):
        super().__init__()
        
        self.device = device
        
        self.word_embedding = torch.nn.Embedding(output_size, hidden_size)
        self.pos_embedding = torch.nn.Embedding(max_length, hidden_size)
        
        self.layers = torch.nn.ModuleList([DecoderLayer(hidden_size, heads, pf_size, dropout, device) for _ in range(layers)])
        
        self.fc_out = torch.nn.Linear(hidden_size, output_size)
        
        self.dropout = torch.nn.Dropout(dropout)
        
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)  # [batch size, trg len] 
            
        trg = self.dropout((self.word_embedding(trg)) + self.pos_embedding(pos))  # [batch size, trg len, hid dim] 
        
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)  # [batch size, trg len, hid dim]
        
        output = self.fc_out(trg)  # [batch size, trg len, output dim] 
            
        return output

class DecoderLayer(torch.nn.Module):
    def __init__(self, hidden_size, heads, pf_size, dropout, device):
        super().__init__()
        
        self.self_attn_layer_norm = torch.nn.LayerNorm(hidden_size)
        self.enc_attn_layer_norm = torch.nn.LayerNorm(hidden_size)
        self.ff_layer_norm = torch.nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttentionLayer(hidden_size, heads, device)
        self.encoder_attention = MultiHeadAttentionLayer(hidden_size, heads, device)
        self.positionwise_feedforward = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, pf_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(pf_size, hidden_size), 
        )
        self.dropout = torch.nn.Dropout(dropout)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]
        
        _trg = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))  # [batch size, trg len, hid dim]

        _trg = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))  # [batch size, trg len, hid dim]
        
        _trg = self.positionwise_feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))  # [batch size, trg len, hid dim]
        
        
        return trg

class NMTmodel(torch.nn.Module):
    def preparePaddedBatch(self, source, word2ind):
        m = max(len(s) for s in source)
        sents = [[word2ind.get(w,self.unkTokenIdx) for w in s] for s in source]
        sents_padded = [ s+(m-len(s))*[self.padTokenIdx] for s in sents]
        return torch.tensor(sents_padded, dtype=torch.long, device=self.device)
    
    def save(self,fileName):
        torch.save(self.state_dict(), fileName)
    
    def load(self,fileName,device):
        self.load_state_dict(torch.load(fileName, map_location = device))

    def make_src_mask(self, src):
        return (src != self.padTokenIdx).unsqueeze(1).unsqueeze(2)   # [batch size, 1, 1, src len]
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.padTokenIdx).unsqueeze(1).unsqueeze(2)  # [batch size, 1, 1, trg len]
        trg_len = trg.shape[1] # trg.shape = (batch size, trg len)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()  # [trg len, trg len]
        return trg_pad_mask & trg_sub_mask  # [batch size, 1, trg len, trg len]
    
    def __init__(self, d_model,n_head,Nx,dropout, device, sourceWord2ind, targetWord2ind, startToken, unkToken, padToken, endToken):
        super(NMTmodel, self).__init__()
        self.device = device
        self.sourceWord2ind = sourceWord2ind
        self.targetWord2ind = targetWord2ind
        self.startTokenIdx = sourceWord2ind[startToken]
        self.unkTokenIdx = sourceWord2ind[unkToken]
        self.padTokenIdx = sourceWord2ind[padToken]
        self.endTokenIdx = sourceWord2ind[endToken]
        self.encoder = Encoder(len(sourceWord2ind),d_model,Nx,n_head,d_model*2,dropout,device)
        self.decoder = Decoder(len(targetWord2ind),d_model,Nx,n_head,d_model*2,dropout,device)

    def forward(self, src, trg):
        src_padded = self.preparePaddedBatch(src, self.sourceWord2ind)    # [batch size, src len]
        trg_padded = self.preparePaddedBatch(trg, self.targetWord2ind)    # [batch size, trg len]
        
        src_mask = self.make_src_mask(src_padded)               # [batch size, 1, 1, src len]      
        trg_mask = self.make_trg_mask(trg_padded[:, :-1])       # [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src_padded, src_mask)                                           # [batch size, src len, hid dim]
        output = self.decoder(trg_padded[:, :-1], enc_src, trg_mask, src_mask)      # [batch size, trg len, output dim]

            
        output = output.flatten(0,1)
        trg_padded = trg_padded[:,1:].flatten(0,1)

        H = torch.nn.functional.cross_entropy(output, trg_padded, ignore_index=self.padTokenIdx)

        return H

    def translateSentence(self, sentence, limit=1000):
        ind2word = dict(enumerate(self.targetWord2ind))
        tokens = [self.sourceWord2ind[w] if w in self.sourceWord2ind.keys() else self.unkTokenIdx for w in sentence]
        print(tokens)
        src = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        src_mask = self.make_src_mask(src)
        result = [self.startTokenIdx]

        with torch.no_grad():
            encoder_outputs = self.encoder(src, src_mask)

            for i in range(limit):
                trg = torch.tensor(result, dtype=torch.long, device=self.device).unsqueeze(0)

                trg_mask = self.make_trg_mask(trg)

                output = self.decoder(trg, encoder_outputs, trg_mask, src_mask)
                output = output[:, -1, :].squeeze() #take last token distribution

                output = torch.nn.functional.softmax(output,dim=0)
                                
                topk = output.topk(2).indices.tolist()

                pred_token = topk[0] if topk[0] != self.unkTokenIdx else topk[1] 
                result.append(pred_token)

                if pred_token == self.endTokenIdx:
                    break

        return [ind2word[i] for i in result[1:] if i != self.endTokenIdx]

    
