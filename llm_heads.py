import torch 




class SingleHead(torch.nn.Module):

    def __init__(self, n_embed_input, n_embed_output, t):
        super().__init__()
        
        self.key = torch.nn.Linear(n_embed_input, n_embed_output, bias = False)
        self.query = torch.nn.Linear(n_embed_input, n_embed_output, bias = False)
        self.value = torch.nn.Linear(n_embed_input, n_embed_output, bias= False)
        self.layer = torch.nn.LayerNorm(n_embed_output)
        self.t = t

        self.register_buffer('tril', torch.tril(torch.ones(t, t)))

    
    def forward(self, x):

        k = self.key(x)
        q = self.query(x)

        aff = q @ k.transpose(-2, -1)

        aff = aff.masked_fill((self.tril[:self.t, :self.t] == 0), value = float('-inf'))
        aff = torch.nn.functional.softmax(aff, dim = -1)

        out = aff @ self.value(x)

        out = self.layer(out)

        return out

class MultiHead(torch.nn.Module):

    def __init__(self, n_heads, n_embed, t):

        super().__init__()

        self.net = torch.nn.ModuleList([SingleHead(n_heads* n_embed, n_embed, t) for h in range(0, n_heads)])


    def forward(self, x):

        out = [h(x) for h in self.net]
        out = torch.cat(out, dim=-1)
        out = out + x
        return out

class FeedForward(torch.nn.Module):

    def __init__(self, n_embed):

        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embed, 4*n_embed), 
            torch.nn.ReLU(), 
            torch.nn.Linear(4*n_embed, n_embed),
            torch.nn.LayerNorm(n_embed)
        )

    def forward(self, x):

        out = x + self.net(x)
        return out

class BlockSH(torch.nn.Module):

    def __init__(self, n_embed, n_heads, t):

        super().__init__()

        self.net = torch.nn.Sequential(
            SingleHead(n_embed, t),
            SingleHead(n_embed,  t),
            FeedForward(n_embed)
        )

    def forward(self, x):
        out = self.net(x)
        return out

class BlockMH(torch.nn.Module):

    def __init__(self, n_embed, n_heads, t):

        super().__init__()

        self.net = torch.nn.Sequential(
            MultiHead(n_heads, n_embed//n_heads, t),
            MultiHead(n_heads, n_embed//n_heads, t),
            FeedForward(n_embed)
        )

    def forward(self, x):
        out = self.net(x)
        return out