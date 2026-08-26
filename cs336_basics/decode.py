from cs336_basics.model import softmax
import torch
def sample_next_token(logits, temperature, top_p):
    next_logits=logits[:,-1,:]
    next_logits=next_logits/temperature
    probs=softmax(next_logits,-1) #沿着 vocabulary 那个维度计算概率
    sorted_probs, sorted_indices = torch.sort(probs,descending=True,dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative_probs - sorted_probs <= top_p
    sorted_probs = sorted_probs.masked_fill(~mask, 0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled_index = torch.multinomial(sorted_probs, 1)
    next_token = torch.gather(sorted_indices,1,sampled_index)

    return next_token

def decoding(model,tokenizer,prompt,max_new_tokens,temperature=1.0,top_p=1.0,):
    token_ids = tokenizer.encode(prompt)

    device = next(model.parameters()).device

    tokens = torch.tensor([token_ids],dtype=torch.long,device=device,)

    eos_token_id = tokenizer.vocab_reverse[b"<|endoftext|>"]

    for _ in range(max_new_tokens):
        logits = model(tokens)
        next_token = sample_next_token(logits,temperature,top_p,)
        next_token_id = next_token.item()
        token_ids.append(next_token_id)
        tokens = torch.cat([tokens, next_token],dim=1,)
        if next_token_id == eos_token_id:
            break

    return tokenizer.decode(token_ids)
    


