import numpy as np
import torch


def flatten(x):
    """
    Flattens a list of lists into a numpy array
    """
    out = []
    for episode in x:
        for item in episode:
            out.append(item)
    return np.array(out, dtype=np.float32).squeeze()


def tensor_flatten(x):
    """
    Flattens a list of lists of tensor with gradients to torch tensor with gradients
    """
    out = []
    for episode in x:
        for item in episode:
            out.append(item)
    return torch.stack(out).squeeze()


def get_lang_embedding(lang, tokenizer, model, device):
    """
    Embeds the given language instruction using the model and tokenizer
    and returns the CLS token embedding as a numpy array
    """
    lm_input = tokenizer(
        text=lang, add_special_tokens=True, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        lm_embeddings = model(
            lm_input["input_ids"],
            lm_input["attention_mask"],
        ).last_hidden_state

    ## return only CLS embedding as numpy array
    return lm_embeddings[:, 0, :].cpu().numpy().squeeze()
