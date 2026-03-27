import torch


def ste_quantize(x: torch.Tensor, num_bits: int = 16) -> torch.Tensor:
    """
    Bit precision control of Gaussian parameters using a straight-through estimator.
    Reference: https://arxiv.org/abs/1308.3432
    fare quantizzazione direttamente durante il training permette di aggiustare nei vari cicli
    anche l'errore della quantizzazione stesso, che altrimenti se allenassimo a 32 bit e poi 
    spostassimo a 8 bit solo alla fine del training, ogni gaussiana avrebbe un piccolo errore, e sommato
    darebbe un errore molto grosso su tutta l'immagine
    """
    # es per num_bits=8 va da [0,255] 
    qmin, qmax = 0, 2**num_bits - 1
    #trova valori minimi massimi di x e y 
    min_val, max_val = x.min().item(), x.max().item()
    scale = max((max_val - min_val) / (qmax - qmin), 1e-8)

    # Quantize in forward pass (non-differentiable)
    # (usa lo scale intero di 8 bit)
    q_x = torch.round((x - min_val) / scale).clamp(qmin, qmax)
    
    # dequantizza in "float" di nuovo, i valori saranno nel range giusto ma "a scalini"
    # es se quantizzazione trasforma sia 0.701 sia 0.702 in 178 
    # poi dequantizzati avranno stesso valore
    dq_x = q_x * scale + min_val 
    #la derivata di una funzione a gradini(funzione dequantizzata)
    # è vicina allo 0 quindi non imparerebbe nulla il modello
    #per evitare questo indichimo a pytorch di ignorare la derivata di (dq_x - x) (detach)
    # Restore gradients in backward pass
    dq_x = x + (dq_x - x).detach()
    return dq_x
