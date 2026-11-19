"""Same evaluation math; avoid a redundant dense-logit copy and unused loss."""
import math
import torch
import torch.nn.functional as F

def cuda_batch(batch):
    return {k:v.to(device='cuda',dtype=torch.long if k in ('input_ids','labels') else v.dtype).contiguous()
            for k,v in batch.items()}


def _dense_loss_and_correct(logits, labels, chunk_rows):
    """Bound FP32 workspace while retaining the original per-batch NLL reduction."""
    selected = torch.empty((labels.numel(), 1), device=logits.device, dtype=torch.float32)
    targets = torch.zeros(labels.numel(), device=labels.device, dtype=torch.long)
    correct = torch.zeros((), device=logits.device, dtype=torch.long)
    rows = labels.shape[1]
    for sample in range(labels.shape[0]):
        for first in range(0, rows, chunk_rows):
            end = min(first + chunk_rows, rows)
            # Slice within each sample: reshaping the shifted 3-D view would
            # otherwise copy the entire vocabulary-sized tensor.
            block = logits[sample, first:end].float()
            expected = labels[sample, first:end]
            correct.add_((block.argmax(-1) == expected).sum())
            log_probs = F.log_softmax(block, dim=-1)
            selected[sample*rows+first:sample*rows+end].copy_(
                log_probs.gather(1, expected.unsqueeze(1)))
            del block, log_probs
    # Do not sum per-chunk losses: that changes floating-point reduction order.
    # Keep the original number/order of rows and the same public NLL sum kernel.
    return F.nll_loss(selected, targets, reduction='sum'), correct


def _evaluate(model,records,counts,*,skip_unused_loss,loss_chunk_rows=None):
    # Check on the supplied CPU records once per panel. Non-dense targets use
    # the original selection path, preserving its exact reduction order.
    dense=bool((records['labels'][:max(counts),1:].ne(-100)
                & records['attention_mask'][:max(counts),1:].bool()).all())
    prior=model.training
    model.eval()
    nll,tokens,correct=0.,0,0
    panels={}
    try:
        with torch.inference_mode():
            for first in range(0,max(counts),2):
                batch=cuda_batch({k:v[first:first+2] for k,v in records.items()})
                inputs={k:v for k,v in batch.items() if k!='labels'} if skip_unused_loss else batch
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(**inputs,use_cache=False)
                labels=batch['labels'][:,1:]
                valid=labels.ne(-100)&batch['attention_mask'][:,1:].bool()
                if dense and loss_chunk_rows is not None:
                    loss, batch_correct = _dense_loss_and_correct(output.logits, labels, loss_chunk_rows)
                else:
                    logits=output.logits[:,:-1].float()
                    if dense:
                        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                                             labels.reshape(-1),reduction='sum')
                    else:
                        mask=valid.reshape(-1)
                        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1])[mask],
                                             labels.reshape(-1)[mask],reduction='sum')
                    batch_correct=((logits.argmax(-1)==labels)&valid).sum()
                    del logits
                nll+=float(loss)
                tokens+=int(valid.sum())
                correct+=int(batch_correct)
                del output,loss,batch_correct
                if not math.isfinite(nll):raise FloatingPointError('Nonfinite evaluation')
                if first+2 in counts:
                    panels[first+2]=dict(loss=nll/tokens,ppl=math.exp(nll/tokens),
                                         accuracy=correct/tokens,tokens=tokens)
    finally:
        model.train(prior)
    return panels


def evaluate(model,records,counts,*,loss_chunk_rows=64):
    """Evaluate unchanged model batches; None disables chunked loss workspace."""
    if not counts or any(not isinstance(n,int) or n<=0 or n%2 or n>len(records["labels"]) for n in counts):
        raise ValueError("Evaluation panel counts must be positive even row counts within the supplied records")
    if loss_chunk_rows is not None and (type(loss_chunk_rows) is not int or loss_chunk_rows <= 0):
        raise ValueError("loss_chunk_rows must be a positive integer or None")
    return _evaluate(model,records,counts,skip_unused_loss=True,loss_chunk_rows=loss_chunk_rows)


def copy_only(model,records,counts):
    return _evaluate(model,records,counts,skip_unused_loss=False)
