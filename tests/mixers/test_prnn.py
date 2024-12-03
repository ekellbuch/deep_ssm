"""
Test the pRNN model implementation, it should be able to replicate the GRU implementation.

One full run can be tested with:
  python run.py --config-name="baseline_gru" trainer_cfg.fast_dev_run=1 model_cfg.configs.layer_dim=1
  python run.py --config-name="prnn" trainer_cfg.fast_dev_run=1 model_cfg.configs.layer_dim=1 model_cfg.configs.parallel=false

"""
import torch
import torch.nn as nn
from deep_ssm.mixers.prnn import pRNN


def test_prnn():
    
    class Arguments:
        def __init__(self, 
                 input_size=10, 
                 hidden_size=10, 
                 num_layers=1, 
                 batch_first=True, 
                 dropout=0, 
                 bidirectional=True, 
                 num_iters=2,
                 method="gru", 
                 parallel=False):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.batch_first = batch_first
            self.dropout = dropout
            self.bidirectional = bidirectional
            self.method = method
            self.parallel = parallel
            self.num_iters = num_iters

    # Create an instance of Arguments
    cfg = Arguments()
    D = 2 if cfg.bidirectional else 1
    hidden_size = cfg.hidden_size * D


    # Pass arguments to pRNN
    torch.manual_seed(42)
    prnn = pRNN(**vars(cfg))

    batch, seqlen, dim = 1, 10, 10

    x = torch.randn(batch, seqlen, dim)
    output, hidden = prnn(x)

    assert output.shape == (batch, seqlen, hidden_size)
    assert hidden.shape == (D, batch, cfg.hidden_size)

    torch.manual_seed(42)
    gru_model = nn.GRU(cfg.input_size,
                       cfg.hidden_size, 
                       cfg.num_layers, 
                       batch_first=cfg.batch_first, 
                       dropout=cfg.dropout, 
                       bidirectional=cfg.bidirectional)
    output1, hidden1 = gru_model(x)

    assert output1.shape == (batch, seqlen, hidden_size)
    assert hidden1.shape == (D, batch, cfg.hidden_size)

    gru_params = sum(p.numel() for p in gru_model.parameters())
    prnn_params = sum(p.numel() for p in prnn.parameters())
    
    assert gru_params == gru_params
    assert torch.allclose(output, output1)
    assert torch.allclose(hidden, hidden1)

    # Test backward pass
    loss_fn = nn.MSELoss()

    # Compute loss and gradients for pRNN
    prnn_loss = loss_fn(output, torch.zeros_like(output))
    prnn_loss.backward()
    prnn_grads = [p.grad.clone() for p in prnn.parameters()]

    # Compute loss and gradients for GRU
    gru_loss = loss_fn(output1, torch.zeros_like(output1))
    gru_loss.backward()
    gru_grads = [p.grad.clone() for p in gru_model.parameters()]

    # Compare gradients
    for prnn_grad, gru_grad in zip(prnn_grads, gru_grads):
        assert torch.allclose(prnn_grad, gru_grad, atol=1e-5), "Gradient mismatch between pRNN and GRU"


    # now check convergence runs with fixed number of iterations
    cfg2 = Arguments(parallel=True, num_iters=10)

    torch.manual_seed(42)
    prnn2 = pRNN(**vars(cfg2))
    output2, hidden2 = prnn2(x)

    assert output2.shape == (batch, seqlen, hidden_size)
    assert hidden2.shape == (D, batch, cfg.hidden_size)

    #print((output2- output).abs().max())
    assert torch.allclose(output, output2)
    assert torch.allclose(hidden, hidden2)
    print("All tests passed")

    # Compute loss and gradients for pRNN2
    prnn2_loss = loss_fn(output2, torch.zeros_like(output2))
    prnn2_loss.backward()
    prnn2_grads = [p.grad.clone() for p in prnn2.parameters()]

    # Compare gradients
    for prnn2_grad, gru_grad in zip(prnn2_grads, gru_grads):
        assert torch.allclose(prnn2_grad, gru_grad, atol=1e-5), "Gradient mismatch between pRNN and GRU"


if __name__ == "__main__":
    test_prnn()
