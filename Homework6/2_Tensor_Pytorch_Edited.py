import torch
#----------------------------------------------------------------------------
dtype = torch.float
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
#----------------------------------------------------------------------------
Batch_size = 64     # Batch size
Q = 1000            # Input size
S = 100             # Number of neurons
a = 10              # Network output size
#----------------------------------------------------------------------------
p = torch.randn(Batch_size, Q, device=device, dtype=dtype)
t = torch.randn(Batch_size, a, device=device, dtype=dtype)
#----------------------------------------------------------------------------
w1 = torch.randn(Q, S, device=device, dtype=dtype)
w2 = torch.randn(S, a, device=device, dtype=dtype)
learning_rate = 1e-6
#----------------------------------------------------------------------------
for index in range(500):

    # h = p.mm(w1)
    # h_relu = h.clamp(min=0)
    # a_net = h_relu.mm(w2)
    a_0 = p
    n_1 = p.mm(w1)
    a_1 = n_1.clamp(min=0)
    a_2 = a_1.mm(w2)

    # loss = (a_net - t).pow(2).sum()
    # print(index, loss.item())
    loss = (a_2 - t).pow(2).sum()
    print(index, loss.item())

    # grad_y_pred = 2.0 * (a_net - t)
    # grad_w2 = h_relu.t().mm(grad_y_pred)
    # grad_h_relu = grad_y_pred.mm(w2.t())
    # grad_h = grad_h_relu.clone()
    # grad_h[h < 0] = 0
    # grad_w1 = p.t().mm(grad_h)
    s_2 = 2.0 * (a_2 - t)
    s_1 = s_2.mm(w2.t())
    s_1[n_1 < 0] = 0

    # w1 -= learning_rate * grad_w1
    # w2 -= learning_rate * grad_w2
    w1 = w1 - learning_rate * p.t().mm(s_1)
    w2 = w2 - learning_rate * a_1.t().mm(s_2)