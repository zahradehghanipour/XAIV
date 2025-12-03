lbs

```python
[
    torch.Size([1, 3, 32, 32]),
    torch.Size([1, 8, 16, 16]),
    torch.Size([1, 16, 8, 8]),
    torch.Size([1, 100]),
    torch.Size([1, 1])
]
```

ubs

```python
[
    torch.Size([1, 3, 32, 32]),
    torch.Size([1, 8, 16, 16]),
    torch.Size([1, 16, 8, 8]),
    torch.Size([1, 100]),
    torch.Size([1, 1])
]
```

duals

```python
[
    torch.Size([2048, 3]),
    torch.Size([1024, 3]),
    torch.Size([100, 3])
]
```

primals

```python
[
    torch.Size([2048]),
    torch.Size([2048]),
    torch.Size([1024]),
    torch.Size([1024]),
    torch.Size([1024]),
    torch.Size([100]),
    torch.Size([100]),
    torch.Size([1])
]
```

primal_input

```python
torch.Size([1, 3, 32, 32])
```

layers (props is a fused layer which maps the ground truth and adv class)

```python
{
    "fixed_layers": [
        Conv2d(3, 8, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
        ReLU(),
        Conv2d(8, 16, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
        ReLU(),
        Flatten(),
        Linear(in_features=1024, out_features=100, bias=True),
        ReLU(),
    ],
    "prop_layers": [Linear(in_features=100, out_features=1, bias=True)],
}
```

mask1d

```python
torch.Size([1, 3172])
```
