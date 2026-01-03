#!/usr/bin/env python3
"""
Debug script to check parameter requires_grad state in the model pipeline.
"""

import torch
from transformers import AutoModel, AutoTokenizer
import sys
sys.path.insert(0, '/Users/ayushkumartalreja/Downloads/Thesis_2/hindi-babylm')

from src.evaluation.indicglue_evaluator import MultipleChoiceWrapper

# Load IndicBERT
print("Loading IndicBERT...")
base_model = AutoModel.from_pretrained('ai4bharat/indic-bert')
print(f"Base model loaded")

# Check initial state
print("\n1. Base model parameter state (just loaded):")
total = 0
requires_grad_true = 0
requires_grad_false = 0
for name, param in base_model.named_parameters():
    total += 1
    if param.requires_grad:
        requires_grad_true += 1
    else:
        requires_grad_false += 1
print(f"   Total: {total}, requires_grad=True: {requires_grad_true}, requires_grad=False: {requires_grad_false}")

# Create MultipleChoiceWrapper
print("\n2. Creating MultipleChoiceWrapper...")
wrapped = MultipleChoiceWrapper(
    base_model=base_model,
    hidden_size=768,
    num_choices=4,
    pooling_strategy='first'
)
print(f"   Wrapped model created")

# Check wrapped model state
print("\n3. Wrapped model parameter state:")
total = 0
requires_grad_true = 0
requires_grad_false = 0
base_params = 0
classifier_params = 0

for name, param in wrapped.named_parameters():
    total += 1
    if param.requires_grad:
        requires_grad_true += 1
    else:
        requires_grad_false += 1

    if 'base_model' in name:
        base_params += 1
    elif 'classifier' in name:
        classifier_params += 1

print(f"   Total: {total}, requires_grad=True: {requires_grad_true}, requires_grad=False: {requires_grad_false}")
print(f"   Base model params: {base_params}, Classifier params: {classifier_params}")

# Now enable gradients for classifier only (mimicking evaluate_indicbert.py)
print("\n4. Enabling gradients for classifier only...")
for param in wrapped.classifier.parameters():
    param.requires_grad = True

# Check state after enabling classifier gradients
print("\n5. Parameter state after enabling classifier gradients:")
total = 0
requires_grad_true = 0
requires_grad_false = 0
base_true = 0
base_false = 0
classifier_true = 0
classifier_false = 0

for name, param in wrapped.named_parameters():
    total += 1
    if param.requires_grad:
        requires_grad_true += 1
        if 'base_model' in name:
            base_true += 1
        elif 'classifier' in name:
            classifier_true += 1
    else:
        requires_grad_false += 1
        if 'base_model' in name:
            base_false += 1
        elif 'classifier' in name:
            classifier_false += 1

print(f"   Total: {total}, requires_grad=True: {requires_grad_true}, requires_grad=False: {requires_grad_false}")
print(f"   Base model: True={base_true}, False={base_false}")
print(f"   Classifier: True={classifier_true}, False={classifier_false}")

# Create optimizer (mimicking fine_tuning_manager._create_optimizer)
print("\n6. Creating optimizer...")
no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in wrapped.named_parameters()
                  if not any(nd in n for nd in no_decay) and p.requires_grad],
        'weight_decay': 0.01
    },
    {
        'params': [p for n, p in wrapped.named_parameters()
                  if any(nd in n for nd in no_decay) and p.requires_grad],
        'weight_decay': 0.0
    }
]

optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=2e-5)

print(f"   Optimizer created: AdamW with {len(optimizer_grouped_parameters[0]['params'])} "
      f"params with decay, {len(optimizer_grouped_parameters[1]['params'])} without decay")
print(f"   Total params in optimizer: {len(optimizer_grouped_parameters[0]['params']) + len(optimizer_grouped_parameters[1]['params'])}")

# Test forward pass and gradient flow
print("\n7. Testing gradient flow...")
wrapped.train()

# Create dummy input
batch_size = 2
num_choices = 4
seq_len = 128

input_ids = torch.randint(0, 200000, (batch_size, num_choices, seq_len))
attention_mask = torch.ones((batch_size, num_choices, seq_len))
labels = torch.randint(0, 4, (batch_size,))

# Forward pass
outputs = wrapped(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
loss = outputs.loss

print(f"   Loss: {loss.item():.4f}")

# Backward pass
loss.backward()

# Check if gradients were computed
print("\n8. Gradient state after backward():")
base_has_grad = 0
base_no_grad = 0
classifier_has_grad = 0
classifier_no_grad = 0

for name, param in wrapped.named_parameters():
    if param.grad is not None and param.grad.abs().sum() > 0:
        if 'base_model' in name:
            base_has_grad += 1
        elif 'classifier' in name:
            classifier_has_grad += 1
    else:
        if 'base_model' in name:
            base_no_grad += 1
        elif 'classifier' in name:
            classifier_no_grad += 1

print(f"   Base model: has_grad={base_has_grad}, no_grad={base_no_grad}")
print(f"   Classifier: has_grad={classifier_has_grad}, no_grad={classifier_no_grad}")

# Step optimizer
print("\n9. Stepping optimizer...")
optimizer.step()
print("   Optimizer step completed")

print("\n✅ Debug script completed")
