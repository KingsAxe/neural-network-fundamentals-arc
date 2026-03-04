# Building a CNN for Grid Pattern Recognition: Implementation & Analysis

##  Table of Contents
1. [Part 1: MLP Debugging Deep Dive](#part-1-mlp-debugging-deep-dive)
2. [Part 2: CNN Implementation & Analysis](#part-2-cnn-implementation--analysis)
3. [Critical Architecture Analysis](#critical-architecture-analysis)
4. [ARC Task Adaptation Strategies](#arc-task-adaptation-strategies)
5. [Lessons Learned & Best Practices](#lessons-learned--best-practices)

---

## Part 1: MLP Debugging Deep Dive

###  Critical Bugs Fixed & Why They Mattered
| # | Component | Bug | Impact | Fix |
|---|-----------|-----|--------|-----|
| 1 | `Activation_ReLU.forward()` | `np.minimum(0, inputs)` | Killed all activations → zero learning | `np.maximum(0, inputs)` |
| 2 | `Activation_Softmax.forward()` | In-place modification (`inputs -= max`) | Corrupted logits for backprop | Copy before modification |
| 3 | `Optimizer_SGD.update_params()` | Missing negative sign in gradient update | Performed gradient *ascent* → loss exploded | Added `-` before learning rate |
| 4 | `Loss_CategoricalCrossentropy.backward()` | Gradients not normalized by batch size | 300× oversized gradients → instability | Divided by `samples` |
| 5 | `calculate_accuracy()` | Wrong axis for argmax on predictions | Accuracy always 33.3% (random guess) | Fixed axis parameter |

###  Debugging Methodology
```python
# Critical diagnostic probes added during training
if DEBUG_MODE and epoch % print_every == 0:
    print(f" -→ Dead ReLUs: Layer 1: {dead_relu1:.1f}%, Layer 2: {dead_relu2:.1f}%")
    print(f" -→ Max Gradient Magnitude: Layer 3: {max_grad_d3:.4f}, Layer 1: {max_grad_d1:.4f}")