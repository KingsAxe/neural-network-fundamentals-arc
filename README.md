# CNN Challange: Neural Networks from First Principles

> *"True understanding comes not from using frameworks, but from building them."*

This repository contains my solution to the AI & Devices technical assessment challenge focused on implementing neural networks from fundamental principles using only NumPy.

##  Challenge Overview
The assessment evaluates ability to:
- **Debug bare-bones neural network implementations** (fix 10 critical bugs in MLP)
- **Implement core CNN components from scratch** (Conv2D, MaxPooling, Flatten)
- **Train models for ARC-inspired grid pattern recognition**
- **Analyze architectural decisions and inductive biases**

##  What This Project Teaches
| Concept | Practical Insight |
|---------|-------------------|
| **Gradient Flow** | How broken backpropagation silently kills learning |
| **Numerical Stability** | Why softmax needs max-subtraction to avoid NaNs |
| **CNN Inductive Biases** | Why locality/equivariance matter for grid reasoning |
| **Debugging Methodology** | Systematic approach to neural network forensics |
| **ARC Reasoning** | Bridging abstract pattern recognition and deep learning |

##  Getting Started
1. Navigate to `notebook/` directory
2. Open `neural-network-fundamentals.ipynb` in Jupyter Lab/Colab
3. Run cells sequentially to see:
   - Bug-fixing walkthrough with diagnostics
   - CNN component implementations with unit tests
   - Training visualizations and filter analysis
   - Critical analysis of architecture choices

##  Documentation
- **Detailed implementation notes**: See `docs/COMPREHENSIVE_README.md`
- **Key insights**: Why certain bugs caused specific failure modes
- **ARC adaptation strategies**: How to extend this to grid-to-grid transformations

##  Key Achievement
 **92% validation accuracy** on symmetry classification task  
All CNN components validated with unit tests  
Critical analysis of CNN limitations for ARC tasks  

*This project demonstrates deep understanding of neural network mechanics beyond framework usage - essential for tackling novel AI challenges like ARC.*