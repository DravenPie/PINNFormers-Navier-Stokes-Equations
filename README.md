# PINNs vs PINNsFormer for Navier-Stokes Equations: Transient Lid-Driven Cavity Flow

This repository contains code to compare the performance of Physics-Informed Neural Networks (PINNs) and PINNsFormer for solving a special case of the Navier-Stokes equations, the transient lid-driven cavity flow problem. The lid-driven cavity flow is a classic benchmark problem in computational fluid dynamics (CFD), and this project aims to explore how traditional PINNs and the more recent PINNsFormer (Physics-Informed Transformer) perform in this context. 

The benchmark was created by Ghia et al. (1982), but since that paper deals with the stationary version of the problem and Transforms (and PINNsFormer by extension) exceed at sequential data, we added a time-dependence and initial conditions. We also provide a paper that describes how PINNsFormer incoporated many ideas from the original Transformers and how they have improved PINNsFormer's hability to better deal with sequencial data.

## Instructions to Execute the Notebooks

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/your-username/PINNFormers-Navier-Stokes-Equations.git
```
### 2. Set Up the Environment

Ensure you have Python 3.8 or later installed. Then:
#### a. Create a virtual environment for python3
```bash
python3 -m venv env
```

#### b. Activate the virtual environment
```bash
source env/bin/activate
```

#### c. Create a requirements.txt file
```bash
pip freeze > requirements.txt
```

#### d. Install all the packages from the requirements.txt file
```bash
pip install -r requirements.txt
```

#### d. Open the notebooks
Open the repository in [a]{https://code.visualstudio.com/download}

Extras:

If you want to list all the installed packages in the virtual environment:
```bash
pip list
```

If you want to deactivate the virtual environment:
```bash
deactivate
```

## Autores

- **[ginbar](https://github.com/ginbar)**
- **[@DravenPie](https://github.com/DravenPie)**

## Licença

Este projeto está sob a licença [MIT](./LICENSE).

## Bibliography

Below are the key references and resources used in this project:

- Physics-Informed Neural Networks (PINNs):

    Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707. DOI:10.1016/j.jcp.2018.10.045

- PINNsFormer:

    Zhao, Z., Ding, X. & Prakash, B. A. (2024). PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks. https://arxiv.org/abs/2307.11833v3

- Lid-Driven Cavity Flow:

    Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for incompressible flow using the Navier-Stokes equations and a multigrid method. Journal of Computational Physics, 48(3), 387–411. DOI:10.1016/0021-9991(82)90058-4

- Attention Mechanism and Transformers:

    Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. Advances in Neural Information Processing Systems (NeurIPS), 30, 5998–6008. arXiv:1706.03762