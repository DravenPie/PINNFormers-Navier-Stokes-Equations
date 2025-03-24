# PINNs vs PINNsFormer for Navier-Stokes Equations: Transient Lid-Driven Cavity Flow

## Introduction

This repository contains code to compare the performance of Physics-Informed Neural Networks (PINNs) and PINNsFormer for solving a special case of the Navier-Stokes equations: the transient lid-driven cavity flow problem. The lid-driven cavity flow is a classic benchmark problem in computational fluid dynamics (CFD), and this project aims to explore how traditional PINNs and the more recent PINNsFormer (Physics-Informed Transformer) perform in this context.

The benchmark was created by Ghia et al. (1982). However, since that paper addresses the stationary version of the problem and Transformers (and PINNsFormer by extension) excel at sequential data, we added a time dependency and initial conditions. Additionally, we provide a paper that describes how PINNsFormer incorporates many ideas from the original Transformers and how these innovations improve PINNsFormer's ability to handle sequential data.

## Instructions to Execute the Notebooks

### 1. Clone the Repository

Clone the repository to your local machine and open its root directory:
```bash
git clone https://github.com/your-username/PINNFormers-Navier-Stokes-Equations.git
cd PINNFormers-Navier-Stokes-Equations
```

### 2. Set Up the Environment

Ensure you have Python 3.8 or later installed. Then:
#### a. Create a Virtual Environment for Python3
```bash
python3 -m venv env
```

#### b. Activate the Virtual Environment
```bash
source env/bin/activate
```

#### c. Install Dependencies
If you have a requirements.txt file:
```bash
pip install -r requirements.txt
```
If you don't have one, create it and install:
```bash
pip freeze > requirements.txt
pip install -r requirements.txt
```

#### d. Open and Run the Notebooks
Launch your preferred IDE, such as [Visual Studio Code](https://code.visualstudio.com/download), [Jupyter Notebook](https://jupyter.org/install), or any other IDE that supports .ipynb files. Then, run the notebooks directly from the IDE.

### Extra commands:

List all installed packages:
```bash
pip list
```

Deactivate the virtual environment:
```bash
deactivate
```

## Instructions to Download the Article
The paper is available at: [PINNFormers for Navier Stokes Equations](./PINNFormers_for_Navier_Stokes_Equations.pdf)

You can download it from the GitHub webpage or use the following command:
```bash
wget https://github.com/DravenPie/PINNFormers-Navier-Stokes-Equations/blob/main/PINNFormers_for_Navier_Stokes_Equations.pdf
```

## Authors

- **[@ginbar - Gabriel Inácio Barboza](https://github.com/ginbar)**
- **[@DravenPie - Vinicius Nunes Pereira](https://github.com/DravenPie)**

## License

This project is licensed under the [MIT License](./LICENSE). See the LICENSE file for details.

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