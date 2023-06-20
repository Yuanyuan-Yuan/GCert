# GCert
Research Artifact of USENIX Security 2023 Paper: *Precise and Generalized Robustness Certification for Neural Networks*

Preprint: https://arxiv.org/pdf/2306.06747.pdf


## Installation

- Build from source code

    ```setup
    git clone https://github.com/Yuanyuan-Yuan/GCert
    cd GCert
    pip install -r requirements.txt
    ```

## Structure

This repo is organized as follows:

- `implementation` - This folder provides implementations and examples of regulating
generative models with continuity and independence. See detailed documents [here](https://github.com/Yuanyuan-Yuan/GCert/tree/main/implementation)

- `experiments` - This folder provides scripts of our evaluations. See detailed documents [here](https://github.com/Yuanyuan-Yuan/GCert/tree/main/experiments)

- `frameworks` - GCert is incorporated into three conventional certification frameworks (i.e.,
AI$^2$/Eran, GenProver, and ExactLine). This folder provides the scripts for configurations; see
detailed documents [here](https://github.com/Yuanyuan-Yuan/GCert/tree/main/frameworks)

- `data` - This folder provides scripts for data processing and shows examples of some data samples. See detailed documents [here](https://github.com/Yuanyuan-Yuan/GCert/tree/main/data).
