**NOTE: This repo is not my original work. This is based on deepcrack repo: https://github.com/qinnzou/DeepCrack. This is only intended for personal use of HKU Innovation Wing**

## Getting Started
1.  **Clone the repository:**
    ```bash
    cd ~
    git clone https://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@github.com/howardtse2005/Deepcrack.git
    cd Deepcrack/
    ```

2.  **Install dependencies:**
    The code was tested on Python 3.10 in Conda Environment
    ```bash
    conda create -n deepcrack python=3.10
    conda activate deepcrack
    conda install conda-forge::pytorch conda-forge::visdom conda-forge::opencv conda-forge::tqdm anaconda::numpy
    ```
3.  **Run visdom:**
    In a new terminal, run
    ```bash
    visdom
    ```
