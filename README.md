**NOTE: This is only intended for personal use of HKU Innovation Wing**

## Getting Started
1.  **Clone the repository:**
    ```bash
    cd ~
    git clone https://github.com/howardtse2005/Slope_Crack_Seg.git
    cd Deepcrack/
    ```

2.  **Install dependencies:**
    The code was tested on Python 3.10 in Conda Environment
    ```bash
    conda create -n deepcrack python=3.10
    conda activate deepcrack
    conda install conda-forge::pytorch conda-forge::visdom conda-forge::opencv conda-forge::tqdm anaconda::numpy conda-forge::einops
    pip install segformer-pytorch
    ```
3.  **Train on your own images**
    Put all your image-ground truth pairs in the data directory. Then, edit the train_example.txt (for training) and val_example.txt (for validation). The first column contains the path to the rgb image and the second column contains the path to ground truth mask.
    In the config.py, choose the preferred architecture (unet, deepcrack, hnet).
    Then, run
    ```bash
    python3 train.py
    ```
4. **Test on your own images**
    Put all your image-ground truth pairs in the data directory. Then, edit the test_example.txt. The first column contains the path to the rgb image and the second column contains the path to ground truth mask. If you do not have ground truth mask, puth the same path to the rgb image in the second column.
    In the config.py, choose the preferred architecture (unet, deepcrack, hnet).
    Then, put the path to the pretrained model (.pth file) in the test.py
    Then, run
    ```bash
    python3 test.py
    ```
