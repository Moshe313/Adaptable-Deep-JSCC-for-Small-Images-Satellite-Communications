# Adaptable Deep JSCC for Satellite Communications 🛰️

This repository contains the implementation and simulation of a simplified Deep Joint Source and Channel Coding (JSCC) system for small satellite (CubeSat) applications.  
It is based on the paper:  
**"Adaptable Deep Joint Source-and-Channel Coding for Small Satellite Applications" – Farsad, Goldhahn, Goldsmith, IEEE GLOBECOM 2022**

## 📄 Full Report

A detailed academic report describing the project's background, methodology, architecture, experiments, results, and conclusions is available here:

👉 [ADJSCC.pdf](./ADJSCC.pdf)

It includes:
- Full theoretical background on JSCC
- Problem formulation and system diagram
- Implementation details and training setup
- Quantitative and visual evaluation across SNR levels
- Comparison with baseline methods (Naive, JPEG)

## 📌 Project Goals

- Reproduce and simplify the JSCC model for constrained environments  
- Train models across different SNR levels using CIFAR-10  
- Evaluate JSCC performance under AWGN channels  
- Compare against naive transmission and JPEG-based methods

## 📁 Repository Structure

```
├── main.py                # Main training loop and experiment setup
├── JSCC.py                # Definition of encoder, decoder, and channel
├── model/                 # Trained model checkpoints for different SNRs
│   ├── model_snr_0.h5
│   ├── model_snr_2.h5
│   └── ...
├── images/                # Visualizations and plots
│   ├── system_diagram.png
│   ├── psnr_vs_snr.png
│   ├── jscc_vs_jpeg.png
│   └── JPEGsnr.png
├── ADJSCC.pdf             # Full LaTeX report of the project
├── README.md              # You are here :)
```

## 🚀 Usage
## 📦 Installation

To install the required Python dependencies, run:

```bash
pip install -r requirements.txt

### 🧠 Training a Model

To train a JSCC model at a specific SNR level, run:

```bash
python JSCC.py
```

Make sure to manually set the desired SNR value inside `JSCC.py`.  
The model will be trained on the CIFAR-10 dataset and saved under the `model/` directory as:

```
model/model_snr_{SNR}.h5
```

---

### 📊 Inference and Evaluation

To test the model on a custom image or compare across SNR levels:

1. Open `JSCC.py`  
2. Modify the path to the input image (e.g., replace with your own)  
3. Adjust the list of SNRs you'd like to test  
4. Run the script to generate PSNR plots and reconstructions

---

### ✅ Simple Inference Example

You can also use the `run_inference_jscc()` function directly:

```python
from JSCC import run_inference_jscc

psnr = run_inference_jscc("images/sample.png", snr=10)
print(f"PSNR at 10 dB: {psnr:.2f} dB")
```

Make sure the model for that SNR (`model/model_snr_10.h5`) has already been trained.  
Otherwise, run the training process first using `JSCC.py`.

---

## 🧠 Methodology

- **Encoder**: CNN that maps an image of shape 32×32×3 to a complex-valued vector of size k  
- **Channel**: AWGN layer simulating satellite noise  
- **Decoder**: CNN that reconstructs the image from a noisy vector (y = z + n)  
- **Loss**: Mean Squared Error (MSE)  
- **Normalization**: Encoder output is normalized to satisfy a unit power constraint

## 🧪 Evaluation

We trained separate models for SNRs in the range 0–20 dB and evaluated performance using:

- **PSNR**: Primary metric to compare quality vs. SNR  
- **Visual comparison**: JPEG vs. JSCC (see `images/JPEGsnr.png`)

Key finding:  
👉 Our simplified JSCC model achieves up to **20 dB PSNR improvement** over naive transmission in low-SNR environments, and outperforms JPEG when transmitted over noisy channels.

## 📊 Example Outputs

- `images/psnr_vs_snr.png`: JSCC vs. Naive across SNRs  
- `images/jscc_vs_jpeg.png`: Side-by-side image reconstructions  
- `images/JPEGsnr.png`: Large-scale comparison across multiple methods and SNR levels  

## 💬 Citation

If using this work, please cite the original paper:

```bibtex
@inproceedings{adjscc,
  author    = {Nir Farsad and Alexander Goldhahn and Andrea Goldsmith},
  title     = {Adaptable Deep Joint Source-and-Channel Coding for Small Satellite Applications},
  booktitle = {IEEE Global Communications Conference (GLOBECOM)},
  year      = {2022},
  doi       = {10.1109/GLOBECOM48099.2022.10000929}
}
```
