import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.io import imread
from skimage.transform import resize
from JSCC import AWGN

SNR_VALUES = list(range(0, 21, 2))
MODEL_DIR = "model"
INPUT_SHAPE = (32, 32, 3)
IMAGE_PATH = "1.jpg"

def add_awgn_noise(image, snr_db):
    signal_power = np.mean(image ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), image.shape)
    noisy_image = np.clip(image + noise, 0, 1)
    return noisy_image

def jpeg_compress_decompress(image, quality):
    image_bgr = (image * 255).astype(np.uint8)
    success, encoded = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        raise ValueError("JPEG compression failed")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    decoded = cv2.resize(decoded, (INPUT_SHAPE[1], INPUT_SHAPE[0]))
    return decoded, len(encoded)

def run_inference_jscc(image, snr_db):
    model_path = os.path.join(MODEL_DIR, f"jscc_model_snr_{snr_db}.h5")
    model = load_model(model_path, custom_objects={"AWGN": AWGN}, compile=False)
    input_image = np.expand_dims(image, axis=0)
    reconstructed = model.predict(input_image)[0]
    return reconstructed

def evaluate_all():
    image = imread(IMAGE_PATH)
    image = resize(image, INPUT_SHAPE, anti_aliasing=True).astype(np.float32)

    original_size_bytes = image.size  # מספר ערכים = פיקסלים * ערוצי צבע
    print(f"Original uncompressed size (float32): {original_size_bytes * 4} bytes")  # float32 = 4 bytes

    psnr_jscc = []
    psnr_naive = []
    psnr_jpeg_10 = []
    psnr_jpeg_50 = []
    psnr_jpeg_96 = []

    jpeg_ratios = {}

    for snr in SNR_VALUES:
        noisy_image = add_awgn_noise(image, snr)
        psnr_naive.append(compute_psnr(image, noisy_image))

        jscc_out = run_inference_jscc(noisy_image, snr)
        psnr_jscc.append(compute_psnr(image, jscc_out))

        jpeg_10, size_10 = jpeg_compress_decompress(image, 10)
        jpeg_50, size_50 = jpeg_compress_decompress(image, 50)
        jpeg_96, size_96 = jpeg_compress_decompress(image, 96)

        jpeg_ratios["Q10"] = (original_size_bytes * 4) / size_10
        jpeg_ratios["Q50"] = (original_size_bytes * 4) / size_50
        jpeg_ratios["Q96"] = (original_size_bytes * 4) / size_96

        jpeg_10_noisy = add_awgn_noise(jpeg_10, snr)
        jpeg_50_noisy = add_awgn_noise(jpeg_50, snr)
        jpeg_96_noisy = add_awgn_noise(jpeg_96, snr)

        psnr_jpeg_10.append(compute_psnr(image, jpeg_10_noisy))
        psnr_jpeg_50.append(compute_psnr(image, jpeg_50_noisy))
        psnr_jpeg_96.append(compute_psnr(image, jpeg_96_noisy))

    return SNR_VALUES, psnr_jscc, psnr_naive, psnr_jpeg_10, psnr_jpeg_50, psnr_jpeg_96, jpeg_ratios

def plot_results(snr, jscc, naive, jpeg_10, jpeg_50, jpeg_96, jpeg_ratios):
    label_10 = f"JPEG compressed by x{jpeg_ratios['Q10']:.1f}"
    label_50 = f"JPEG compressed by x{jpeg_ratios['Q50']:.1f}"
    label_96 = f"JPEG compressed by x{jpeg_ratios['Q96']:.1f}"

    plt.plot(snr, jscc, label="JSCC", marker='o')
    plt.plot(snr, naive, label="Naive (AWGN only)", marker='*')
    plt.plot(snr, jpeg_10, label=label_10, marker='x')
    plt.plot(snr, jpeg_50, label=label_50, marker='s')
    plt.plot(snr, jpeg_96, label=label_96, marker='^')
    plt.xlabel("SNR [dB]")
    plt.ylabel("PSNR [dB]")
    plt.title("PSNR vs SNR with JPEG Compression Ratios")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def show_example_images(image, snr_values, jpeg_ratios):
    qualities = [10, 50, 96]
    snrs_to_show = [0, 8, 16]

    fig, axs = plt.subplots(len(snrs_to_show), 6, figsize=(18, 10))
    for j in range(1, 6):
        axs[0, j].axis('off')

    for i, snr in enumerate(snrs_to_show):
        row = i

        noisy = add_awgn_noise(image, snr)
        jscc_out = run_inference_jscc(noisy, snr)

        jpeg_10, _ = jpeg_compress_decompress(image, 10)
        jpeg_50, _ = jpeg_compress_decompress(image, 50)
        jpeg_96, _ = jpeg_compress_decompress(image, 100)

        jpeg_10_noisy = add_awgn_noise(jpeg_10, snr)
        jpeg_50_noisy = add_awgn_noise(jpeg_50, snr)
        jpeg_96_noisy = add_awgn_noise(jpeg_96, snr)

        axs[row, 0].imshow(noisy)
        axs[row, 0].set_title(f"Naive SNR={snr}")
        axs[row, 1].imshow(jpeg_10_noisy)
        axs[row, 1].set_title(f"JPEG compressed by x{jpeg_ratios['Q10']:.1f}")
        axs[row, 2].imshow(jpeg_50_noisy)
        axs[row, 2].set_title(f"JPEG compressed by x{jpeg_ratios['Q50']:.1f}")
        axs[row, 3].imshow(jpeg_96_noisy)
        axs[row, 3].set_title(f"JPEG compressed by x{jpeg_ratios['Q96']:.1f}")
        axs[row, 4].imshow(jscc_out)
        axs[row, 4].set_title("JSCC")
        axs[row, 5].imshow(image)
        axs[row, 5].set_title("Original (Ref)")

        for j in range(6):
            axs[row, j].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    snr, jscc, naive, jpeg_10, jpeg_50, jpeg_96, ratios = evaluate_all()
    plot_results(snr, jscc, naive, jpeg_10, jpeg_50, jpeg_96, ratios)

    img = imread(IMAGE_PATH)
    img = resize(img, INPUT_SHAPE, anti_aliasing=True).astype(np.float32)
    show_example_images(img, snr, ratios)
