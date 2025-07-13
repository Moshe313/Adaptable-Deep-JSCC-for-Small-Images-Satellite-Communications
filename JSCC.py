# JSCC Simulation from Scratch on CIFAR-10 with AWGN Channel

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Configuration
IMG_SHAPE = (32, 32, 3)
SNR_VALUES = list(range(0, 21, 2))
BATCH_SIZE = 64
EPOCHS = 10  # For full runs use higher value
CHANNEL_DIM = 128

# 1. Load CIFAR-10
def load_data():
    (x_train, _), (x_test, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    return x_train, x_test

# 2. AWGN Layer
class AWGN(tf.keras.layers.Layer):
    def __init__(self, snr_db=10, **kwargs):
        super().__init__(**kwargs)
        self.snr_db = snr_db

    def call(self, inputs, training=False):
        signal_power = tf.reduce_mean(tf.square(inputs))
        snr_linear = tf.pow(10.0, self.snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise = tf.random.normal(shape=tf.shape(inputs), stddev=tf.sqrt(noise_power))
        return inputs + noise if training else inputs

    def get_config(self):
        config = super().get_config()
        config.update({"snr_db": self.snr_db})
        return config


# 3. Build Encoder-Channel-Decoder Model
def build_model(snr_db):
    input_img = tf.keras.Input(shape=IMG_SHAPE)
    x = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(input_img)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(CHANNEL_DIM, 3, activation='relu', padding='same')(x)

    x = AWGN(snr_db)(x)  # Simulated channel

    x = tf.keras.layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.UpSampling2D()(x)
    output = tf.keras.layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)

    return tf.keras.Model(input_img, output)

# 4. PSNR Calculation
def compute_psnr(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return 10.0 * tf.math.log(1.0 / mse) / tf.math.log(10.0)

# 5. Train and Evaluate
def train_and_evaluate(snr_db, x_train, x_test):
    model = build_model(snr_db)
    model.compile(optimizer='adam', loss='mse')
    model.fit(x_train, x_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    preds = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    model.save(f"model/jscc_model_snr_{snr_db}.h5")
    return compute_psnr(x_test, preds).numpy()

# 6. Naive Baseline Transmission through AWGN
def evaluate_naive_baseline(x_test, snr_db):
    x = tf.convert_to_tensor(x_test)
    signal_power = tf.reduce_mean(tf.square(x))
    snr_linear = tf.pow(10.0, snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = tf.random.normal(shape=tf.shape(x), stddev=tf.sqrt(noise_power))
    x_noisy = x + noise
    x_noisy = tf.clip_by_value(x_noisy, 0.0, 1.0)
    mse = tf.reduce_mean(tf.square(x - x_noisy))
    psnr = 10.0 * tf.math.log(1.0 / mse) / tf.math.log(10.0)
    return psnr.numpy()

# 7. Main Simulation
def main():
    x_train, x_test = load_data()
    jscc_psnr_results = []
    naive_psnr_results = []

    print("\n[🚀] Starting JSCC simulation...")
    for snr in SNR_VALUES:
        print(f"[🔁] SNR = {snr} dB → Training JSCC model")
        jscc_psnr = train_and_evaluate(snr, x_train, x_test)
        naive_psnr = evaluate_naive_baseline(x_test, snr)
        print(f"[✅] JSCC PSNR: {jscc_psnr:.2f} dB, Naive PSNR: {naive_psnr:.2f} dB")
        jscc_psnr_results.append(jscc_psnr)
        naive_psnr_results.append(naive_psnr)

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(SNR_VALUES, jscc_psnr_results, marker='o', label='JSCC')
    plt.plot(SNR_VALUES, naive_psnr_results, marker='x', linestyle='--', label='Naive Transmission')
    plt.title("PSNR vs SNR for JSCC vs Naive on CIFAR-10")
    plt.xlabel("SNR (dB)")
    plt.ylabel("PSNR (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("psnr_vs_snr_jscc_vs_naive.png")
    plt.show()

if __name__ == "__main__":
    main()
