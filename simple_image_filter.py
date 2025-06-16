from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import io
import base64
import re
import cv2  # Menggunakan OpenCV untuk pemrosesan yang cepat dan efisien

# Inisialisasi aplikasi Flask
app = Flask(__name__, template_folder='templates')

# --- Fungsi Utilitas Gambar ---

def decode_image(base64_string):
    """Mendekode string base64 menjadi gambar OpenCV (format BGR)."""
    # Menghapus header data URL jika ada (cth: "data:image/png;base64,")
    img_data = re.sub('^data:image/.+;base64,', '', base64_string)
    img_bytes = base64.b64decode(img_data)
    
    # Membaca data byte menjadi array numpy
    nparr = np.frombuffer(img_bytes, np.uint8)
    # Mendekode array numpy menjadi gambar OpenCV. Format default adalah BGR.
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def encode_image_to_base64(cv_image):
    """Meng-encode gambar OpenCV (format BGR) menjadi string base64 untuk ditampilkan di web."""
    # Encode gambar ke format PNG di dalam memori
    # cv2.imencode bekerja dengan format BGR secara default, jadi tidak perlu konversi warna.
    success, buffer = cv2.imencode(".png", cv_image)
    if not success:
        raise ValueError("Gagal meng-encode gambar ke PNG.")
    
    # Konversi buffer ke byte dan encode ke base64
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('ascii')
    
    # Kembalikan sebagai data URL
    return f"data:image/png;base64,{img_base64}"

# --- Fungsi Pemrosesan Gambar ---

def process_image(image, filter_type, brightness, contrast, saturation):
    """Memproses gambar dengan filter dan penyesuaian yang dipilih."""
    # Buat salinan untuk menghindari modifikasi gambar asli
    processed = image.copy()

    # 1. Terapkan FILTER utama
    if filter_type == 'grayscale':
        # Konversi ke Grayscale dan kembali ke BGR agar penyesuaian warna tetap bisa diterapkan
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    elif filter_type == 'sepia':
        # FIX: Menggunakan cv2.transform untuk performa yang jauh lebih baik
        sepia_kernel = np.array([
            [0.272, 0.534, 0.131],
            [0.349, 0.686, 0.168],
            [0.393, 0.769, 0.189]
        ])
        processed = cv2.transform(processed, sepia_kernel)

    elif filter_type == 'invert':
        processed = cv2.bitwise_not(processed)

    elif filter_type == 'blur':
        processed = cv2.GaussianBlur(processed, (9, 9), 0)

    elif filter_type == 'sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        processed = cv2.filter2D(processed, -1, kernel)

    elif filter_type == 'edge':
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        processed = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif filter_type == 'emboss':
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        processed = cv2.filter2D(processed, -1, kernel)

    # 2. Terapkan PENYESUAIAN (adjustments) setelah filter
    
    # FIX: Logika Kecerahan (Brightness) dan Kontras (Contrast) yang lebih baik
    if brightness != 0 or contrast != 0:
        # Kontras (alpha) dan Kecerahan (beta) dapat diterapkan dalam satu operasi
        alpha = 1.0 + (contrast / 100.0) # Faktor untuk kontras
        beta = float(brightness)         # Nilai absolut untuk kecerahan
        
        # Menggunakan cv2.addWeighted untuk kontrol yang lebih baik, atau convertScaleAbs
        # processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)
        # Mari kita terapkan secara manual untuk memastikan rentang nilai 0-255
        processed = np.clip(alpha * processed.astype(np.float32) + beta, 0, 255).astype(np.uint8)

    # Saturasi (Saturation)
    if saturation != 0 and len(processed.shape) == 3:
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        saturation_factor = 1.0 + (saturation / 100.0)
        s = np.clip(s.astype(np.float32) * saturation_factor, 0, 255).astype(np.uint8)
        
        final_hsv = cv2.merge([h, s, v])
        processed = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return processed

# --- Rute Flask (API) ---

@app.route('/')
def index():
    """Menyajikan halaman HTML utama."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_endpoint():
    """Endpoint API untuk menerima dan memproses gambar."""
    try:
        data = request.json
        image_data = data.get('image')
        filter_type = data.get('filter', 'identity')
        brightness = int(data.get('brightness', 0))
        contrast = int(data.get('contrast', 0))
        saturation = int(data.get('saturation', 0))

        if not image_data:
            return jsonify({'error': 'Data gambar tidak ditemukan'}), 400

        # Dekode gambar
        img = decode_image(image_data)

        # Lewati pemrosesan jika tidak ada perubahan
        if filter_type == 'identity' and all(v == 0 for v in [brightness, contrast, saturation]):
            return jsonify({'processed_image': image_data})

        # Proses gambar
        processed_img = process_image(img, filter_type, brightness, contrast, saturation)

        # Encode kembali ke base64
        processed_data = encode_image_to_base64(processed_img)

        return jsonify({'processed_image': processed_data})

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return jsonify({'error': 'Terjadi kesalahan internal saat memproses gambar.'}), 500

# Menjalankan aplikasi
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)