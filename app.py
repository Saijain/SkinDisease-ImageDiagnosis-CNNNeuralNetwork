from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

model_path = 'DermAI.keras'

# Define class names based on training (Cell 18 from Colab notebook)
CLASS_NAMES = {
    0: 'Eczema',
    1: 'Warts Molluscum and other Viral Infections',
    2: 'Melanoma',
    3: 'Atopic Dermatitis',
    4: 'Basal Cell Carcinoma (BCC)',
    5: 'Melanocytic Nevi (NV)',
    6: 'Benign Keratosis-like Lesions (BKL)',
    7: 'Psoriasis pictures Lichen Planus and related diseases',
    8: 'Seborrheic Keratoses and other Benign Tumors',
    9: 'Tinea Ringworm Candidiasis and other Fungal Infections'
}

# Disease information database
DISEASE_INFO = {
    'Eczema': {
        'description': 'Eczema (also called atopic dermatitis) is a condition that causes dry, itchy, and inflamed skin. It\'s a common condition that can affect people of all ages.',
        'treatment': [
            'Use gentle, fragrance-free moisturizers daily',
            'Avoid harsh soaps and hot water',
            'Apply over-the-counter hydrocortisone cream for mild cases',
            'Take antihistamines to reduce itching',
            'Use a humidifier to keep air moist',
            'Wear soft, breathable fabrics like cotton'
        ],
        'consult_derm': 'See a dermatologist if your eczema is severe, covers large areas, is infected, or doesn\'t improve with over-the-counter treatments.',
        'urgency': 'moderate'
    },
    'Melanoma': {
        'description': 'Melanoma is the most serious type of skin cancer. It develops in melanocytes (the cells that produce melanin) and can spread to other parts of the body if not caught early.',
        'treatment': [
            'Surgical removal is the primary treatment',
            'Early detection is crucial for successful treatment',
            'Additional treatments may include immunotherapy, targeted therapy, or radiation',
            'Regular skin checks are essential for monitoring'
        ],
        'consult_derm': '⚠️ URGENT: If you have a suspicious mole or spot that changes in size, shape, or color, see a dermatologist immediately. Early detection saves lives.',
        'urgency': 'high'
    },
    'Atopic Dermatitis': {
        'description': 'Atopic dermatitis is a chronic form of eczema that often begins in childhood. It causes red, itchy patches on the skin, commonly in skin folds like elbows, knees, and neck.',
        'treatment': [
            'Keep skin moisturized with thick emollients',
            'Avoid triggers like allergens, stress, and irritants',
            'Use prescription topical corticosteroids for flare-ups',
            'Consider wet wrap therapy for severe cases',
            'Take short, lukewarm baths with mild soap',
            'Wear loose-fitting, cotton clothing'
        ],
        'consult_derm': 'Consult a dermatologist for proper diagnosis, prescription medications, and a personalized treatment plan. They can help identify and manage triggers.',
        'urgency': 'moderate'
    },
    'Basal Cell Carcinoma (BCC)': {
        'description': 'Basal cell carcinoma is the most common type of skin cancer. It grows slowly and rarely spreads, but should be treated to prevent local damage.',
        'treatment': [
            'Surgical removal (most common treatment)',
            'Mohs surgery for sensitive areas or large tumors',
            'Cryotherapy (freezing) for small, superficial lesions',
            'Topical medications for certain cases',
            'Radiation therapy in some situations'
        ],
        'consult_derm': '⚠️ IMPORTANT: See a dermatologist promptly for evaluation and treatment. While rarely fatal, BCC can cause significant local damage if left untreated.',
        'urgency': 'high'
    },
    'Melanocytic Nevi (NV)': {
        'description': 'Melanocytic nevi (moles) are common, usually harmless growths on the skin. Most people have 10-40 moles. They can be flat or raised, and various colors.',
        'treatment': [
            'Most moles require no treatment',
            'Monitor for changes using the ABCDE rule',
            'Protect from sun exposure with sunscreen',
            'Surgical removal if cosmetically desired or if suspicious',
            'Regular self-examinations recommended'
        ],
        'consult_derm': 'See a dermatologist if a mole changes in size, shape, or color, or if you notice new moles after age 30. Regular annual skin checks are recommended.',
        'urgency': 'low'
    },
    'Benign Keratosis-like Lesions (BKL)': {
        'description': 'Benign keratosis-like lesions are non-cancerous growths that appear as rough, scaly patches on the skin. They\'re common in older adults and harmless.',
        'treatment': [
            'Usually no treatment needed unless bothersome',
            'Cryotherapy (freezing) for removal',
            'Topical treatments like retinoids',
            'Curettage (scraping) for removal',
            'Laser therapy in some cases',
            'Regular monitoring recommended'
        ],
        'consult_derm': 'Consult a dermatologist for proper diagnosis and removal options if the lesions are bothersome, growing, or you\'re unsure of the diagnosis.',
        'urgency': 'low'
    },
    'Psoriasis pictures Lichen Planus and related diseases': {
        'description': 'Psoriasis is an autoimmune condition that causes rapid skin cell growth, leading to thick, scaly patches. Lichen planus is an inflammatory condition causing itchy, purple bumps.',
        'treatment': [
            'Topical corticosteroids and vitamin D analogs',
            'Phototherapy (light therapy)',
            'Systemic medications for moderate to severe cases',
            'Biologic medications for severe psoriasis',
            'Moisturizers and gentle skin care',
            'Stress management techniques'
        ],
        'consult_derm': 'See a dermatologist for proper diagnosis and treatment. These conditions often require prescription medications and specialized treatment plans.',
        'urgency': 'moderate'
    },
    'Seborrheic Keratoses and other Benign Tumors': {
        'description': 'Seborrheic keratoses are common, benign (non-cancerous) skin growths that look like warts or moles. They\'re typically brown or black and have a waxy, stuck-on appearance.',
        'treatment': [
            'No treatment necessary unless bothersome',
            'Cryotherapy (freezing) for removal',
            'Curettage (scraping) for removal',
            'Laser therapy',
            'Electrosurgery',
            'Regular monitoring for changes'
        ],
        'consult_derm': 'Consult a dermatologist if you want them removed for cosmetic reasons, if they\'re irritated, or if you\'re unsure of the diagnosis.',
        'urgency': 'low'
    },
    'Tinea Ringworm Candidiasis and other Fungal Infections': {
        'description': 'Fungal skin infections like ringworm (tinea) and candidiasis are common conditions caused by fungi. They can cause red, itchy, scaly patches or rashes.',
        'treatment': [
            'Over-the-counter antifungal creams (clotrimazole, miconazole)',
            'Prescription antifungal medications for severe cases',
            'Keep affected areas clean and dry',
            'Avoid sharing towels or clothing',
            'Wear breathable, moisture-wicking fabrics',
            'Complete the full course of treatment'
        ],
        'consult_derm': 'See a dermatologist if the infection doesn\'t improve after 2-4 weeks of over-the-counter treatment, spreads, or becomes severe.',
        'urgency': 'moderate'
    },
    'Warts Molluscum and other Viral Infections': {
        'description': 'Viral skin infections like warts and molluscum contagiosum are caused by viruses. Warts are rough, raised bumps while molluscum causes smooth, dome-shaped bumps.',
        'treatment': [
            'Many resolve on their own over time',
            'Over-the-counter salicylic acid for warts',
            'Cryotherapy (freezing) by healthcare provider',
            'Topical treatments (podophyllin, imiquimod)',
            'Laser therapy for stubborn cases',
            'Avoid picking or scratching to prevent spread'
        ],
        'consult_derm': 'Consult a dermatologist if warts or molluscum are spreading, painful, in sensitive areas, or persist after home treatment.',
        'urgency': 'moderate'
    }
}

# Load model once at startup
print("Loading model...")
model = load_model(model_path, safe_mode=False, compile=False)
print("Model loaded successfully!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_file, model):
    """Predict disease from image file"""
    try:
        # Open image from file
        image = Image.open(io.BytesIO(image_file.read()))
        image_file.seek(0)  # Reset file pointer
        
        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = image.resize((224, 224))
        image = np.array(image)
        
        # Ensure image is 3-channel RGB (handles edge cases)
        if len(image.shape) == 2:  # Grayscale
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:  # RGBA
            image = image[:, :, :3]  # Remove alpha channel
        
        # CRITICAL: Use the same preprocessing as during training (ResNet50 ImageNet preprocessing)
        # This converts RGB to BGR and applies ImageNet mean normalization
        image = preprocess_input(image.astype('float32'))
        
        image = np.expand_dims(image, axis=0)
        predictions = model.predict(image, verbose=0)
        
        # Get all predictions sorted by confidence
        results = []
        for i in range(len(CLASS_NAMES)):
            results.append({
                'disease': CLASS_NAMES[i],
                'confidence': float(predictions[0][i] * 100)
            })
        
        # Sort by confidence (highest first)
        results.sort(key=lambda x: x['confidence'], reverse=True)
        
        predicted_disease = results[0]['disease']
        
        # Get disease information
        disease_info = DISEASE_INFO.get(predicted_disease, {
            'description': 'This condition requires professional medical evaluation.',
            'treatment': ['Consult a dermatologist for proper diagnosis and treatment'],
            'consult_derm': 'Please see a dermatologist for accurate diagnosis and treatment recommendations.',
            'urgency': 'moderate'
        })
        
        return {
            'predicted_disease': predicted_disease,
            'confidence': results[0]['confidence'],
            'all_predictions': results,
            'disease_info': disease_info
        }
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a JPG, PNG, or GIF image.'}), 400
    
    try:
        # Get prediction
        result = predict_image(file, model)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("DermAI Web Application Starting...")
    print("="*60)
    print(f"Open your browser and navigate to: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
