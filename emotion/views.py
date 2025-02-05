from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from deepface import DeepFace
from deepface.commons import functions
import numpy as np
from PIL import Image
import tensorflow as tf
import gc

# Preload models and backend items once at startup
models = {
    'emotion': functions.build_model('Emotion')
}
tf.keras.backend.clear_session()  # Clear initial session after preload

@csrf_exempt
def analyze_emotion(request):
    if request.method == 'POST':
        try:
            image_file = request.FILES.get('image')
            if not image_file:
                return JsonResponse({'error': 'No image provided'}, status=400)

            with Image.open(image_file) as img:
                img = img.convert('RGB')
                img_array = np.array(img)

            # Use preloaded model and clear session after processing
            try:
                analysis = DeepFace.analyze(
                    img_array,
                    actions=['emotion'],
                    models=models,
                    enforce_detection=False
                )
            finally:
                tf.keras.backend.clear_session()

            dominant_emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis.get('dominant_emotion')
            
            return JsonResponse({'emotion': dominant_emotion})
            
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
        finally:
            # Cleanup resources
            del img_array
            if 'analysis' in locals():
                del analysis
            gc.collect()
            
    return JsonResponse({'error': 'Invalid method'}, status=400)

def show_home(request):
    return render(request, 'index.html')
