from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from deepface import DeepFace
import numpy as np
from PIL import Image
import gc

@csrf_exempt
def analyze_emotion(request):
    if request.method == 'POST':
        try:
            # Get the image from the request
            image_file = request.FILES.get('image')

            if not image_file:
                return JsonResponse({'error': 'No image provided'}, status=400)

            # Open the image file using PIL and convert it to RGB
            with Image.open(image_file) as img:
                img = img.convert('RGB')  # Ensure the image is in RGB format
                # Convert the PIL image to a NumPy array for processing
                img_array = np.array(img)

            # Analyze the image using DeepFace to detect emotions
            analysis = DeepFace.analyze(img_array, actions=['emotion'], enforce_detection=False)

            # Extract the dominant emotion
            dominant_emotion = analysis[0]['dominant_emotion'] if isinstance(analysis, list) else analysis.get('dominant_emotion')

            if dominant_emotion:
                return JsonResponse({'emotion': dominant_emotion}, status=200)
            else:
                return JsonResponse({'error': 'Emotion could not be detected'}, status=500)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

        finally:
            # Explicitly delete large objects and invoke garbage collection
            locals_cleanup = ['img_array', 'analysis']
            for obj in locals_cleanup:
                if obj in locals():
                    del locals()[obj]
            gc.collect()

    return JsonResponse({'error': 'Invalid request method'}, status=400)

def show_home(request):
    return render(request, 'index.html')
