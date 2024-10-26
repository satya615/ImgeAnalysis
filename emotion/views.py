from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from deepface import DeepFace
import numpy as np
from PIL import Image
import io

@csrf_exempt
def analyze_emotion(request):
    if request.method == 'POST':
        try:
            # Get the image from the request
            image_file = request.FILES.get('image')

            # Open the image file using PIL and convert it to RGB
            img = Image.open(image_file)
            img = img.convert('RGB')  # Ensure the image is in RGB format

            # Convert the PIL image to a NumPy array for processing
            img = np.array(img)

            # Analyze the image using DeepFace to detect emotions
            analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

            # DeepFace returns a list of results, even for one face, so access the first result
            if isinstance(analysis, list):
                analysis = analysis[0]

            # Extract the dominant emotion
            dominant_emotion = analysis.get('dominant_emotion', None)

            if dominant_emotion:
                # Send back the detected emotion as a JSON response
                return JsonResponse({'emotion': dominant_emotion}, status=200)
            else:
                return JsonResponse({'error': 'Emotion could not be detected'}, status=500)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Invalid request method'}, status=400)
def show_home(request):
    return HttpResponse(request, 'home.html')