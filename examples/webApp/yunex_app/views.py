from django.shortcuts import render
from django.http import HttpResponse
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.middleware.csrf import get_token
from django.http import HttpResponseServerError
from django.shortcuts import render
import json
import time
import sys
import cv2
import numpy as np
import mimetypes
import subprocess

def yunex_app(request):
	return render(request, 'main/index.html')

def run_external_app(request):
    try:
        # Replace 'app.exe' with the path to your executable
        subprocess.run(['main/test.exe'], check=True)
        return HttpResponse("External application executed successfully.")
    except subprocess.CalledProcessError as e:
        return HttpResponse(f"Error executing the external application: {e}")

def display_string(request, input_string):
    print(input_string, file=sys.stderr)
    return render(request, 'main/index.html', {'input_string': input_string})

@csrf_exempt
def receive_json(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            return JsonResponse({'message': 'Received JSON data', 'data': data})
        except json.JSONDecodeError as e:
            return JsonResponse({'message': 'Invalid JSON data', 'error': str(e)}, status=400)
    else:
        return JsonResponse({'message': 'Invalid request method'}, status=400)

def csrf_token_endpoint(request):
    csrf_token = get_token(request)
    return JsonResponse({'csrf_token': csrf_token})

# Create a queue to receive frames from your main.py script
#@csrf_exempt
def generate_processed_frames(request):
    if request.method == 'POST':
        # Retrieve the image data from the request
        image_data = request.body
        if not image_data:
            return HttpResponseServerError("No image data received")

        # Convert the byte string back to a NumPy array
        #decoded_frame = np.frombuffer(image_data, dtype=np.uint8)

        # Decode the NumPy array to an image
        #frame = cv2.imdecode(decoded_frame, cv2.IMREAD_COLOR)
        #print("************", file=sys.stderr)
        #print(image_data, file=sys.stderr)
        #print("************", file=sys.stderr)
        
        
        #####size = 200, 200, 3
        ######img = np.zeros(size, dtype=np.uint8)
        ######frame = cv2.imencode('.jpg', img)[1].tobytes()

        # Yield the image data as a multipart response
        ######yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image_data + b'\r\n')

        img = cv2.imread("main/lizerd.jpg")
        img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
        frame = cv2.imencode('.jpg', img)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@csrf_exempt
def stream_image(request):
    #print(request.body, file=sys.stderr)
    return StreamingHttpResponse(generate_processed_frames(request), content_type="multipart/x-mixed-replace;boundary=frame")
    #print(response, file=sys.stderr)