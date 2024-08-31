import cv2
import numpy as np
import asyncio
import websockets
import base64
import json
from deepface import DeepFace
from datetime import datetime, timedelta
from collections import Counter
import time
import csv
import os
from concurrent.futures import ThreadPoolExecutor
import torch

# WebSocket server URL
WEBSOCKET_SERVER_URL = 'ws://localhost:8081'

def create_tracker():
    """Create a new tracker."""
    return cv2.legacy.TrackerKCF_create()

def cosine_similarity(embedding1, embedding2):
    """Calculate the cosine similarity between two embeddings."""
    vec1 = np.array(embedding1['embedding'])
    vec2 = np.array(embedding2['embedding'])
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

def find_matching_face_id(new_embedding, stored_embeddings):
    """Find the matching face ID based on face embedding similarity."""
    max_similarity = 0.7
    matched_face_id = None
    for face_id, embedding in stored_embeddings.items():
        similarity = cosine_similarity(embedding, new_embedding)
        if similarity > max_similarity:
            matched_face_id = face_id
            max_similarity = similarity
    return matched_face_id

def get_current_time():
    """Return the current time."""
    return datetime.now()

def save_to_csv(data, filename):
    """Save collected data to a CSV file."""
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data)

def rename_emotion(emotion):
    """Rename emotion based on the provided mapping."""
    emotion_mapping = {
        'sad': 'stressed',
        'fear': 'tensed'
    }
    return emotion_mapping.get(emotion, emotion)

def analyze_face(face_roi):
    """Analyze the face to get emotion and embedding."""
    try:
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        emotion = rename_emotion(result[0]['dominant_emotion'])
        embedding = DeepFace.represent(face_roi, model_name='Facenet', enforce_detection=False)[0]
        return emotion, embedding
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return None, None

async def send_frame(frame, websocket):
    """Send a frame to the WebSocket server."""
    _, buffer = cv2.imencode('.jpg', frame)
    frame_bytes = buffer.tobytes()
    frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
    await websocket.send(json.dumps({'image': frame_base64}))

# Ensure the directory for the CSV file exists
log_file = 'face_tracking_log.csv'
log_dir = os.path.dirname(log_file)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
cap = cv2.VideoCapture(0)

# Set camera FPS and resolution to improve performance
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

face_trackers = {}
face_times = {}
face_emotions = {}
face_embeddings = {}
face_last_update = {}
stored_embeddings = {}
face_id_count = 0  # Initialize face_id_count

# Initialize last_detection_time
last_detection_time = time.time()
detection_interval = 5

# Write CSV header
save_to_csv(['Face ID', 'Event', 'Time', 'Duration', 'Cumulative Emotion'], log_file)

# Thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

async def main():
    global face_id_count  # Ensure that we're using the global variable
    global last_detection_time  # Ensure that we're using the global variable
    
    async with websockets.connect(WEBSOCKET_SERVER_URL) as websocket:
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                face_ids_to_remove = []
                for face_id, tracker in face_trackers.items():
                    success, bbox = tracker.update(frame)
                    if success:
                        x, y, w, h = [int(v) for v in bbox]
                        face_roi = frame[y:y + h, x:x + w]
                        
                        future = executor.submit(analyze_face, face_roi)
                        emotion, embedding = future.result()
                        
                        if emotion and embedding:
                            face_emotions[face_id].append(emotion)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(frame, f"ID {face_id} {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                            face_last_update[face_id] = get_current_time()
                            face_embeddings[face_id] = embedding
                    else:
                        if face_times[face_id]['time_out'] is None:
                            face_times[face_id]['time_out'] = get_current_time()
                            duration = face_times[face_id]['time_out'] - face_times[face_id]['time_in']
                            cumulative_emotion = Counter(face_emotions[face_id]).most_common(1)[0][0]
                            save_to_csv([face_id, 'time-out', face_times[face_id]['time_out'], str(duration), cumulative_emotion], log_file)
                            print(f"Face ID {face_id} time-out at: {face_times[face_id]['time_out']}")
                            print(f"Face ID {face_id} was present for: {duration}")
                            print(f"Face ID {face_id} cumulative emotion: {cumulative_emotion}")
                        face_ids_to_remove.append(face_id)
                
                for face_id in face_ids_to_remove:
                    face_trackers.pop(face_id, None)
                    face_times.pop(face_id, None)
                    face_emotions.pop(face_id, None)
                    face_embeddings.pop(face_id, None)
                    face_last_update.pop(face_id, None)
                    stored_embeddings.pop(face_id, None)

                current_time = time.time()
                if current_time - last_detection_time > detection_interval:
                    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
                    last_detection_time = current_time
                    
                    for (x, y, w, h) in faces:
                        face_roi = frame[y:y + h, x:x + w]
                        
                        future = executor.submit(analyze_face, face_roi)
                        emotion, face_embedding = future.result()

                        if face_embedding:
                            matched_face_id = find_matching_face_id(face_embedding, stored_embeddings)

                            if matched_face_id is None:
                                face_id_count += 1
                                tracker = create_tracker()
                                tracker.init(frame, (x, y, w, h))
                                face_trackers[face_id_count] = tracker
                                face_times[face_id_count] = {'time_in': get_current_time(), 'time_out': None}
                                face_emotions[face_id_count] = [emotion]
                                face_embeddings[face_id_count] = face_embedding
                                face_last_update[face_id_count] = get_current_time()
                                stored_embeddings[face_id_count] = face_embedding
                                save_to_csv([face_id_count, 'time-in', face_times[face_id_count]['time_in'], '', ''], log_file)
                                print(f"Face ID {face_id_count} time-in at: {face_times[face_id_count]['time_in']}")
                            else:
                                if get_current_time() - face_last_update[matched_face_id] > timedelta(seconds=1):
                                    tracker = create_tracker()
                                    tracker.init(frame, (x, y, w, h))
                                    face_trackers[matched_face_id] = tracker
                                    face_last_update[matched_face_id] = get_current_time()
                                    face_embeddings[matched_face_id] = face_embedding
                                    print(f"Face ID {matched_face_id} re-entered at: {datetime.now()}")
                
                await send_frame(frame, websocket)

                fps = 1 / (time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Real-time Emotion Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except Exception as e:
            print(f"Error in main loop: {e}")

    cap.release()
    cv2.destroyAllWindows()
    print("Resources released, exiting.")

if __name__ == "__main__":
    asyncio.run(main())
