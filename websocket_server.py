import asyncio
import websockets
import base64
import cv2
import numpy as np
import json

async def handle_client(websocket, path):
    print(f"Client connected: {path}")
    async for message in websocket:
        data = json.loads(message)
        image_base64 = data['image']
        image_bytes = base64.b64decode(image_base64)
        np_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        
        # Display the received frame
        cv2.imshow('Received Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyWindow('Received Frame')

async def main():
    start_server = websockets.serve(handle_client, 'localhost', 8081)
    print("Starting WebSocket server...")
    await start_server
    print("WebSocket server started. Listening for connections...")
    await asyncio.Future()  # Run forever

if __name__ == "__main__":
    asyncio.run(main())
