import asyncio
import websockets
import json
import threading
import time

# Flag to signal the thread to stop
stop_websocket_thread = False

async def send_videoStreamData(uri):
    async with websockets.connect(uri) as websocket:
        while not stop_websocket_thread:
            # Create your JSON data
            data = {"key": "value"}

            # Convert the data to JSON format
            json_data = json.dumps(data)

            # Send the JSON data to the server
            await websocket.send(json_data)
            print(f"Sent data to server: {json_data}")

            # Wait for 5 seconds before sending the next data
            await asyncio.sleep(5)

def startWebsocket_videoStreamCommunication(uri):
    asyncio.set_event_loop(asyncio.new_event_loop())
    asyncio.get_event_loop().run_until_complete(send_videoStreamData(uri))

if __name__ == "__main__":
    server_uri = "ws://127.0.0.1:12346"

    # Create a thread for WebSocket communication
    websocket_thread = threading.Thread(target=startWebsocket_videoStreamCommunication, args=(server_uri,))
    websocket_thread.start()

    # Main thread continues to print
    while True:
        print("Main thread is still printing...")
        time.sleep(2)

        # Check if the user wants to stop the WebSocket thread
        user_input = input("Type 'stop' to stop the WebSocket thread: ")
        if user_input.lower() == 'stop':
            stop_websocket_thread = True
            websocket_thread.join()  # Wait for the thread to finish
            print("WebSocket thread stopped.")
            break
