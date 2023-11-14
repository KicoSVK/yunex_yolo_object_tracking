import asyncio
import websockets

async def handle_client(websocket, path):
    while True:
        try:
            # Receive data from the client
            data = await websocket.recv()
            print(f"Received data from client: {data}")

            # Add your processing logic here

        except websockets.exceptions.ConnectionClosed:
            print("Client disconnected")
            break

async def main():
    server = await websockets.serve(handle_client, "127.0.0.1", 8051)

    print("WebSocket server started")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())