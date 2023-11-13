import asyncio
import websockets
import json
import time

async def send_data(uri):
    async with websockets.connect(uri) as websocket:
        detectedObjects = [1,2,3,4,5,6,7,8,9,10]

        ObjectListJson = {
          "AnalyticsId": 0,
          "EvaluationTimestamp": str(int(time.time())*1000),
          "Failure": False,
          "FailureState": "None",
          "Objects": [],
          "Part": 1,
          "TotalParts": 1
        }

        if len(detectedObjects) > 0:
            for object in detectedObjects:
                objectJson = {
                "Category": object,
                "Color": None,
                "Id": None,
                    "LastState": {
                        "BoundingBox": [None, None, None, None],
                        "MapAcceleration": None,
                        "MapPosition": [None, None],
                        "MapSpeed": None,
                        "MapSpeedAngle": None,
                        "SensorPosition": [None, None],
                        "Timestamp": str(int(time.time())*1000),
                        "WGS84Position": [None, None]
                    }
                }
                ObjectListJson["Objects"].append(objectJson)
        else:
            objectJson = None
            ObjectListJson["Objects"].append(objectJson)
  
        

        while True:
            # Create your JSON data
            #data = {"key": "value"}
            data = ObjectListJson

            # Convert the data to JSON format
            json_data = json.dumps(data)

            # Send the JSON data to the server
            await websocket.send(json_data)
            print(f"Sent data to server: {json_data}")

            # Wait for 5 seconds before sending the next data
            await asyncio.sleep(5)

if __name__ == "__main__":
    server_uri = "ws://localhost:8765"
    asyncio.run(send_data(server_uri))