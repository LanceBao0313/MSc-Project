import paho.mqtt.client as mqtt
import time
import random
import threading

def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe(topic_sub)

def on_message(client, userdata, message):
    command = message.payload.decode("utf-8")
    print(f"Received command: {command}")
    if command == "publish":
        start_publishing()
    elif command == "subscribe":
        stop_publishing()
    else:
        print("Unknown command")

def start_publishing():
    global publishing
    publishing = True
    while publishing:
        data = random.randint(1, 100)  # Simulate sensor data
        client.publish(topic_pub, data)
        print(f"Published: {data} to topic: {topic_pub}")
        time.sleep(5)  # Adjust the sleep time as needed

def stop_publishing():
    global publishing
    publishing = False
    print("Stopped publishing")

client.on_connect = on_connect
client.on_message = on_message

client.connect(broker, port, 60)
client.loop_start()

if __name__ == "__main__":
    publishing = False
    while True:
        time.sleep(1)  # Keep the main thread alive
