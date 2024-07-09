import random
import numpy as np
import paho.mqtt.client as mqtt
import time
import pickle
import threading
from configuration import PORT, DEVICE, MOVE_SPEED, COMM_RANGE, DEVICE_DEFAULT_COLOUR, MIN_CLUSTER_SIZE, NUM_OF_CLASSES
from model import replace_fc_layer, train
from mcunet.mcunet.model_zoo import build_model
import torch.nn as nn
import torch


class Device:
    def __init__(self, id, x, y, dataloader, is_head=False, color = DEVICE_DEFAULT_COLOUR):
        self.id = id
        self.x = x
        self.y = y
        model, resolution, description = build_model(net_id="mcunet-in3", pretrained=True)
        self.model = replace_fc_layer(model, NUM_OF_CLASSES)
        self.model.to(DEVICE)
        self.dataloader = dataloader #self.load_local_data()
        self.cluster_id = None
        self.in_range_devices = []
        self.is_head = is_head
        self.cluster = []

        self.broker = "localhost"
        self.port = PORT
        self.topic_pub = f"device_{self.id}/data"
        self.topic_sub = f"device_{self.id}/command"
        self.topic_comm = f"device_{self.id}/comm"

        self.paired = False
        self.publishing = False
        self.publish_thread = None  
        self.direction = random.uniform(0, 2*np.pi) 
        self.color = color

###############################################################################################
################################# MQTT Client #################################################
###############################################################################################
    def start_client(self):
        self.client = mqtt.Client(client_id=str(self.id), protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()

        self.publish_thread = threading.Thread(target=self.publish_data)
        self.publish_thread.start()

    def on_connect(self, client, userdata, flags, rc):
        #print("Connected with result code " + str(rc))
        self.client.subscribe(self.topic_sub)

    def on_message(self, client, userdata, message):
        payload = pickle.loads(message.payload)
        command = payload.get("command")
        sender_id = payload.get("sender_id")
        message_data = payload.get("message_data")

        if command:
            #print(f"Device {self.id} received command: {command}")
            if command == "publish" and not self.publishing:
                self.publishing = True
                self.publish_thread = threading.Thread(target=self.publish_data)
                self.publish_thread.start()
            elif command == "subscribe" and self.publishing:
                self.publishing = False
                if self.publish_thread is not None:
                    self.publish_thread.join()
                print("Stopped publishing")
        elif message_data:
            pass
            #print(f"Device {self.id} received message from Device {sender_id}: {message_data}")

    def publish_data(self):
        while self.publishing:
            data = np.random.rand(10)  # Simulate sensor data
            payload = {
                "client_id": self.id,
                "data": data
            }
            self.client.publish(self.topic_pub, pickle.dumps(payload))
            print(f"{self.id} published data")
            time.sleep(random.uniform(1, 5))  # Simulate varying data publishing intervals

    def stop_publishing(self):
        global publishing
        publishing = False
        print("Stopped publishing")

    def communicate(self, partner, data):
        # Function for communication between devices
        #print(f"Device {self.id} communicating with Device {partner.id}")
        # Generate a random message 
        if self.paired or partner.paired:
            return
        
        #message = f"Hello Device {partner.id}, from Device {self.id}"

        payload = pickle.dumps({
            "sender_id": self.id,
            "message_data": data, #self.model.classifier.linear.weight.data,
        })
        # Publish the message to the partner's communication topic
        self.client.publish(partner.topic_sub, payload)
        #print(f"Device {self.id} sent message to Device {partner.id}")

###############################################################################################
################################# Local Training ##############################################
###############################################################################################

    def local_training(self):
        # Function for local training
        #print(f"Device {self.id} is training")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        train(self.model, self.dataloader, criterion, optimizer, DEVICE)
    
###############################################################################################
#################################### DK-means #################################################
###############################################################################################

    def update_cluster_member(self):
        for device in self.cluster:
            if (device not in self.in_range_devices) or (device.cluster_id != self.id):
                self.cluster.remove(device)

    def assign_clusters(self):
        if self.is_head:
            if (len(self.in_range_devices) <= MIN_CLUSTER_SIZE) or (len(self.cluster) <= MIN_CLUSTER_SIZE):
                self.is_head = False
                self.cluster_id = None
                self.color = DEVICE_DEFAULT_COLOUR
                self.cluster = []
            
            self.update_cluster_member()
            return
        
        cluster_heads = [device for device in self.in_range_devices if device.is_head]
        if not cluster_heads:
            self.cluster_id = None
            self.color = DEVICE_DEFAULT_COLOUR
            self.cluster = []
            # If there are no cluster heads in range, become a cluster head if it is in a dense area
            if len(self.in_range_devices) > 0:
                if len(self.in_range_devices) > MIN_CLUSTER_SIZE:
                    self.is_head = True
                    self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                else:
                    # Find the nearest neighbor in comm range
                    distances = [self.calculate_distance(self, device) for device in self.in_range_devices]
                    closest_index = np.argmin(distances)
                    closest_device = self.in_range_devices[closest_index]
                    # Join the cluster of the closest device (one-hop clustering) 
                    if closest_device.cluster_id is not None:
                        self.cluster_id = closest_device.cluster_id
                        self.color = closest_device.color
                        self.cluster = closest_device.cluster
                        # Inform the cluster head
                        head = self.find_device(closest_device.in_range_devices, closest_device.cluster_id)
                        if head is not None:
                            head.cluster.append(self)

        else:
            closest_head = self.find_closest_cluster_head()
            self.cluster_id = closest_head.id
            closest_head.cluster.append(self)
            self.cluster = closest_head.cluster
            self.color = closest_head.color

    def find_device(self, list, id):
        for device in list:
            if device.id == id:
                return device
        return None
    
    def find_closest_cluster_head(self):
        if self.is_head:
            return None
        cluster_heads = [device for device in self.in_range_devices if device.is_head]
        if cluster_heads:
            distances = [self.calculate_distance(self, head) for head in cluster_heads]
            closest_index = np.argmin(distances)
            closest_head = cluster_heads[closest_index]
            return closest_head

    def calculate_cluster_center(self):
        if self.is_head:
            number_of_devices = len(self.cluster)
            if number_of_devices > 0:
                x = 0
                y = 0
                for device in self.cluster:
                    x += device.x
                    y += device.y
                # return x / number_of_devices, y / number_of_devices
                center = Device(-1, x / number_of_devices, y / number_of_devices, 0)
                return center
            
    def update_cluster_heads(self):
        if self.is_head and len(self.cluster) > 0:
            # find device closest to the center
            center = self.calculate_cluster_center()
            distances = [self.calculate_distance(center, device) for device in self.cluster]
            closest_index = np.argmin(distances)
            closest_device = self.cluster[closest_index]
            if closest_device.id != self.id:# and not closest_device.is_head:
                #print(f'head {self.id} is giving head to {closest_device.id}')
                closest_device.is_head = True
                closest_device.color = self.color
                self.is_head = False
                closest_device.cluster = self.cluster

    def gossip(self):
        if self.paired:
            return

        for device in self.in_range_devices:
            if device.paired:
                self.in_range_devices.remove(device)

        if self.in_range_devices:
            partner = random.choice(self.in_range_devices)
            self.communicate(partner, self.model.classifier.weight.data)
            self.paired = True
            partner.paired = True


###############################################################################################
######################################## Util #################################################
###############################################################################################
    
    def out_of_bounds(self):
        return self.x < 0 or self.x > 100 or self.y < 0 or self.y > 100

    def move(self):
        # Simulate slow and smooth movement
        if random.uniform(0, 1) < 0.05:
            self.direction = random.uniform(0, 2*np.pi)

        if self.out_of_bounds():
            self.direction += np.pi

        self.x += MOVE_SPEED * np.cos(self.direction)
        self.y += MOVE_SPEED * np.sin(self.direction)

    def calculate_distance(self, device_a, device_b):
        return np.sqrt((device_a.x - device_b.x)**2 + (device_a.y - device_b.y)**2)

    def in_range(self, other_device):
        distance = self.calculate_distance(self, other_device)
        return distance <= COMM_RANGE