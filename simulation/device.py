import random
import numpy as np
import paho.mqtt.client as mqtt
import time
import pickle
# import threading
from configuration import EMD_FLAG, RANDOM_SEED, PORT, DEVICE, MOVE_SPEED, COMM_RANGE, DEVICE_DEFAULT_COLOUR, MIN_CLUSTER_SIZE, NUM_OF_CLASSES
from model import save_checkpoint, train, load_checkpoint, reset_classifier_weights, get_parameters, set_parameters, cumulative_fedAvg
# from mcunet.mcunet.model_zoo import build_model
import torch.nn as nn
import torch
from math import exp
from dataset import calculate_emd, get_emd_distance, calculate_label_distribution
import os
import csv
from eval import evaluate_model
# from torchvision.datasets import CIFAR10
# from torchvision import transforms
# from torch.utils.data import DataLoader
# from configuration import DATA_PATH, BATCH_SIZE


np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)

class Device:
    def __init__(self, id, x, y, dataloader, is_head=False, color = DEVICE_DEFAULT_COLOUR):
        self.id = id
        self.x = x
        self.y = y

        if self.id != -1:
            # model, resolution, description = build_model(net_id="mcunet-in3", pretrained=True)
            # self.model = replace_fc_layer(model, NUM_OF_CLASSES)
            self.model, _, self.optimizer = load_checkpoint("./checkpoints/baseline_checkpoint_2.pth")
            reset_classifier_weights(self.model)
            self.model.to(DEVICE)
            self.checkpoint_path = f"./checkpoints/device_{self.id}_checkpoint.pth"
            self.learning_rate = 0.001
            self.rounds = 0

            self.dataloader = dataloader
            self.distribution = calculate_label_distribution(dataloader)
            self.emd = calculate_emd(self.distribution)
            
            if EMD_FLAG:
                self.num_samples = len(dataloader.dataset)*(1-2*self.emd)**3
            else:
                self.num_samples = len(dataloader.dataset)
            print(f"Device {self.id} EMD: {self.emd}, num_samples: {self.num_samples}")
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
            # self.publish_thread = None  
            self.direction = random.uniform(0, 2*np.pi) 
            self.color = color
            self.total_samples = self.num_samples
            self.total_weights = [param * self.num_samples for param in get_parameters(self.model)]
            self.seen_devices = []
###############################################################################################
################################# MQTT Client #################################################
###############################################################################################

    def start_client(self):
        self.client = mqtt.Client(client_id=str(self.id), protocol=mqtt.MQTTv311)
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()

        # self.publish_thread = threading.Thread(target=self.publish_data)
        # self.publish_thread.start()

    def on_connect(self, client, userdata, flags, rc):
        #print("Connected with result code " + str(rc))
        self.client.subscribe(self.topic_sub)

    def on_message(self, client, userdata, message):

        payload = pickle.loads(message.payload)
        sender_id = payload.get("sender_id")
        sender_topic_sub = payload.get("sender_topic_sub")
        command = payload.get("command")
        message_data = payload.get("message_data")
        num_samples = payload.get("num_samples")

        if sender_id not in self.seen_devices:
            self.seen_devices.append(sender_id)
        else:
            return
        
        if message_data is not None:
            if command == "reply":
                # print(f"sender: {sender_id}, reci: {self.id}")
                self.total_weights, self.total_samples = cumulative_fedAvg(self.total_weights, message_data, self.total_samples, num_samples)
                # self.total_weight = total_weight
                # self.total_samples = total_samples
                # Reply to the sender
                payload = pickle.dumps({
                    "sender_id": self.id,
                    "sender_topic_sub": self.topic_sub,
                    "command": "None",
                    "message_data": get_parameters(self.model),
                    "num_samples": self.num_samples,
                    "emd": self.emd
                })
                self.client.publish(sender_topic_sub, payload)
            else:
                self.total_weights, self.total_samples = cumulative_fedAvg(self.total_weights, message_data, self.total_samples, num_samples)

        
        # print(f"Device {self.id} received message from Device {sender_id}")

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
            "sender_topic_sub": self.topic_sub,
            "command": "reply",
            "message_data": data, #self.model.classifier.linear.weight.data,
            "num_samples": self.num_samples,
            "emd": self.emd
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
        # optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=0.005)
        self.rounds += 1
        self.learning_rate *= exp(-0.1 * self.rounds)
        self.optimizer = torch.optim.Adam(self.model.classifier.parameters(), lr=0.001, weight_decay=1e-5)
        epoch_loss, epoch_acc = train(self.rounds, self.model, self.dataloader, self.optimizer, self.checkpoint_path, self.num_samples, self.emd)
        csv_filename = "./data/DFL_60device_60range_05alpha_1layer.csv"
        file_exists = os.path.isfile(csv_filename)
        with open(csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['round','device_id', 'train_loss', 'num_train_samples','x_pos', 'y_pos', 'emd'])
            if isinstance(epoch_loss, torch.Tensor):
                epoch_loss = epoch_loss.item()
            writer.writerow([self.rounds, self.id, epoch_loss, self.num_samples, self.x, self.y, self.emd])
        return epoch_loss, epoch_acc
    
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
                    distances = [self.calculate_distance(self, device, emd=EMD_FLAG) for device in self.in_range_devices]
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
            distances = [self.calculate_distance(self, head, emd=EMD_FLAG) for head in cluster_heads]
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

    def inner_gossip(self):
        if self.paired:
            return

        # available_devices = [device for device in self.in_range_devices if (not device.paired) and
        #                      (device.cluster_id == self.cluster_id) and (device.cluster_id not in self.seen_devices)]

        available_devices = [device for device in self.in_range_devices if (not device.paired) and
                             (device.cluster_id not in self.seen_devices)]
        
        if available_devices:
            partner = random.choice(available_devices)
            weights = get_parameters(self.model)
            self.communicate(partner, weights)
            self.paired = True
            partner.paired = True

    def inter_gossip(self):
        if self.paired:
            return

        available_devices = [device for device in self.in_range_devices if (not device.paired) and
                              (device.cluster_id != self.cluster_id) and (device.cluster_id not in self.seen_devices)]

        if available_devices:
            partner = random.choice(available_devices)
            weights = get_parameters(self.model)
            self.communicate(partner, weights)
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

    def calculate_distance(self, device_a, device_b, emd = False):
        if emd:
            return get_emd_distance(device_a.distribution, device_b.distribution)*100
        else:
            return np.sqrt((device_a.x - device_b.x)**2 + (device_a.y - device_b.y)**2)
        

    def in_range(self, other_device):
        distance = self.calculate_distance(self, other_device)
        return distance <= COMM_RANGE
    
    def aggregate_model(self):
        # avg_weights = [weight / self.total_samples for weight in self.total_weights]
        avg_weights = []
        for weight in self.total_weights:  
            avg_weight = weight / self.total_samples
            avg_weights.append(avg_weight)
        set_parameters(self.model, avg_weights)
        save_checkpoint(self.model, self.optimizer, self.num_samples, self.emd, self.checkpoint_path)

        self.total_samples = self.num_samples
        self.total_weights = [param * self.num_samples for param in get_parameters(self.model)]
        self.seen_devices = []

        # transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        #     ])

        # test_dataset = CIFAR10(root= DATA_PATH, train=False, download=False, transform=transform)
        # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # acc, f1 = evaluate_model(self.model, test_loader, DEVICE)
        # print(f"Device {self.id}, seen:{len(self.seen_devices)} accuracy: {acc:.4f}, F1 score: {f1:.4f}")
