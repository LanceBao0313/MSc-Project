from device import Device
import random
import configuration
import threading
import time

random.seed(configuration.RANDOM_SEED)

class Network:
    def __init__(self, num_devices, num_clusters, dataloaders):
        self.devices = self.initialize_devices(num_devices, num_clusters, dataloaders)

    def initialize_devices(self, num_devices, num_clusters, dataloaders):
        devices = []
        head_prob = num_clusters / num_devices
        for i in range(num_devices):
            x, y = random.uniform(0, 100), random.uniform(0, 100)
            rand = random.uniform(0, 1)
            if rand < head_prob:
                print(f"Device {i} is a cluster head")
                rand_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                device = Device(i, x, y, dataloaders[i], True, rand_color)
            else:
                device = Device(i, x, y, dataloaders[i], False)
            device.start_client()
            devices.append(device)
        return devices

    def update_device_positions(self):
        for device in self.devices:
            device.move()
        
    def update_in_range_devices(self):
        for device in self.devices:
            device.in_range_devices = [d for d in self.devices if 
                                       d.id != device.id and 
                                       device.in_range(d)]
            
    def reset_paired_devices(self):
        for device in self.devices:
            device.paired = False

    def run_gossip_protocol(self):
        # shaffle the devices to avoid bias
        random.shuffle(self.devices)
        for device in self.devices:
            device.gossip()
        #time.sleep(1)  # Adjust the interval as needed
        #print("__________________________")

    def run_dkmeans(self):
        for i in range(configuration.DKMEANS_ITERATIONS):
            for device in self.devices:
                device.assign_clusters()
            
            for device in self.devices:
                device.update_cluster_heads()
    
    def run_local_training(self):
        for device in self.devices:
            device.local_training()
        