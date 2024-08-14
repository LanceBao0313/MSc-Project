from device import Device
import random
import configuration
import threading
import time
import torch.multiprocessing as mp
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from model import get_parameters

random.seed(configuration.RANDOM_SEED)
torch.manual_seed(configuration.RANDOM_SEED)
torch.cuda.manual_seed(configuration.RANDOM_SEED)

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

    def run_inner_gossip_comm(self):
        # self.reset_gossip_counters()
        # shaffle the devices to avoid bias
        random.shuffle(self.devices)
        for device in self.devices:
            device.inner_gossip()
        #time.sleep(1)  # Adjust the interval as needed
        #print("__________________________")
    
    def run_inter_gossip_comm(self):
        # self.reset_gossip_counters()
        random.shuffle(self.devices)
        for device in self.devices:
            device.inter_gossip()
        #time.sleep(1)  # Adjust the interval as needed
        #print("__________________________")
    
    def reset_gossip_counters(self):
        for device in self.devices:
            device.gossip_counter = 1
        

    def run_dkmeans(self):
        for i in range(configuration.DKMEANS_ITERATIONS):
            for device in self.devices:
                device.assign_clusters()
            
            for device in self.devices:
                device.update_cluster_heads()
    
    def run_local_training(self):
        pbar = tqdm(self.devices, desc="Training devices")
        for counter, device in enumerate(pbar, start=1):
            loss, acc = device.local_training()
            pbar.set_description(f"loss {loss:.4f}, accuracy {acc:.4f}")

    def train_device(self, device):
        device.local_training()

    def parallel_training(self):
        with mp.Pool(processes=torch.cuda.device_count()) as pool:  # Use the number of GPUs available
            pool.map(self.train_device, self.devices)

    def save_device_location(self):
        # Save the results to a CSV file
        with open('device_locations.csv', 'w') as f:
            f.write('id,x,y\n')
            for device in self.devices:
                f.write(f'{device.id},{device.x},{device.y}\n')

    def aggregate_models(self):
        for device in self.devices:
            device.aggregate_model()
        
    def clear_seen_devices(self):
        for device in self.devices:
            # print(f"Device {device.id} has seen devices {len(device.seen_devices)}")
            device.seen_devices = []