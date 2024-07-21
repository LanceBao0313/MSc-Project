from network import Network
from visualization import Visualization
import pyglet
from configuration import DATA_PATH, NUMBER_OF_DEVICES, BATCH_SIZE
import dataset
import torch.multiprocessing as mp

def main():
    number_of_clusters = NUMBER_OF_DEVICES // 5
    # dataloaders = dataset.get_federated_dataloaders(DATA_PATH, NUMBER_OF_DEVICES, 
    #                                                 NUMBER_OF_CLASSES_PER_CLIENT, BATCH_SIZE)
    dataloaders = dataset.get_nonIID_dataloader(DATA_PATH, train=True)
    network = Network(NUMBER_OF_DEVICES, number_of_clusters, dataloaders)
    mp.set_start_method('spawn')
    Visualization(network).run()
    
if __name__ == "__main__":
    main()