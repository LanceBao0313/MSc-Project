from network import Network
from visualization import Visualization
import pyglet
from configuration import DATA_PATH, NUMBER_OF_DEVICES, NUMBER_OF_CLASSES_PER_CLIENT, BATCH_SIZE
import dataset

def main():
    number_of_clusters = NUMBER_OF_DEVICES // 5
    dataloaders = dataset.get_federated_dataloaders(DATA_PATH, NUMBER_OF_DEVICES, 
                                                    NUMBER_OF_CLASSES_PER_CLIENT, BATCH_SIZE)

    network = Network(NUMBER_OF_DEVICES, number_of_clusters, dataloaders)
    Visualization(network).run()
    
if __name__ == "__main__":
    main()