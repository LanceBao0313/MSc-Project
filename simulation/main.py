from network import Network
from visualization import Visualization
import pyglet
import configuration

def main():
    num_devices = configuration.NUMBER_OF_DEVICES
    number_of_clusters = num_devices // 5
    network = Network(num_devices, number_of_clusters)
    Visualization(network).run()
    
if __name__ == "__main__":
    main()