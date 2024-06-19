from network import Network
from visualization import Visualization
import pyglet
import configuration

def main():
    num_devices = 50
    comm_range = 10  # Communication range
    number_of_clusters = num_devices // 5
    network = Network(num_devices, comm_range, number_of_clusters)
    Visualization(network).run()
    
if __name__ == "__main__":
    main()