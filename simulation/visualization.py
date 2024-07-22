import pyglet
from pyglet import shapes
from pyglet.window import key
import random
import configuration
import time
import signal
import sys
from eval import run_evaluation

class Visualization:
    def __init__(self, network):
        self.network = network
        self.window = pyglet.window.Window(configuration.WINDOW_SIZE, configuration.WINDOW_SIZE)
        pyglet.gl.glClearColor(*configuration.BACKGROUND_COLOUR_GL)
        self.window.clear()
        self.batch = pyglet.graphics.Batch()
        self.circles = []
        self.labels = []
        self.start_time = time.time()
        # self.counter = 0

        # To handle the on_draw event correctly
        @self.window.event
        def on_draw():
            if configuration.GRAPHICS_ON:
                self.window.clear()
                self.batch.draw()

        

        # To handle key events for closing the window
        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol == key.ESCAPE:
                print("Closing the window")
                self.window.close()

    def update(self, dt):
        # Update physical environment
        self.network.reset_paired_devices()
        self.network.update_device_positions()
        self.network.update_in_range_devices()
        # DK-means
        self.network.run_dkmeans()
        # Federated learning
        self.network.run_local_training()
        
        self.network.reset_gossip_counters()

        # Inner-cluster communication
        for _ in range(configuration.INNER_GOSSIP_ITERATIONS):
            print(f"Running Inner-cluster protocol: {_ + 1}")
            self.network.reset_paired_devices()
            self.network.run_inner_gossip_comm()
        
        # Inter-cluster communication
        # for _ in range(configuration.INTER_GOSSIP_ITERATIONS):
        #     print(f"Running inter-cluster protocol: {_ + 1}")
        #     self.network.reset_paired_devices()
        #     self.network.run_inter_gossip_comm()

        

        # eval every 10 minutes
        if time.time() - self.start_time > 600:
            run_evaluation()
            self.start_time = time.time()

        self.draw_devices()

        # i = input("Press Enter to continue...")
        # self.counter += 1

    def draw_devices(self):
        self.circles.clear()
        self.labels.clear()
        #self.batch = pyglet.graphics.Batch()  # Clear previous batch
        for device in self.network.devices:
            x, y = device.x * 8, device.y * 8
            if device.is_head:
                circle = shapes.Triangle(x, y+10, x-8, y-6, x+8, y-6, batch=self.batch, color=device.color)
            else:
                circle = shapes.Circle(x, y, 8, batch=self.batch, color=device.color)
            
            self.circles.append(circle)            
        
            label = pyglet.text.Label(str(device.id),
                                        font_size=13,
                                        x=x,
                                        y=y,
                                        anchor_x='center',
                                        anchor_y='center',
                                        batch=self.batch,
                                        color=(255, 255, 255, 255))
            self.labels.append(label)

    def run(self):
        start_time = time.time()
         # Define the signal handler
        def signal_handler(sig, frame):
            run_time = time.time() - start_time
            print(f"Program run time: {run_time:.2f} seconds")
            sys.exit(0)  # Ensure the program exits after printing

        # Set the signal handler for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, signal_handler)        
        pyglet.clock.schedule_interval(self.update, 1/configuration.UPDATE_RATE)
        pyglet.app.run()