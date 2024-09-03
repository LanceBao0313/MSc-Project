import pyglet
from pyglet import shapes
from pyglet.window import key
import random
import configuration
import time
import signal
import sys
from eval import run_evaluation

random.seed(configuration.RANDOM_SEED)

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
        self.counter = 0

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
        # if self.counter == 0:
        self.network.run_local_training()
        # self.network.reset_gossip_counters()

        print(f"Running gossip protocols in loop: {self.counter}")
        # Inner-cluster communication
        self.network.reset_gossip_counters()
        for _ in range(configuration.INNER_GOSSIP_ITERATIONS):
            
            self.network.reset_paired_devices()
            self.network.run_inner_gossip_comm()
        # print("______________________________________________________________________________")
        time.sleep(1) 
        self.network.aggregate_models()
        # # Inter-cluster communication
        # self.network.reset_gossip_counters()
        # for _ in range(configuration.INTER_GOSSIP_ITERATIONS):
        #     self.network.reset_paired_devices()
        #     self.network.run_inter_gossip_comm()
        
        # print("aggregating the models")
        # self.network.aggregate_models()
        self.network.clear_seen_devices()

        run_evaluation(0.5)
        print("time: ", time.time() - self.start_time)
        # eval every 10 minutes
        # if time.time() - self.start_time > 600:
        #     run_evaluation(1.0)
        #     self.start_time = time.time()

        self.draw_devices()

        # i = input("Press Enter to continue...")
        #save image
        
        self.counter += 1

    def draw_devices(self):
        # pyglet.image.get_buffer_manager().get_color_buffer().save(f"../graphs/image_{self.counter}.png")
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
                                        color=(0, 0, 0, 255))
            self.labels.append(label)

    def run(self):
        start_time = time.time()
         # Define the signal handler
        def signal_handler(sig, frame):
            run_time = time.time() - start_time
            print(f"Program run time: {run_time:.2f} seconds")
            self.network.save_device_location()
            sys.exit(0)  # Ensure the program exits after printing

        # Set the signal handler for SIGINT (Ctrl+C)
        signal.signal(signal.SIGINT, signal_handler)        
        pyglet.clock.schedule_interval(self.update, 1/configuration.UPDATE_RATE)
        pyglet.app.run()