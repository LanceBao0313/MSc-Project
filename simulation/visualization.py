import pyglet
from pyglet import shapes
from pyglet.window import key
import random
import configuration

class Visualization:
    def __init__(self, network):
        self.network = network
        self.window = pyglet.window.Window(configuration.WINDOW_SIZE, configuration.WINDOW_SIZE)
        pyglet.gl.glClearColor(*configuration.BACKGROUND_COLOUR_GL)
        self.window.clear()
        self.batch = pyglet.graphics.Batch()
        self.circles = []
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
        
        self.network.run_gossip_protocol()

        self.draw_devices()
        self.counter += 1
        input("")

    def draw_devices(self):
        self.circles.clear()
        #self.batch = pyglet.graphics.Batch()  # Clear previous batch
        for device in self.network.devices:
            x, y = device.x * 8, device.y * 8
            if device.is_head:
                circle = shapes.Triangle(x, y, x + 10, y, x + 5, y + 10, batch=self.batch, color=device.color)
            else:
                circle = shapes.Circle(x, y, 5, batch=self.batch, color=device.color)
            
            self.circles.append(circle)


    def run(self):
        pyglet.clock.schedule_interval(self.update, 1/configuration.UPDATE_RATE)
        pyglet.app.run()