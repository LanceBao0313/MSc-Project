import pyglet
from pyglet import shapes
from pyglet.window import key
import random
import configuration

class Visualization:
    def __init__(self, network):
        self.network = network
        self.window = pyglet.window.Window(configuration.WINDOW_SIZE, configuration.WINDOW_SIZE)
        # Set the background colour
        pyglet.gl.glClearColor(*configuration.BACKGROUND_COLOUR_GL)
        self.window.clear()
        self.batch = pyglet.graphics.Batch()
        self.circles = []

        # To handle the on_draw event correctly
        @self.window.event
        def on_draw():
            if configuration.GRAPHICS_ON:
                self.window.clear()
                self.batch.draw()
                # Draw the window
                # self.window.flip()

        # To handle key events for closing the window
        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol == key.ESCAPE:
                self.window.close()

    def update(self, dt):
        self.network.reset_paired_devices()
        self.network.update_device_positions()
        self.network.update_in_range_devices()
        self.network.run_form_clusters()
        self.network.run_gossip_protocol()

        self.draw_devices()

    def draw_devices(self):
        self.circles.clear()
        #self.batch = pyglet.graphics.Batch()  # Clear previous batch
        for device in self.network.devices:
            x, y = device.x * 8, device.y * 8
            circle = shapes.Circle(x, y, 5, batch=self.batch, color=device.color)
            self.circles.append(circle)

    def run(self):
        pyglet.clock.schedule_interval(self.update, 1/10.0)
        pyglet.app.run()