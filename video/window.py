from contextlib import contextmanager
import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable
import pycuda.driver
from pycuda.gl import graphics_map_flags
from glumpy import app, gloo, gl


class GLWindow():
    def __init__(self, init_f, per_frame_f, width=512, height=512):
        """
        constructor for the GLWindow object
        :param init_f: function to initialize window with, should return a float tensor in format:
                       [batch_size, channels, height, width] with values between -1 and 1
        :param per_frame_f: function run per frame to update window,
                            takes current frame as argument and should return then next frame
                            as a float tensor in format: [batch_size, channels, height, width]
        :param width: Width of the window in pixels
        :param height: Height of the window in pixels
        """

        # create window with OpenGL context
        self.init_f = init_f
        self.per_frame_f = per_frame_f

        app.use('glfw')
        window = app.Window(width, height, fullscreen=False)

        self.window = window
        self.setup()

        @window.event
        def on_draw(dt):
            global state
            self.window.set_title(str(self.window.fps).encode("ascii"))

            tex = screen['tex']
            h,w = tex.shape[:2]

            # mutate state in torch
            img = self.per_frame_f(state).detach() # prevent autograd from filling memory

            # convert into proper format
            tensor = img.squeeze().transpose(0,2)
            tensor = tensor.transpose(0,1).data # put in texture order
            tensor = torch.cat((tensor, tensor[:,:,0].unsqueeze(2)), 2) # add the alpha channel
            tensor[:,:,3] = 1 # set alpha

            # check that tensor order matches texture:
            tensor = (255*tensor).byte().contiguous() # convert to ByteTensor

            # copy from torch into buffer
            assert tex.nbytes == tensor.numel()*tensor.element_size()
            with self.cuda_activate(cuda_buffer) as ary:
                cpy = pycuda.driver.Memcpy2D()
                cpy.set_src_device(tensor.data_ptr())
                cpy.set_dst_array(ary)
                cpy.width_in_bytes = cpy.src_pitch = cpy.dst_pitch = tex.nbytes // h
                cpy.height = h
                cpy(aligned=False)
                torch.cuda.synchronize()

            # draw to screen
            self.window.clear()
            screen.draw(gl.GL_TRIANGLE_STRIP)


    def setup(self):
        global screen, cuda_buffer, state
        w, h = self.window.get_size()

        # setup pycuda and torch
        import pycuda.gl.autoinit
        import pycuda.gl
        assert torch.cuda.is_available()
        print('using GPU {}'.format(torch.cuda.current_device()))

        state = torch.cuda.FloatTensor(1,3,h,w)
        state = self.init_f(state)

        # create a buffer with pycuda and gloo views
        tex, cuda_buffer = self.create_shared_texture(w,h)

        # create a shader to program to draw to the screen
        vertex = """
        uniform float scale;
        attribute vec2 position;
        attribute vec2 texcoord;
        varying vec2 v_texcoord;
        void main()
        {
            v_texcoord = texcoord;
            gl_Position = vec4(scale*position, 0.0, 1.0);
        } """
        fragment = """
        uniform sampler2D tex;
        varying vec2 v_texcoord;
        void main()
        {
            gl_FragColor = texture2D(tex, v_texcoord);
        } """

        # Build the program and corresponding buffers (with 4 vertices)
        screen = gloo.Program(vertex, fragment, count=4)

        # Upload data into GPU
        screen['position'] = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
        screen['texcoord'] = [(0,0), (0,1), (1,0), (1,1)]
        screen['scale'] = 1.0
        screen['tex'] = tex


    def create_shared_texture(self, w, h, c=4, flags=graphics_map_flags.WRITE_DISCARD, dtype=np.uint8):
        """Create and return a Texture2D with gloo and pycuda views."""
        tex = np.zeros((h,w,c), dtype).view(gloo.Texture2D)
        tex.activate() # force gloo to create on GPU
        tex.deactivate()
        cuda_buffer = pycuda.gl.RegisteredImage(int(tex.handle), tex.target, flags)
        return tex, cuda_buffer


    @contextmanager
    def cuda_activate(self, img):
        """Context manager simplifying use of pycuda.gl.RegisteredImage"""
        mapping = img.map()
        yield mapping.array(0,0)
        mapping.unmap()


    def run(self):
        app.run()
