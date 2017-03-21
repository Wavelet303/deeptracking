"""
    Opengl based rendering tool

    date : 2017-20-03
"""
import glfw

__author__ = "Mathieu Garon"
__version__ = "0.0.1"

from OpenGL.GL import shaders
from deeptracking.data.glew import *
# import import_export.PLYReader as ply
import numpy as np


class ModelRenderer():
    def __init__(self, model_path, shader_path, camera, window):
        self.vertices, self.faces = ply.read(model_path)
        self.camera = camera
        self.window = window
        # Opengl Flags
        glEnable(GL_DEPTH_TEST)
        glDepthMask(GL_TRUE)
        glDepthFunc(GL_LESS)
        glDepthRange(0.0, 1.0)

        self.texcoord_buffer = None

        self.setup_shaders(shader_path)
        self.setup_buffers(self.vertices, self.faces)
        self.setup_attributes()
        self.setup_camera()

    def setup_shaders(self, shader_path):
        with open(os.path.join(shader_path, "vertex_light.txt"), 'r') as myfile:
            vertex_shader_data = myfile.read()
        VERTEX_SHADER = shaders.compileShader(vertex_shader_data, GL_VERTEX_SHADER)
        with open(os.path.join(shader_path, "fragment_light.txt"), 'r') as myfile:
            fragment_shader_data = myfile.read()
        FRAGMENT_SHADER = shaders.compileShader(fragment_shader_data, GL_FRAGMENT_SHADER)
        self.shader_program = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)
        glLinkProgram(self.shader_program)
        glUseProgram(self.shader_program)

    def setup_buffers(self, pointcloud, face_index):
        # ---- Setup data ----
        color = pointcloud["RGB"].astype(np.float32) / 255.
        vertex = pointcloud["XYZ"].astype(np.float32)
        ambiant_occlusion = np.ones(pointcloud["XYZ"].shape, dtype=np.float32)
        normals = pointcloud["UVW"].astype(np.float32)
        normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]

        self.vertex_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, vertex.nbytes, vertex, GL_STATIC_DRAW)

        self.color_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glBufferData(GL_ARRAY_BUFFER, color.nbytes, color, GL_STATIC_DRAW)

        self.normal_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buffer)
        glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)

        self.ambiant_occlusion_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.ambiant_occlusion_buffer)
        glBufferData(GL_ARRAY_BUFFER, ambiant_occlusion.nbytes, ambiant_occlusion, GL_STATIC_DRAW)

        try:
            texcoord = pointcloud["ST"].astype(np.float32)
            self.texcoord_buffer = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.texcoord_buffer)
            glBufferData(GL_ARRAY_BUFFER, texcoord.nbytes, texcoord, GL_STATIC_DRAW)
        except KeyError:
            pass

        self.buffer_index = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffer_index)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, face_index.nbytes, face_index, GL_STATIC_DRAW)

        # load a default texture by default else get the pointcloud's texture
        self.texture = glGenTextures(1)
        tex = np.ones((1, 1, 4), dtype=np.uint8)
        tex.fill(255)
        if self.vertices.texture is not None:
            tex = self.vertices.texture[::-1, :, :]
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glEnable(GL_TEXTURE_2D)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, tex.shape[1], tex.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, tex)
        glGenerateMipmap(GL_TEXTURE_2D)

    def setup_attributes(self):
        loc = glGetAttribLocation(self.shader_program, "position")
        glEnableVertexAttribArray(loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glVertexAttribPointer(loc, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))

        loc = glGetAttribLocation(self.shader_program, "color")
        glEnableVertexAttribArray(loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.color_buffer)
        glVertexAttribPointer(loc, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))

        loc = glGetAttribLocation(self.shader_program, "normal")
        glEnableVertexAttribArray(loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buffer)
        glVertexAttribPointer(loc, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))

        loc = glGetAttribLocation(self.shader_program, "ambiant_occlusion")
        glEnableVertexAttribArray(loc)
        glBindBuffer(GL_ARRAY_BUFFER, self.ambiant_occlusion_buffer)
        glVertexAttribPointer(loc, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))

        if self.texcoord_buffer:
            loc = glGetAttribLocation(self.shader_program, "texcoords")
            glEnableVertexAttribArray(loc)
            glBindBuffer(GL_ARRAY_BUFFER, self.texcoord_buffer)
            glVertexAttribPointer(loc, 2, GL_FLOAT, False, 0, ctypes.c_void_p(0))

        # ---- Setup Uniforms ----
        self.uniform_locations = {
            'view': glGetUniformLocation(self.shader_program, 'view'),
            'proj': glGetUniformLocation(self.shader_program, 'proj'),
            'ambientLightForce': glGetUniformLocation(self.shader_program, 'ambientLightForce'),
            'lightA_direction': glGetUniformLocation(self.shader_program, 'lightA.direction'),
            'lightA_diffuse': glGetUniformLocation(self.shader_program, 'lightA.diffuse'),
            'lightB_direction': glGetUniformLocation(self.shader_program, 'lightB.direction'),
            'lightB_diffuse': glGetUniformLocation(self.shader_program, 'lightB.diffuse')
        }

        glUniform3f(self.uniform_locations['lightA_direction'], -1, -1, 1)
        glUniform3f(self.uniform_locations['lightA_diffuse'], 1, 1, 1)

        glUniform3f(self.uniform_locations['lightB_direction'], 1, 1, 1)
        glUniform3f(self.uniform_locations['lightB_diffuse'], 0, 0, 0)

        glUniform3f(self.uniform_locations['ambientLightForce'], 0.65, 0.65, 0.65)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffer_index)  # bind faces buffer

    def load_ambiant_occlusion_map(self, path):
        try:
            vertices, faces = ply.read(path)
            ambiant_occlusion = vertices["RGB"].astype(np.float32) / 255
            glBindBuffer(GL_ARRAY_BUFFER, self.ambiant_occlusion_buffer)
            glBufferData(GL_ARRAY_BUFFER, ambiant_occlusion.nbytes, ambiant_occlusion, GL_STATIC_DRAW)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.buffer_index)  # bind faces buffer
        except FileNotFoundError:
            print("ViewpointRender, ambiant occlusion file not found ... continue with basic render")

    def setup_camera(self):
        self.near_plane = 0.1
        self.far_plane = 2

        # see : http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
        proj = np.array([[self.camera.focal_x, 0, -self.camera.center_x, 0],
                         [0, self.camera.focal_y, -self.camera.center_y, 0],
                         [0, 0, self.near_plane + self.far_plane, self.near_plane * self.far_plane],
                         [0, 0, -1, 0]])
        self.projection_matrix = ModelRenderer.perspectiveMatrix(0.0,
                                                                 self.camera.width,
                                                                 self.camera.height,
                                                                 0.0,
                                                                 self.near_plane,
                                                                 self.far_plane).dot(proj).T

        glUniformMatrix4fv(self.uniform_locations['proj'], 1, GL_FALSE, self.projection_matrix)

    @staticmethod
    def perspectiveMatrix(left, right, bottom, top, near, far):
        right = float(right)
        left = float(left)
        top = float(top)
        bottom = float(bottom)
        mat = np.array([[2. / (right - left), 0, 0, -(right + left) / (right - left)],
                        [0, 2. / (top - bottom), 0, -(top + bottom) / (top - bottom)],
                        [0, 0, -2 / (far - near), -(far + near) / (far - near)],
                        [0, 0, 0, 1]], dtype=np.float32)
        return mat

    def gldepth_to_worlddepth(self, frame):
        A = self.projection_matrix[2, 2]
        B = self.projection_matrix[3, 2]
        distance = B / (frame * -2.0 + 1.0 - A) * -1
        idx = distance[:, :] >= B / (A + 1)
        distance[idx] = 0
        return (distance * 1000).astype(np.uint16)

    def render(self, view_transform, light_direction=None, light_diffuse=None):
        light_normal = np.ones(4)
        if light_direction is None:
            light_direction = np.array([0, 0.1, -0.9])
        light_normal[0:3] = light_direction
        light_direction = np.dot(view_transform.inverse().matrix, light_normal)
        glUniform3f(self.uniform_locations['lightA_direction'], light_direction[0], light_direction[1],
                    light_direction[2])

        if light_diffuse is None:
            light_diffuse = np.array([0.4, 0.4, 0.4])
        glUniform3f(self.uniform_locations['lightA_diffuse'], light_diffuse[0], light_diffuse[1], light_diffuse[2])

        glUniformMatrix4fv(self.uniform_locations['view'], 1, GL_FALSE, view_transform.matrix)

        # --- draw window ---
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glEnable(GL_MULTISAMPLE)
        glEnable(GL_TEXTURE_2D)
        glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)

        glDrawElements(GL_TRIANGLES, len(self.faces) * 3, GL_UNSIGNED_INT, ctypes.c_void_p(0))

        # -- retrieve data
        depth_array = glReadPixels(0, 0, self.camera.width, self.camera.height, GL_DEPTH_COMPONENT, GL_FLOAT)
        depth_array = depth_array.reshape((self.camera.height, self.camera.width))
        depth_array = self.gldepth_to_worlddepth(depth_array)
        rgb_array = glReadPixels(0, 0, self.camera.width, self.camera.height, GL_RGB, GL_UNSIGNED_BYTE)
        rgb_array = np.frombuffer(rgb_array, dtype=np.uint8).reshape((self.camera.height, self.camera.width, 3))
        return rgb_array, depth_array


def InitOpenGL(width, height, hide_window=True):
    """
    dpy = Display()
    conf = pegl.config.get_configs(dpy, {'RENDERABLE_TYPE': ClientAPIs(OPENGL=1),
                                         "SURFACE_TYPE": SurfaceTypes(PBUFFER=1),
                                         "BLUE_SIZE": 8,
                                         "GREEN_SIZE": 8,
                                         "RED_SIZE": 8,
                                         "DEPTH_SIZE": 8})
    conf = conf[0]
    surf = pegl.surface.PbufferSurface(dpy, conf, {'WIDTH': width, 'HEIGHT': height})

    pegl.context.bind_api(ContextAPIs.OPENGL)
    ctx = pegl.context.Context(dpy, conf)
    ctx.make_current(draw_surface=surf)
    """
    if not glfw.init():
        print("Failed to initialize GLFW\n", file=sys.stderr)
        sys.exit(-1)
    window = glfw.create_window(width, height, "ViewpointRender", None, None)
    if not window:
        print(
            "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n",
            file=sys.stderr)
        glfw.terminate()
        sys.exit(-1)
    glfw.make_context_current(window)

    glewExperimental = True
    if glewInit() != GLEW_OK:
        print("Failed to initialize GLEW\n", file=sys.stderr)
        sys.exit(-1)
    glClearColor(0, 0, 0, 0)
    window = None
    return window
