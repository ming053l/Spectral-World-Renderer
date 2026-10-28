"""The viser GUI of `abgs viewer` (in the manner of the nerfstudio viewer): every client gets a render thread that
renders its camera with the session (a low-resolution preview while the camera moves, then the full resolution) and
shows the result as the scene background; the panel binds the session's modes, queries, inspection, jobs and exports.

Tabs: Scene (scene, checkpoint, dataset cameras), Render (display mode), Query (material queries, threshold,
highlight), Endmembers (dictionary, class map, on/off), Inspect (spectrum readout and the hits of a clicked pixel),
Train and Eval (abgs subprocesses with live progress), Export (PNG, spectral cube, camera-path video, .ply).
Every handler is a method, so the GUI can be driven without a browser (tests set handle values and call methods).
"""
import functools
import json
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import viser
import viser.transforms as vtf

from .. import scenes
from . import jobs as J
from . import queries as Q
from .session import FALSE_COLOUR, MODES, Camera, interpolate_path, save_png

QUERY_TYPES = ['material name', 'endmember', 'library spectrum', 'uploaded spectrum (CSV)', 'click (pick a pixel)',
               'band index', 'Boolean combination', 'instances (DBSCAN)']
HANDLING = {'library': ['auto (illumination cone)', 'none (spectral angle)', 'sunlit radiance'],
            'click': ['auto (whitening)', 'cone {hR, h, h/R}', 'none (spectral angle)']}
CLICK_ACTIONS = ['readout + hits', 'query by click']


def guarded(method):
    """GUI callbacks: an exception is shown in the status line instead of ending the callback thread."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as error:                      # noqa: BLE001 (reported to the user)
            self.status(f'**error** in {method.__name__}: {error}', error=True)
            traceback.print_exc()
    return wrapper


class ClientView:
    """Render loop of one browser client."""

    def __init__(self, app, client, view=None):
        self.app, self.client, self.view = app, client, view
        self.dirty, self.moved, self.alive, self.frame = True, 0., True, None
        client.camera.on_update(self.touch)
        threading.Thread(target=self.loop, daemon=True).start()

    def ready(self, timeout=10.):
        """Wait for the client's first camera message (a pose set before it would be overwritten)."""
        end = time.time() + timeout
        while time.time() < end:
            try:
                return self.client.camera.update_timestamp > 0
            except AssertionError:
                time.sleep(.05)
        return False

    def touch(self, _=None):
        self.moved, self.dirty = time.time(), True

    def loop(self):
        if self.ready() and self.view is not None:
            self.app.snap_to(self.view, self.client)
        while self.alive:
            if not self.dirty or self.app.session.model is None:
                time.sleep(.02)
                continue
            settling = time.time() - self.moved < .3
            try:
                camera, view = self.app.client_camera(self.client, self.app.preview_side.value if settling else self.app.max_side.value)
                frame = self.app.session.render(camera, view)
                image = self.app.display(frame)
                self.client.scene.set_background_image(image, format='jpeg', jpeg_quality=88, depth=self.app.depth(frame))
                self.frame = self.app.frame = frame
                self.app.fps.content = f'{camera.width}x{camera.height}, {1000 * frame.seconds:.0f} ms render'
            except Exception as error:                  # noqa: BLE001
                self.app.status(f'**render error**: {error}', error=True)
                traceback.print_exc()
                self.dirty = False
                continue
            if not settling:
                self.dirty = False


class ViewerApp:
    def __init__(self, session, host='127.0.0.1', port=7007, registries=(), jobs_dir=None):
        self.session = S = session
        self.registry = {}
        for path in registries:
            self.registry.update(json.loads(Path(path).read_text()))
        self.jobs_dir = Path(jobs_dir or S.workdir / 'jobs')
        self.jobs, self.keyframes, self.views, self.snap = [], [], {}, {}
        self.frame, self.inspection, self.upload, self.followed = None, None, None, None
        self.server = viser.ViserServer(host=host, port=port, label='SWR viewer', verbose=False)
        self.camera = S.camera(S.pack.test[0] if S.pack.test else 0)
        self.camera_view = S.pack.test[0] if S.pack.test else 0
        self.build()
        self.scene_objects()
        self.server.on_client_connect(self.connect)
        self.server.on_client_disconnect(lambda client: self.views.pop(client.client_id, None))
        self.server.scene.on_click()(self.on_click)
        threading.Thread(target=self.watch_jobs, daemon=True).start()

    # ------------------------------------------------------------ GUI
    def build(self):
        g = self.server.gui
        S = self.session
        self.status_md = g.add_markdown('ready')
        self.fps = g.add_markdown('')
        tabs = g.add_tab_group()
        with tabs.add_tab('Scene', viser.Icon.CUBE):
            self.scene_name = g.add_dropdown('scene', scenes.names(), initial_value=S.scene or scenes.names()[0])
            self.ckpt = g.add_dropdown('checkpoint', self.checkpoints(S.scene), initial_value='paper model')
            self.ckpt_path = g.add_text('or model path', '')
            g.add_button('Load', icon=viser.Icon.FOLDER_OPEN).on_click(lambda _: self.load())
            self.scene_name.on_update(lambda _: self.refresh_checkpoints())
            self.info = g.add_markdown('')
            with g.add_folder('Dataset cameras'):
                self.view_name = g.add_dropdown('view', self.view_options())
                g.add_button('Snap to view', icon=viser.Icon.CAMERA).on_click(lambda _: self.snap_to(self.view_index(self.view_name.value)))
                self.show_cameras = g.add_checkbox('show cameras', True)
                self.show_cameras.on_update(lambda _: [setattr(h, 'visible', self.show_cameras.value) for h in self.frustums])
            with g.add_folder('Renderer'):
                self.renderer = g.add_dropdown('renderer', ['fused', 'fast', 'reference'], initial_value=S.renderer)
                self.renderer.on_update(self.set_renderer)
                self.max_side = g.add_slider('resolution (long side)', 256, 1920, 32, 960)
                self.preview_side = g.add_slider('preview while moving', 128, 960, 32, 384)
                self.max_side.on_update(lambda _: self.rerender())
        with tabs.add_tab('Render', viser.Icon.PHOTO):
            wl = S.world.wavelengths
            self.mode = g.add_dropdown('mode', MODES, initial_value='pseudo-RGB')
            self.wavelength = g.add_slider('wavelength (nm)', float(wl[0]), float(wl[-1]), 1., float(np.clip(700, wl[0], wl[-1])))
            self.fc_preset = g.add_dropdown('false colour', list(FALSE_COLOUR))
            self.endmember = g.add_dropdown('endmember', self.endmember_options())
            self.cmap = g.add_dropdown('colour map', ['viridis', 'inferno', 'gray', 'turbo', 'magma'])
            self.exposure = g.add_slider('exposure (EV)', -4., 4., .1, 0.)
            self.legend = g.add_html('')
            for h in (self.mode, self.wavelength, self.fc_preset, self.endmember, self.cmap, self.exposure):
                h.on_update(lambda _: self.rerender())
            self.mode.on_update(lambda _: self.show_legend())
        with tabs.add_tab('Query', viser.Icon.SEARCH):
            self.qtype = g.add_dropdown('query type', QUERY_TYPES)
            self.q_class = g.add_dropdown('material name', self.class_options())
            self.q_em = g.add_dropdown('endmember ', self.endmember_options())
            self.q_lib = g.add_dropdown('library spectrum', self.library_options())
            with g.add_folder('CSV spectrum') as self.q_upload_folder:   # the folder hides it (upload buttons ignore visible)
                self.q_upload = g.add_upload_button('upload CSV spectrum', icon=viser.Icon.UPLOAD, mime_type='.csv,text/csv,text/plain')
                self.q_upload.on_upload(self.on_upload)
                self.q_upload_md = g.add_markdown('(wavelength, value) rows or one value per band')
            self.q_handling = g.add_dropdown('illumination handling', HANDLING['library'])
            self.q_index_preset = g.add_dropdown('index preset', list(Q.INDEX_PRESETS))
            self.q_index = g.add_text('index formula', list(Q.INDEX_PRESETS.values())[0])
            self.q_rule = g.add_dropdown('select where index', ['>=', '<='])
            self.q_expr = g.add_text('Boolean expression', 'q1 AND NOT q2')
            self.q_base = g.add_dropdown('instances of', ['-'])
            self.q_eps = g.add_number('DBSCAN radius', .6, min=.01, step=.05)
            self.q_min_size = g.add_number('min Gaussians / instance', 40, min=1, step=1)
            self.q_min_height = g.add_number('min height above ground', 0., min=0., step=.1)
            self.q_spectral = g.add_dropdown('keep instances matching', ['-'])
            self.q_run = g.add_button('Run query', icon=viser.Icon.PLAYER_PLAY)
            self.q_run.on_click(lambda _: self.run_query())
            self.q_index_preset.on_update(lambda _: setattr(self.q_index, 'value', Q.INDEX_PRESETS[self.q_index_preset.value]))
            self.qtype.on_update(lambda _: self.query_fields())
            with g.add_folder('Selection'):
                self.q_active = g.add_dropdown('active query', ['-'])
                self.q_active.on_update(lambda _: self.activate(self.q_active.value))
                self.q_threshold = g.add_slider('threshold', 0., 1., .001, .5)
                self.q_threshold.on_update(lambda _: self.set_threshold())
                g.add_button('default threshold (Otsu / 0.5)').on_click(lambda _: self.reset_threshold())
                self.q_overlay = g.add_checkbox('highlight in render', True)
                self.q_points = g.add_checkbox('selected Gaussians as 3D points', True)
                self.q_point_size = g.add_slider('point size', .002, .2, .002, .012)
                self.q_colour = g.add_rgb('highlight colour', (255, 0, 255))
                for h in (self.q_overlay, self.q_points, self.q_point_size, self.q_colour):
                    h.on_update(lambda _: self.show_selection())
                g.add_button('delete active query').on_click(lambda _: self.delete_query())
            self.q_md = g.add_markdown('')
            self.q_hist = g.add_image(np.zeros((8, 8, 3), np.uint8), label='histogram')
            self.q_list = g.add_markdown('')
        with tabs.add_tab('Endmembers', viser.Icon.CHART_LINE):
            self.em_plot = g.add_image(np.zeros((8, 8, 3), np.uint8), label='endmembers')
            with g.add_folder('on / off'):
                self.em_boxes = [g.add_checkbox(f'e{k} {S.mapping.name(k)}', True) for k in range(S.model.num_materials)]
                for k, box in enumerate(self.em_boxes):
                    box.on_update(lambda _, k=k: self.toggle_endmember(k))
                g.add_button('all on').on_click(lambda _: self.endmembers_only(None))
                self.em_class = g.add_dropdown('class', self.class_options())
                g.add_button('only this class').on_click(lambda _: self.endmembers_only(self.em_class.value))
            self.em_table = g.add_markdown('')
        with tabs.add_tab('Inspect', viser.Icon.POINTER):
            self.click_action = g.add_dropdown('click in the view', CLICK_ACTIONS)
            g.add_markdown('Click a pixel: rendered spectrum against the endmember contributions, and the Gaussians '
                           'along its ray with blend weights w_i and abundances a_i (render_hits).')
            self.readout_img = g.add_image(np.zeros((8, 8, 3), np.uint8), label='spectrum')
            self.hits_img = g.add_image(np.zeros((8, 8, 3), np.uint8), label='hits')
            self.hits_md = g.add_markdown('')
            g.add_button('save hits (.json)').on_click(lambda _: self.save_inspection())
        with tabs.add_tab('Train', viser.Icon.BARBELL):
            self.t_stage = g.add_dropdown('stage', [f'{k}: {v}' for k, v in J.STAGES.items() if k != 'eval'],
                                          initial_value='train: ' + J.STAGES['train'])
            self.t_dir = g.add_text('output directory', str(self.jobs_dir / 'run'))
            self.t_steps = g.add_number('steps (0: stage default)', 0, min=0, step=100)
            self.t_K = g.add_number('K (0: scene)', 0, min=0, step=1)
            self.t_live = g.add_number('preview every N updates', 500, min=0, step=100)
            self.t_inputs = g.add_checkbox('use earlier stages in the output directory', False)
            self.t_extra = g.add_text('extra options', '')
            g.add_button('Start', icon=viser.Icon.PLAYER_PLAY).on_click(lambda _: self.start_training())
            g.add_button('Stop', icon=viser.Icon.PLAYER_STOP).on_click(lambda _: self.stop_job('train'))
            self.t_follow = g.add_checkbox('refresh the view from the latest checkpoint', True)
            self.t_progress = g.add_progress_bar(0.)
            self.t_md = g.add_markdown('')
            self.t_loss = g.add_image(np.zeros((8, 8, 3), np.uint8), label='loss')
            self.t_log = g.add_markdown('')
        with tabs.add_tab('Eval', viser.Icon.TABLE):
            self.e_dir = g.add_text('output directory', str(self.jobs_dir / 'eval'))
            self.e_fast = g.add_checkbox('fast renderer (scores within 1e-3)', True)
            self.e_extra = g.add_text('extra options', '')
            g.add_button('Run abgs eval on the loaded model', icon=viser.Icon.PLAYER_PLAY).on_click(lambda _: self.start_eval())
            g.add_button('Stop', icon=viser.Icon.PLAYER_STOP).on_click(lambda _: self.stop_job('eval'))
            self.e_load = g.add_text('show results of', str(S.eval_dir or ''))
            g.add_button('Show').on_click(lambda _: self.show_eval(self.e_load.value))
            self.e_progress = g.add_progress_bar(0.)
            self.e_md = g.add_markdown('')
            self.e_table = g.add_markdown('')
        with tabs.add_tab('Export', viser.Icon.DOWNLOAD):
            self.x_dir = g.add_text('directory', str(S.workdir / 'exports'))
            self.x_native = g.add_checkbox('dataset resolution when snapped', True)
            g.add_button('PNG of the view').on_click(lambda _: self.export_png())
            self.x_radiance = g.add_checkbox('cube in radiance units (x normalization scale)', False)
            g.add_button('spectral cube (.npy + .mat)').on_click(lambda _: self.export_cube())
            g.add_button('selected Gaussians (.ply)').on_click(lambda _: self.export_ply())
            with g.add_folder('Camera path'):
                g.add_button('add keyframe (current camera)').on_click(lambda _: self.add_keyframe())
                g.add_button('clear keyframes').on_click(lambda _: self.clear_keyframes())
                self.x_frames = g.add_number('frames', 90, min=2, step=1)
                self.x_fps = g.add_number('fps', 30, min=1, step=1)
                self.x_side = g.add_slider('video long side', 256, 1920, 32, 960)
                g.add_button('render video (.mp4)').on_click(lambda _: self.export_video())
                self.x_progress = g.add_progress_bar(0.)
            self.x_md = g.add_markdown('')
        self.query_fields()
        self.refresh_panels()

    def status(self, text, error=False):
        self.status_md.content = text
        if error:
            for client in self.server.get_clients().values():
                client.add_notification('SWR viewer', text.replace('**', ''), color='red', auto_close_seconds=8.)

    # ------------------------------------------------------------ options
    def checkpoints(self, scene):
        options = ['paper model']
        options += [k for k in self.registry if scene and k.split('/')[0] == scene]
        options += [str(job.latest_model()) for job in self.jobs if job.stage in ('train', 'run') and job.latest_model()]
        return options

    def refresh_checkpoints(self):
        self.ckpt.options = self.checkpoints(self.scene_name.value)

    def view_options(self):
        p = self.session.pack
        return [f'{n} ({p.views[i]["split"]})' for i, n in enumerate(p.names)]

    def view_index(self, option):
        return self.session.pack.names.index(option.split(' ')[0])

    def endmember_options(self):
        S = self.session
        return [f'e{k} {S.mapping.name(k)}' for k in range(S.model.num_materials)]

    def class_options(self):
        return list(self.session.mapping.class_names) or ['(no class map)']

    def library_options(self):
        return [n for n, _, _ in self.session.world.library] or ['(no library)']

    def query_fields(self):
        """Show the fields of the chosen query type only."""
        t = self.qtype.value
        fields = {'material name': [self.q_class], 'endmember': [self.q_em],
                  'library spectrum': [self.q_lib, self.q_handling], 'uploaded spectrum (CSV)': [self.q_upload_folder, self.q_handling],
                  'click (pick a pixel)': [self.q_handling], 'band index': [self.q_index_preset, self.q_index, self.q_rule],
                  'Boolean combination': [self.q_expr], 'instances (DBSCAN)': [self.q_base, self.q_eps, self.q_min_size, self.q_min_height, self.q_spectral]}
        shown = {id(h) for h in fields[t]}
        for h in {id(h): h for hs in fields.values() for h in hs}.values():
            h.visible = id(h) in shown
        handling = HANDLING['click' if t.startswith('click') else 'library']
        if list(self.q_handling.options) != handling:
            self.q_handling.options = handling
        self.q_run.label = 'Arm: next click in the view queries' if t.startswith('click') else 'Run query'

    # ------------------------------------------------------------ scene and cameras
    def scene_objects(self):
        S = self.session
        p = S.pack
        c2w = np.stack([np.asarray(v['c2w']) for v in p.views])
        up = -c2w[:, :3, 1].mean(0)
        self.server.scene.set_up_direction(tuple(up / np.linalg.norm(up)))
        extent = float(np.linalg.norm(c2w[:, :3, 3].max(0) - c2w[:, :3, 3].min(0))) or 1.
        for handle in getattr(self, 'frustums', []):
            handle.remove()
        self.frustums = []
        for i, v in enumerate(p.views):              # TRAIN blue, TEST orange; a click snaps the camera
            cam = Camera.from_pack(p, i)
            handle = self.server.scene.add_camera_frustum(
                f'/cameras/{i}', fov=cam.fov, aspect=cam.aspect, scale=.012 * extent, thickness=1.5, thickness_units='screen',
                color=(40, 120, 255) if v['split'] == 'train' else (255, 120, 20),
                wxyz=vtf.SO3.from_matrix(cam.c2w[:3, :3]).wxyz, position=cam.c2w[:3, 3], visible=self.show_cameras.value)
            handle.on_click(lambda event, i=i: self.snap_to(i, event.client))
            self.frustums.append(handle)

    def connect(self, client):
        self.views[client.client_id] = ClientView(self, client, self.camera_view if self.camera_view >= 0 else None)

    def client_camera(self, client, side):
        """Camera of a client: the dataset camera when the client sits at a snapped view with its aspect, else a free
        pinhole camera with the client's pose, vertical field of view and aspect."""
        c = client.camera
        view = self.snap.get(client.client_id)
        if view is not None and np.allclose(c.position, np.asarray(self.session.pack.views[view]['c2w'])[:3, 3], atol=1e-4):
            cam = self.session.camera(view)
            if abs(cam.aspect - c.aspect) < .02 * cam.aspect:
                cam = cam.resized(side)
                self.camera, self.camera_view = cam, view
                return cam, view
        cam = Camera.from_pose(vtf.SO3(np.asarray(c.wxyz)).as_matrix(), np.asarray(c.position), c.fov, c.aspect, side)
        self.camera, self.camera_view = cam, -1
        return cam, -1

    @guarded
    def snap_to(self, view, client=None):
        """Move a client (or all, or the headless state) to a dataset camera."""
        cam = self.session.camera(view)
        self.camera, self.camera_view = cam.resized(self.max_side.value), view
        self.view_name.value = self.view_options()[view]
        clients = [client] if client is not None else list(self.server.get_clients().values())
        for c in clients:
            with c.atomic():
                c.camera.fov = cam.fov
                c.camera.wxyz = vtf.SO3.from_matrix(cam.c2w[:3, :3]).wxyz
                c.camera.position = cam.c2w[:3, 3]
            self.snap[c.client_id] = view
            if c.client_id in self.views:
                self.views[c.client_id].touch()

    def set_renderer(self, _):
        self.session.renderer = self.renderer.value
        self.rerender()

    def rerender(self):
        for v in self.views.values():
            v.touch()
            v.moved = 0.

    @guarded
    def load(self):
        S = self.session
        scene = self.scene_name.value
        model = self.ckpt_path.value.strip() or None
        if model is None and self.ckpt.value != 'paper model':
            model = self.registry.get(self.ckpt.value, self.ckpt.value)
        self.status(f'loading {scene} ...')
        S.load(scene=scene, model=model)
        S.exposure = None
        self.camera_view = S.pack.test[0] if S.pack.test else 0
        self.reset_panels()
        self.scene_objects()
        self.snap_to(self.camera_view)
        self.status(f'loaded {scene}')

    def reset_panels(self):
        """Options that depend on the model or scene."""
        S = self.session
        self.view_name.options = self.view_options()
        self.endmember.options = self.endmember_options()
        self.q_em.options = self.endmember_options()
        self.q_class.options = self.class_options()
        self.em_class.options = self.class_options()
        self.q_lib.options = self.library_options()
        wl = S.world.wavelengths
        self.wavelength.min, self.wavelength.max = float(wl[0]), float(wl[-1])
        for k, box in enumerate(self.em_boxes):
            box.visible = k < S.model.num_materials
            if k < S.model.num_materials:
                box.label = f'e{k} {S.mapping.name(k)}'
                box.value = True
        if S.model.num_materials > len(self.em_boxes):
            self.status('more endmembers than the panel was built for: restart the viewer to toggle all of them')
        self.update_queries()
        self.refresh_panels()

    def refresh_panels(self):
        S = self.session
        self.info.content = S.info()
        self.em_plot.image = S.endmember_image()
        self.em_table.content = S.endmember_table()
        self.show_legend()

    def show_legend(self):
        items = self.session.legend(self.mode.value)
        self.legend.content = ''.join(f'<span style="display:inline-block;margin:1px 6px 1px 0;font-size:11px">'
                                      f'<span style="display:inline-block;width:10px;height:10px;background:rgb({c[0]},{c[1]},{c[2]});'
                                      f'margin-right:3px"></span>{name}</span>' for c, name in items)

    # ------------------------------------------------------------ display
    def display(self, frame):
        S = self.session
        image = S.image(frame, self.mode.value, wavelength=self.wavelength.value, false_colour=FALSE_COLOUR[self.fc_preset.value],
                        endmember=self.endmember_options().index(self.endmember.value) if self.endmember.value in self.endmember_options() else 0,
                        cmap=self.cmap.value, exposure=self.exposure.value)
        if self.q_overlay.value and S.active in S.queries:
            image = S.overlay(frame, image, S.queries[S.active], tuple(self.q_colour.value))
        return image

    def depth(self, frame):
        """Camera-axis depth of the frame for viser's depth compositing (3D points and frustums are occluded by the
        render); empty pixels far away."""
        d, ok = self.session.depth(frame)
        d = d.where(ok, torch.full_like(d, 1e4))
        return d.reshape(frame.camera.height, frame.camera.width).cpu().numpy()

    def render_current(self, camera=None, view=None):
        """Headless render of the current camera (or the given one) as shown in the GUI; returns (image, frame)."""
        camera = camera or self.camera
        frame = self.session.render(camera, self.camera_view if view is None else view)
        self.frame = frame
        return self.display(frame), frame

    # ------------------------------------------------------------ clicks, readout, hits
    def on_click(self, event):
        view = self.views.get(event.client.client_id)
        frame = view.frame if view is not None else self.frame
        if frame is None:
            return
        x = int(np.clip(event.screen_pos[0] * frame.camera.width, 0, frame.camera.width - 1))
        y = int(np.clip(event.screen_pos[1] * frame.camera.height, 0, frame.camera.height - 1))
        self.click(frame, x, y)

    @guarded
    def click(self, frame, x, y):
        S = self.session
        r = S.readout(frame, x, y)
        self.readout_img.image = S.readout_image(r)
        self.inspection = S.inspect(frame, x, y)
        self.hits_md.content = S.hits_table(self.inspection)
        self.hits_img.image = S.hits_image(self.inspection)
        depth = sum(h['weight'] * h['depth'] for h in self.inspection['hits']) / max(self.inspection['alpha'], 1e-8)
        cam = frame.camera
        ray = cam.c2w[:3, :3] @ np.array([(x + .5 - cam.cx) / cam.fx, (y + .5 - cam.cy) / cam.fy, 1.])
        self.server.scene.add_icosphere('/click', radius=.004 * max(depth, 1.), color=(255, 40, 40),
                                        position=cam.c2w[:3, 3] + ray * depth)
        if self.click_action.value == 'query by click' or self.qtype.value.startswith('click') and self.armed:
            self.armed = False
            handling = {'auto (whitening)': 'auto', 'cone {hR, h, h/R}': 'cone', 'none (spectral angle)': 'none'}.get(self.q_handling.value, 'auto')
            q = S.add_query('click', x=x, y=y, frame=frame, handling=handling)
            self.after_query(q)
        self.status(f'pixel ({x}, {y}): {len(self.inspection["hits"])} hits, opacity {r["alpha"]:.2f}')

    armed = False

    @guarded
    def save_inspection(self):
        if self.inspection is None:
            raise ValueError('click a pixel first')
        i = self.inspection
        path = Path(self.x_dir.value) / f"hits_{i['view']}_{i['x']}_{i['y']}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(i, indent=1))
        self.status(f'saved `{path}`')
        return path

    # ------------------------------------------------------------ queries
    @guarded
    def run_query(self):
        S = self.session
        t = self.qtype.value
        self.status('computing per-Gaussian attributes over the TRAIN rays ...' if S.attributes is None else f'query: {t}')
        S.ensure_attributes(progress=lambda a, b: setattr(self.status_md, 'content', f'per-Gaussian attributes: view {a} / {b}'))
        lib = {'auto (illumination cone)': 'auto', 'none (spectral angle)': 'none', 'sunlit radiance': 'sun'}.get(self.q_handling.value, 'auto')
        if t == 'material name':
            q = S.add_query('name', class_name=self.q_class.value)
        elif t == 'endmember':
            q = S.add_query('endmember', k=self.endmember_options().index(self.q_em.value))
        elif t == 'library spectrum':
            q = S.add_query('library', index=self.library_options().index(self.q_lib.value), handling=lib)
        elif t == 'uploaded spectrum (CSV)':
            if self.upload is None:
                raise ValueError('upload a CSV spectrum first')
            q = S.add_query('spectrum', rho=self.upload[1], name=self.upload[0], handling=lib)
        elif t == 'click (pick a pixel)':
            self.armed = True
            self.status('armed: click a pixel in the view')
            return None
        elif t == 'band index':
            q = S.add_query('index', expression=self.q_index.value, rule=self.q_rule.value)
        elif t == 'Boolean combination':
            q = S.add_query('combine', expression=self.q_expr.value)
        else:
            q = S.add_query('instances', base=self.q_base.value, eps=float(self.q_eps.value), min_size=int(self.q_min_size.value),
                            min_height=float(self.q_min_height.value),
                            spectral=None if self.q_spectral.value == '-' else self.q_spectral.value)
        self.after_query(q)
        return q

    @guarded
    def on_upload(self, _):
        f = self.q_upload.value
        self.load_csv(f.name, f.content)

    def load_csv(self, name, content):
        rho = self.session.world.parse_csv(content)
        self.upload = (name, rho)
        self.q_upload_md.content = f'`{name}`: {len(rho)} bands, max {rho.max():.3g}'
        return rho

    def after_query(self, q):
        self.update_queries()
        self.activate(q.label)
        self.status(f'{q.label} ({q.kind}): {int(q.selected().sum())} Gaussians selected')

    def update_queries(self):
        S = self.session
        labels = list(S.queries) or ['-']
        self.q_active.options = labels
        self.q_base.options = labels
        self.q_spectral.options = ['-'] + [l for l, q in S.queries.items() if q.fingerprint is not None]
        self.q_list.content = '\n'.join(f'- **{l}** {q.kind}: {q.description} ({q.rule} {q.threshold:.3g})' for l, q in S.queries.items())

    def activate(self, label):
        S = self.session
        if label not in S.queries:
            S.active = None
            self.show_selection()
            return
        S.active = label
        q = S.queries[label]
        if self.q_active.value != label:
            self.q_active.value = label
        self._setting = True
        if q.kind == 'instances':                      # the slider sets the DBSCAN radius
            self.q_threshold.label = 'DBSCAN radius'
            self.q_threshold.min, self.q_threshold.max, self.q_threshold.step = .05, 5., .05
            self.q_threshold.value = q.params['eps']
        else:
            self.q_threshold.label = f'threshold ({"deg, " if q.log else ""}{q.rule})'
            step = .01 if q.log else max((q.hi - q.lo) / 500, 1e-4)
            self.q_threshold.min, self.q_threshold.max, self.q_threshold.step = float(q.lo), float(q.hi), float(step)
            self.q_threshold.value = float(np.clip(q.threshold, q.lo, q.hi))
        self._setting = False
        self.show_selection()

    _setting = False

    @guarded
    def set_threshold(self, value=None):
        S = self.session
        if self._setting or S.active not in S.queries:
            return
        q = S.queries[S.active]
        value = self.q_threshold.value if value is None else value
        if q.kind == 'instances':
            S.rerun_instances(q.label, eps=float(value))
        else:
            q.threshold = float(value)
        self.update_queries()
        self.show_selection()

    def reset_threshold(self):
        S = self.session
        if S.active in S.queries and S.queries[S.active].kind != 'instances':
            q = S.queries[S.active]
            q.threshold = q.default
            self.activate(q.label)

    def delete_query(self):
        S = self.session
        if S.active in S.queries:
            del S.queries[S.active]
            S.active = next(reversed(S.queries), None) if S.queries else None
            self.update_queries()
            self.activate(S.active)

    def show_selection(self):
        S = self.session
        q = S.queries.get(S.active) if S.active else None
        if q is None:
            self.q_md.content = ''
            self.server.scene.add_point_cloud('/selection', np.zeros((0, 3)), (255, 0, 255), visible=False)
        else:
            self.q_md.content = q.summary(S.attributes['weight'])
            self.q_hist.image = S.histogram_image(q)
            points, colours = S.selected_points(q, tuple(self.q_colour.value))
            self.server.scene.add_point_cloud('/selection', points, colours, point_size=self.q_point_size.value,
                                              point_shape='circle', visible=bool(self.q_points.value))
        self.rerender()

    # ------------------------------------------------------------ endmembers
    def toggle_endmember(self, k):
        S = self.session
        S.enabled[k] = bool(self.em_boxes[k].value)
        self.em_plot.image = S.endmember_image()
        self.em_table.content = S.endmember_table()
        self.show_legend()
        self.rerender()

    def endmembers_only(self, class_name):
        S = self.session
        for k in range(S.model.num_materials):
            self.em_boxes[k].value = class_name is None or S.mapping.name(k) == class_name

    # ------------------------------------------------------------ jobs
    def job(self, kind):
        live = [j for j in self.jobs if (j.stage == 'eval') == (kind == 'eval')]
        return live[-1] if live else None

    @guarded
    def start_training(self):
        S = self.session
        stage = self.t_stage.value.split(':')[0]
        running = self.job('train')
        if running is not None and running.running:
            raise RuntimeError('a training job is running; stop it first')
        out = Path(self.t_dir.value).expanduser()
        output = out / ({'anchor': 'anchor', 'source': 'source', 'prepare': f'clusters_K{int(self.t_K.value) or "scene"}',
                         'train': 'decoder', 'run': ''}[stage])
        inputs = {}
        if self.t_inputs.value:                         # chain the stages of this output directory
            ply = out / 'anchor/point_cloud/iteration_30000/point_cloud.ply'
            plys = sorted((out / 'anchor/point_cloud').glob('iteration_*/point_cloud.ply'))
            inputs['anchor_ply'] = str(ply if ply.exists() else plys[-1]) if (ply.exists() or plys) else None
            ckpts = sorted((out / 'source/ckpts').glob('ckpt_*_rank0.pt'))
            inputs['source'] = str(ckpts[-1]) if ckpts else None
            clusters = sorted(out.glob('clusters_K*'))
            inputs['clusters'] = str(clusters[-1]) if clusters else None
        argv = J.command(stage, S.scene, output, steps=int(self.t_steps.value) or None, materials=int(self.t_K.value) or None,
                         live_every=int(self.t_live.value) if stage in ('train', 'run') else 0, extra=self.t_extra.value,
                         inputs=inputs)
        total = int(self.t_steps.value) or {'anchor': 30000, 'source': 5000}.get(stage)
        job = J.Job(stage, argv, out if stage == 'run' else output, self.jobs_dir / 'logs', total)
        self.jobs.append(job)
        self.followed = None
        self.status(f'started `{job.command}`')
        return job

    @guarded
    def start_eval(self):
        S = self.session
        running = self.job('eval')
        if running is not None and running.running:
            raise RuntimeError('an evaluation is running')
        output = Path(self.e_dir.value).expanduser()
        argv = J.command('eval', S.scene, output, model=S.model_path, fast=self.e_fast.value, extra=self.e_extra.value)
        job = J.Job('eval', argv, output, self.jobs_dir / 'logs', len(S.pack.test))
        self.jobs.append(job)
        self.status(f'started `{job.command}`')
        return job

    def stop_job(self, kind):
        job = self.job(kind)
        if job is not None:
            job.stop()
            self.status(f'stopped {job.stage}')

    @guarded
    def show_eval(self, directory):
        self.e_table.content = J.metrics_table(directory, self.session.paths.get('label_space'))
        if self.session.set_mapping(directory):
            self.reset_panels()
            self.rerender()

    def watch_jobs(self):
        while True:
            time.sleep(1.)
            try:
                self.poll_jobs()
            except Exception:                           # noqa: BLE001
                traceback.print_exc()

    def poll_jobs(self):
        train, ev = self.job('train'), self.job('eval')
        if train is not None:
            self.t_progress.value = 100 * train.fraction()
            self.t_md.content = train.summary()
            self.t_log.content = '```\n' + '\n'.join(list(train.lines)[-14:]) + '\n```'
            rows = train.live_trace() if train.stage in ('train', 'run') else []
            if rows and len(rows) != getattr(self, '_loss_rows', 0):
                self._loss_rows = len(rows)
                self.t_loss.image = loss_image(rows)
            latest = train.latest_model()
            if self.t_follow.value and latest is not None:
                stamp = (latest, latest.stat().st_mtime)
                if stamp != self.followed:
                    self.followed = stamp
                    self.follow(latest)
        if ev is not None:
            self.e_progress.value = 100 * (1. if not ev.running else ev.fraction())
            self.e_md.content = ev.summary() + '\n```\n' + '\n'.join(list(ev.lines)[-8:]) + '\n```'
            if not ev.running and getattr(self, '_shown_eval', None) is not ev:
                self._shown_eval = ev
                if ev.returncode == 0:
                    self.show_eval(ev.output)

    @guarded
    def follow(self, path):
        """Show the newest checkpoint of the training job (same scene; a class map only from an evaluation of that model)."""
        S = self.session
        S.load_model(path)
        self.reset_panels()
        self.rerender()
        self.status(f'preview: `{path}`')

    # ------------------------------------------------------------ exports
    def export_frame(self):
        """The frame to export: the dataset camera at its own resolution when snapped (and enabled), else the view."""
        S = self.session
        if self.camera_view >= 0 and self.x_native.value:
            return S.render(S.camera(self.camera_view), self.camera_view)
        return S.render(self.camera, self.camera_view)

    def export_name(self):
        name = self.session.pack.names[self.camera_view] if self.camera_view >= 0 else f'free_{time.strftime("%H%M%S")}'
        return f'{self.session.scene or "scene"}_{name}'

    @guarded
    def export_png(self, path=None):
        frame = self.export_frame()
        path = Path(path or Path(self.x_dir.value) / f'{self.export_name()}_{self.mode.value.replace(" ", "_")}.png')
        save_png(path, self.display(frame))
        self.x_md.content = f'saved `{path}`'
        return path

    @guarded
    def export_cube(self, path=None):
        frame = self.export_frame()
        paths = self.session.export_cube(frame, path or Path(self.x_dir.value) / f'{self.export_name()}_cube', self.x_radiance.value)
        self.x_md.content = 'saved ' + ', '.join(f'`{p}`' for p in paths)
        return paths

    @guarded
    def export_ply(self, path=None):
        S = self.session
        if S.active not in S.queries:
            raise ValueError('run a query first')
        path, n = S.export_ply(path or Path(self.x_dir.value) / f'{S.scene or "scene"}_{S.active}.ply')
        self.x_md.content = f'saved {n} Gaussians to `{path}`'
        return path

    def add_keyframe(self, camera=None):
        self.keyframes.append(camera or self.camera)
        pts = np.stack([k.c2w[:3, 3] for k in self.keyframes])
        if len(pts) > 1:
            self.server.scene.add_spline_catmull_rom('/path', pts, color=(255, 200, 0), thickness=3., thickness_units='screen')
        self.x_md.content = f'{len(self.keyframes)} keyframes'

    def clear_keyframes(self):
        self.keyframes = []
        self.server.scene.add_spline_catmull_rom('/path', np.zeros((2, 3)), visible=False)
        self.x_md.content = 'keyframes cleared'

    @guarded
    def export_video(self, path=None):
        cams = [c.resized(self.x_side.value) for c in interpolate_path(self.keyframes, int(self.x_frames.value))]
        path = Path(path or Path(self.x_dir.value) / f'{self.session.scene or "scene"}_path_{self.mode.value.replace(" ", "_")}.mp4')
        self.session.render_video(cams, path, int(self.x_fps.value), self.display,
                                  progress=lambda a, b: setattr(self.x_progress, 'value', 100 * a / b))
        self.x_md.content = f'saved `{path}` ({len(cams)} frames)'
        return path


def loss_image(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .session import figure_image
    steps = [r['step'] for r in rows]
    fig, ax = plt.subplots(figsize=(4.4, 2.2))
    ax.plot(steps, [r['loss'] for r in rows], label='loss', lw=1.2)
    ax.plot(steps, [r['l1'] for r in rows], label='L1', lw=1)
    ax.set_yscale('log')
    ax.set_xlabel('update')
    ax2 = ax.twinx()
    ax2.plot(steps, [r['sam_rad'] * 57.29578 for r in rows], color='C3', lw=1, label='SAM (deg)')
    ax2.set_ylabel('SAM (deg)', fontsize=7)
    ax.legend(fontsize=6, loc='upper right')
    return figure_image(fig)
