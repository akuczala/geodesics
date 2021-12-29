from functools import singledispatchmethod

import bpy
import bmesh
import numpy as np

from blenderhelper.draw_data import CurveData, VectorData, DrawDataList, FrameData, ConeData


class BlenderHelper:
    def __init__(self, bm=None, to_3d_pos=None):
        self.bm = self.init_mesh() if bm is None else bm
        self.to_3d_pos = to_3d_pos
        #self.tangent_to_3d = tangent_to_3d
        self.draw_curve_kwargs = {}

    @staticmethod
    def init_mesh():
        return bmesh.new()

    @staticmethod
    def get_selected():
        return bpy.context.object.data

    def save_mesh(self, obj):
        self.bm.to_mesh(obj)
        self.bm.free()  # free and prevent further access

    # adapted from emfields project
    def draw_curve(self, points, closed=False, return_geom=False, remove_doubles=True):
        verts = [self.bm.verts.new(self.to_3d_pos(p)) for p in points]
        edges = []
        viter = zip(verts, verts[1:] + [verts[0]]) if closed else zip(verts, verts[1:])
        for v1, v2 in viter:
            edges += bmesh.ops.contextual_create(self.bm, geom=[v1, v2])['edges']

        if remove_doubles:
            bmesh.ops.remove_doubles(self.bm, verts=self.bm.verts, dist=1e-6)
        if return_geom:
            return {'verts': verts, 'edges': edges}
        else:
            return self.bm

    def draw_vector(self, pos, vector):
        verts = [self.bm.verts.new(self.to_3d_pos(v)) for v in [pos, np.array(pos) + np.array(vector)]]
        bmesh.ops.contextual_create(self.bm, geom=verts)
        return self.bm

    def draw_frame(self, pos, vecs):
        pos_vert = self.bm.verts.new(self.to_3d_pos(pos))
        frame_verts = [self.bm.verts.new(self.to_3d_pos(np.array(pos) + np.array(v))) for v in vecs]
        for vert in frame_verts:
            bmesh.ops.contextual_create(self.bm, geom=[pos_vert, vert])
        return self.bm

    def draw_vector_field(self, positions, vectors):
        for pos, v in zip(positions, vectors):
            self.draw_vector(pos, v)

    def draw_cone(self, pos, vecs):
        apex_vert = self.bm.verts.new(self.to_3d_pos(pos))
        rim_verts = [self.bm.verts.new(self.to_3d_pos(np.array(pos) + np.array(v))) for v in vecs]
        for v1, v2 in zip(rim_verts, rim_verts[1:] + rim_verts[:1]):
            bmesh.ops.contextual_create(self.bm, geom=[apex_vert, v1, v2])

    @singledispatchmethod
    def draw_data(self, data):
        raise TypeError(f"type {type(data)} not supported")

    @draw_data.register
    def _(self, data: CurveData):
        self.draw_curve(data.points, **self.draw_curve_kwargs)

    @draw_data.register
    def _(self, data: VectorData):
        self.draw_vector(pos=data.point, vector=data.vector)

    @draw_data.register
    def _(self, data: FrameData):
        self.draw_frame(data.point, data.vecs)

    @draw_data.register
    def _(self, data: DrawDataList):
        for d in data.data_list:
            self.draw_data(d)

    @draw_data.register
    def _(self, data: ConeData):
        self.draw_cone(data.apex, data.vecs)