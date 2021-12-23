import bpy
import bmesh


def init_mesh():
    return bmesh.new()


def save_mesh(bm, obj):
    bm.to_mesh(obj)
    bm.free()  # free and prevent further access


def get_selected():
    return bpy.context.object.data


# copied from emfields project
def draw_curve(x_sol, to_3d_pos, bm=None, closed=True, return_geom=False, remove_doubles=True):
    if bm is None:
        bm = bmesh.new()
    verts = [bm.verts.new(to_3d_pos(p)) for p in x_sol]
    edges = []
    viter = zip(verts, verts[1:] + [verts[0]]) if closed else zip(verts, verts[1:])
    for v1, v2 in viter:
        edges += bmesh.ops.contextual_create(bm, geom=[v1, v2])['edges']

    if remove_doubles:
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-6)
    if return_geom:
        return {'verts': verts, 'edges': edges}
    else:
        return bm
