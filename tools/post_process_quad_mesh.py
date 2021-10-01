import os
import numpy as np
from models.layers.mesh_prepare import remove_non_manifolds, build_gemm
import ntpath
from tqdm import tqdm


def fill_from_file(mesh, file):
    mesh.filename = ntpath.split(file)[1]
    mesh.fullfilename = file
    vs, faces = [], []
    f = open(file)
    for line in f:
        line = line.strip()
        splitted_line = line.split()
        if not splitted_line:
            continue
        elif splitted_line[0] == 'v':
            vs.append([float(v) for v in splitted_line[1:4]])
        elif splitted_line[0] == 'f':
            face_vertex_ids = [int(c.split('/')[0]) for c in splitted_line[1:]]
            assert (len(face_vertex_ids) == 4)
            face_vertex_ids = [(ind - 1) if (ind >= 0) else (len(vs) + ind)
                               for ind in face_vertex_ids]
            if len(face_vertex_ids) is not 4:
                print('not pure quad mesh!!!')
                continue

            faces.append(face_vertex_ids)
    f.close()
    vs = np.asarray(vs)
    faces = np.asarray(faces, dtype=int)
    assert np.logical_and(faces >= 0, faces < len(vs)).all()
    return vs, faces


dataset_path = r'E:\Omri\FinalProject\QuadMesh\meshMNIST\Manual'
objs_path = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path)
             for f in filenames if os.path.splitext(f)[1] == '.obj']

max_edges = 0
min_edges = 1000000
pbar = tqdm(total=len(objs_path))
for obj in objs_path:
    class MeshPrep:
        def __getitem__(self, item):
            return eval('self.' + item)

    mesh_data = MeshPrep()
    mesh_data.vs = mesh_data.edges = None
    mesh_data.gemm_edges = mesh_data.sides = None
    mesh_data.edges_count = None
    mesh_data.ve = None
    mesh_data.v_mask = None
    mesh_data.filename = 'unknown'
    mesh_data.edge_lengths = None
    mesh_data.edge_areas = []

    mesh_data.vs, faces = fill_from_file(mesh_data, obj)
    faces_before_remove_manifolds = len(faces)
    faces, face_areas = remove_non_manifolds(mesh_data, faces)
    if len(face_areas) == 0:
        pbar.update(1)
        continue

    if len(faces) < faces_before_remove_manifolds:
        print('obj %s have non manifolds!!!' % obj)

    build_gemm(mesh_data, faces, face_areas)
    if mesh_data.edges_count > max_edges:
        max_edges = mesh_data.edges_count
        print('max edges count: %d %s\n' % (max_edges, obj))
    if mesh_data.edges_count < min_edges:
        min_edges = mesh_data.edges_count
        print('min edges count: %d, %s\n' % (min_edges, obj))

    for e in mesh_data.edges:
        if -1 in mesh_data.gemm_edges[e]:
            print('obj %s boundary edges!!!' % obj)
            break

    pbar.update(1)
pbar.close()
