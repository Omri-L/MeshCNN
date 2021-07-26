import torch
import torch.nn as nn
from threading import Thread
from models.layers.mesh_union import MeshUnion
import numpy as np
from heapq import heappop, heapify
import random
from copy import deepcopy


class MeshPool(nn.Module):

    def __init__(self, target, multi_thread=False):
        super(MeshPool, self).__init__()
        self.__out_target = target
        self.__multi_thread = multi_thread
        self.__fe = None
        self.__updated_fe = []
        self.__meshes = None
        self.__merge_edges = [-1, -1]

    def __call__(self, fe, meshes):
        return self.forward(fe, meshes)

    def forward(self, fe, meshes):
        self.__updated_fe = [[] for _ in range(len(meshes))]
        pool_threads = []
        self.__fe = fe
        self.__meshes = meshes
        # iterate over batch
        for mesh_index in range(len(meshes)):
            if self.__multi_thread:
                pool_threads.append(
                    Thread(target=self.__pool_main, args=(mesh_index,)))
                pool_threads[-1].start()
            else:
                self.__pool_main(mesh_index)
        if self.__multi_thread:
            for mesh_index in range(len(meshes)):
                pool_threads[mesh_index].join()
        a = self.__updated_fe
        out_features = torch.cat(self.__updated_fe).view(len(meshes), -1,
                                                         self.__out_target)
        return out_features

    def __pool_main(self, mesh_index):
        mesh = self.__meshes[mesh_index]
        queue = self.__build_queue(self.__fe[mesh_index, :, :mesh.edges_count],
                                   mesh.edges_count)
        # recycle = []
        # last_queue_len = len(queue)
        last_count = mesh.edges_count + 1
        mask = np.ones(mesh.edges_count, dtype=np.bool)
        edge_groups = MeshUnion(mesh.edges_count, self.__fe.device)
        fe = self.__fe[mesh_index, :, :mesh.edges_count].clone()

        ids = None
        if self.__out_target == 430:
            ids = [137, 248, 168, 105, 152, 75, 173, 187, 123, 154, 306, 242, 218, 207, 132, 121, 275, 115, 150, 247, 99, 181, 303, 16, 246, 208 ]
        elif self.__out_target == 380:
            ids = [147, 123, 205, 401, 75, 405, 19, 201, 407, 44, 175, 403, 207, 226, 124, 253, 16, 400, 102, 101, 151, 250, 413, 76, 411, 186, 233]
        # elif self.__out_target == 300:
        #     ids = [379, 215, 148, 124, 104, 235, 3, 105, 100, 102, 163, 43, 125, 166, 22, 81, 259, 19, 139, 147, 57, 130, 109]
        # elif self.__out_target == 250:
        #     ids = [299, 159, 193, 65, 146, 173, 160, 86, 138, 176, 148, 0, 130, 214, 36, 118, 3, 134, 69, 106, 122, 22, 47, 4, 7, 110, 34, 191, 240, 141, 39, 158, 161, 41, 19, 129, 25, 90, 43, 117, 42, 211, 35, 116, 271, 31, 137, 2, 144, 16, 142, 109, 125, 270, 147, 143]

        # ids = [234, 203, 44, 168, 43, 137, 69, 75, 160]
        # ids = [16953, 13019, 18222, 9867, 541, 17293, 12942, 13217, 14538, 12323, 15846, 11915, 4741, 13079, 15130, 17372, 4831, 14193, 13802, 16145, 10696, 3023, 5939, 2307, 17054, 17451, 7235, 12619, 16337, 16474, 7199, 4663, 7445, 12322, 9974, 8431, 4618, 8933, 12393, 17693, 3509, 6739, 2101, 4681, 7233, 7810, 15245, 5018, 14288, 9117, 7214, 16966, 17040, 16473, 13445, 5602, 4591, 8430, 15431, 16172, 15433, 13779, 10531, 1924, 14903, 5774, 5234, 6835, 15534, 3035, 16198, 13388, 6803, 8426, 12877, 6667, 8844, 10161, 16653, 16482, 6688, 16790, 9463, 4240, 11770, 16120, 18494, 2149, 16463, 2440, 3377, 7793, 13382, 10518, 13086, 4680, 10708, 1798, 5012, 7469, 10692, 17921, 4880, 4586, 13096, 4817, 8932, 7685, 10919, 4938, 13862, 18008, 17144, 7367, 15278, 16343, 6884, 3244, 4628, 14843, 7225, 17412, 16585, 7092, 16485, 9421, 2814, 10212, 5414, 12615, 13193, 10203, 2343, 13848, 17411, 9977, 3515, 16436, 1926, 12805, 7537, 10807, 14197, 14177, 6877, 10208, 6108, 9000, 13268, 14191, 13413, 8544, 7473, 10688, 12776, 6805, 14625, 5768, 9601, 11856, 6573, 8423, 15314, 15112, 13093, 5235, 2811, 14285, 15689, 16429, 12657, 7458, 6245, 13381, 9446, 2145, 2899, 2072, 9408, 17418, 4923, 2635, 13417, 3929, 9724, 2869, 7765, 16580, 13800, 11005, 3736, 15989, 12713, 13294, 18228, 4620, 6570, 2148, 18277, 14365, 14301, 15640, 2466, 12822, 15003, 4823, 17687, 11859, 13472, 5169, 8843, 14534, 14214, 5118, 16168, 4405, 14588, 6284, 13372, 14147, 17253, 2868, 16331, 7800, 15998, 16466, 8928, 6888, 6348, 9407, 3413, 5763, 2637, 14262, 10174, 15357, 12621, 8422, 13325, 18258, 16399, 6238, 7252, 15971, 12188, 5804, 16113, 17297, 16166, 3931, 9050, 4756, 17408, 8414, 14537, 11911, 2470, 4643, 15509, 12816, 11228, 6742, 17784, 12618, 7744, 2463, 16199, 2450, 9602, 10699]
        count = -1
        while mesh.edges_count > self.__out_target:
            if len(queue) == 0:
                queue = self.__build_queue(self.__fe[mesh_index, :, :mesh.edges_count], mesh.edges_count)

            value, edge_id = heappop(queue)
            count += 1
            if ids is not None:
                edge_id = ids[count]
            edge_id = int(edge_id)
            print('pool edge_id %d' % edge_id)
            if mask[edge_id]:
                _, fe = self.__pool_edge(mesh, edge_id, fe, mask, edge_groups)
        mesh.clean(mask, edge_groups)
        fe = edge_groups.rebuild_features(self.__fe[mesh_index], fe, mask,
                                          self.__out_target)  # TODO use features build here?
        self.__updated_fe[mesh_index] = fe
        print('finish pooling')

    def __pool_edge(self, mesh, edge_id, fe, mask, edge_groups):

        if self.has_boundaries(mesh, edge_id):
            return False, fe

        if not np.all(mesh.gemm_edges[mesh.gemm_edges[edge_id],
                                      mesh.sides[edge_id]] == edge_id):
            new_sides = [-1, -1, -1, -1, -1, -1]
            for i_en, en in enumerate(mesh.gemm_edges[edge_id]):
                if en == -1:
                    continue
                side = np.where(mesh.gemm_edges[en] == edge_id)[0]
                if len(side) > 0:
                    new_sides[i_en] = side[0]

            mesh.sides[edge_id] = new_sides

        if self.__clean_side(mesh, edge_id, mask, edge_groups, 0) \
                and self.__clean_side(mesh, edge_id, mask, edge_groups, 3) \
                and self.__is_one_ring_valid(mesh, edge_id):

            fe = self.mesh_decimation(mesh, fe, edge_id, mask, edge_groups)
            # mesh.merge_vertices(edge_id)  # TODO should update this method
            return True, fe
        else:
            return False, fe

    @staticmethod
    def get_all_vertices_of_edges_connected_to_vertex(mesh, vertex_u):
        """
        Get all the vertices of edges which are connected to vertex u,
        exclude u itself. Another output is the edges themselves.
        """
        v_e_u = []
        e_u = mesh.ve[vertex_u].copy()  # edges connected to vertex u

        for e in e_u:
            v1, v2 = mesh.edges[e]
            if v1 == vertex_u:
                v_e_u.append(v2)
            else:
                v_e_u.append(v1)

        return v_e_u, e_u

    @staticmethod
    def get_vertices_connections_via_another_edge(mesh, v1, v2):
        """
        Find mutual connection (vertex) via another edges from vertices v1 and
        v2.
        """
        # get all edges connected to v1 or v2
        e_v1 = mesh.ve[v1]
        e_v2 = mesh.ve[v2]
        # get all vertices of edges e_v1 or e_v2
        v_e_v1 = set(mesh.edges[e_v1].reshape(-1))
        v_e_v2 = set(mesh.edges[e_v2].reshape(-1))
        # get the vertex intersection
        inter_vertices = set.intersection(v_e_v1, v_e_v2)
        return inter_vertices

    @staticmethod
    def collapse_other_vertex_v(mesh, u, v, e_v, diag_vertices,
                                new_features_combination_dict, edge_groups,
                                mask):
        """

        Args:
            mesh:
            v: vertex v
            e_v: edges connected to vertex v

        Returns:

        """
        e_to_collapse = []  # edges we should remove
        e_to_reconnect_with_u = []  # edges we should re-connect with vertex u

        for e in e_v:
            u_e, v_e = mesh.edges[e, :]
            if u_e == v:  # make sure u_e is the other vertex
                u_e = v_e
                v_e = v
            # if it is an edge of the closet hood
            if u_e in diag_vertices or v_e in diag_vertices:
                e_to_collapse.append(e)

                for key in new_features_combination_dict.keys():
                    if u_e in mesh.edges[key]:
                        edge_to_add_feature = key
                        new_features_combination_dict[key].append(e)
                        break
                # collapse
                MeshPool.__remove_group(mesh, edge_groups, e)  # TODO needed?
                mask[e] = False
                mesh.remove_edge(e)
                mesh.edges[e] = [-1, -1]  # TODO needed?
                mesh.gemm_edges[e] = [-1, -1, -1, -1, -1, -1]  # TODO needed?
                mesh.edges_count -= 1
            else:
                e_to_reconnect_with_u.append(e)
                mesh.ve[v].remove(e)
                mesh.ve[u].append(e)
                if mesh.edges[e, 0] == v:
                    mesh.edges[e, 0] = u
                else:
                    mesh.edges[e, 1] = u

        return e_to_collapse, e_to_reconnect_with_u

    @staticmethod
    def rotate_edges_and_connections(mesh, origin_vertex, edges_to_change,
                                     optional_vertices,
                                     optional_vertex_to_edges_dict):
        """
        For each edge goes from original vertex: change the original connection
        to another optional vertex connection in a way that all vertices have
        a new connection (it will make sure the rotation for all will be CCW or
        CW)
        """
        edges_to_change_cpy = edges_to_change.copy()  # make a copy
        optional_vertices_cpy = optional_vertices.copy()  # make a copy
        e = edges_to_change_cpy[0]  # first edge in list (init)
        new_features_combination_dict = dict()
        while len(edges_to_change_cpy) > 0:
            edges_to_change_cpy.remove(e)
            # check which connection is possible according to connectivity
            optional_new_v = set.intersection(set(optional_vertices_cpy), set(
                mesh.edges[mesh.gemm_edges[e]].reshape(-1)))
            # choose one vertex randomly
            random.seed(0)
            v_e_new = random.choice(list(optional_new_v))
            # remove it from options
            optional_vertices_cpy.remove(v_e_new)
            # print('before - edge %d vertices %d %d' % (e, mesh.edges[e, 0],
            #                                            mesh.edges[e, 1]))
            # print(
            #     'after - edge %d vertices %d %d' % (e, origin_vertex, v_e_new))

            v_e_orig = list(mesh.edges[e, :].copy())
            v_e_orig.remove(origin_vertex)

            # update new connection
            mesh.edges[e] = [origin_vertex, v_e_new]
            mesh.ve[v_e_new].append(e)
            mesh.ve[v_e_orig[0]].remove(e)
            new_features_combination_dict[e] = \
                optional_vertex_to_edges_dict[str(v_e_new)]

            # for next iteration - take the second edge option for connection
            # with v_e_new to make sure we go with one direction only for all
            # edges in e_u
            other_e_matched = optional_vertex_to_edges_dict[str(v_e_new)]. \
                copy()
            other_e_matched.remove(e)
            e = other_e_matched[0]

        # new_features_dict = dict()
        # for k in new_features_combination_dict.keys():
        #     combination = new_features_combination_dict[k]
        #     new_feature = torch.sum(fe[:, combination, :], axis=1)/len(combination)
        #     new_feature = new_feature.reshape(new_feature.shape[0], 1, -1)
        #     new_features_dict[k] = new_feature

        return mesh, new_features_combination_dict #, new_features_dict

    def rebuild_edge_hood(self, mesh, edge_id):
        v1, v2 = mesh.edges[edge_id]

        v_e_v1, e_v1 = self.get_all_vertices_of_edges_connected_to_vertex(mesh,
                                                                          v1)
        v_e_v2, e_v2 = self.get_all_vertices_of_edges_connected_to_vertex(mesh,
                                                                          v2)
    # def get_all_faces_connected_to_vertex(self, mesh, u):
    #     """
    #     Gets all the faces connected to vertex u
    #     """
    #     v_e_u, e_u = self.get_all_vertices_of_edges_connected_to_vertex(mesh,
    #                                                                     u)
    #     checked_vertex = []
    #     faces_vertices = []
    #     faces = []
    #     for i, v_i in enumerate(v_e_u):
    #         checked_vertex.append(i)
    #         for j, v_j in enumerate(v_e_u):
    #             if j in checked_vertex:
    #                 continue  # do not check vertex twice
    #             face = []
    #             face.append(v_i)
    #             # find mutual vertices via another edges start from v_i and v_j
    #             connection_vertices = \
    #                 self.get_vertices_connections_via_another_edge(mesh, v_i,
    #                                                                v_j)
    #             if len(connection_vertices) == 1:
    #                 continue
    #             face.append(connection_vertices.pop())
    #             face.append(v_j)
    #             face.append(connection_vertices.pop())
    #             faces_vertices.append(face)
    #             face_edges = []
    #             for i in range(len(face)):
    #                 fe = self.get_edge_from_two_vertices(mesh, face[i],
    #                                                      face[(i + 1) % 4])
    #                 assert (fe is not None)
    #                 face_edges.append(fe)
    #             faces.append(set(face_edges))
    #
    #     return set(faces), face_edges, set(faces_vertices)

    def get_all_faces_connected_to_vertex(self, mesh, u):
        """
        Gets all the faces connected to vertex u
        """
        v_e_u, e_u = self.get_all_vertices_of_edges_connected_to_vertex(mesh,
                                                                        u)
        checked_vertex = []
        faces_vertices = []
        faces = []
        for i, v_i in enumerate(v_e_u):
            checked_vertex.append(i)
            for j, v_j in enumerate(v_e_u):
                if j in checked_vertex:
                    continue  # do not check vertex twice
                face_vertices = []
                face_vertices.append(v_i)
                # find mutual vertices via another edges start from v_i and v_j
                connection_vertices = \
                    self.get_vertices_connections_via_another_edge(mesh, v_i,
                                                                   v_j)
                # if len(connection_vertices) == 1:
                if len(connection_vertices) < 2:
                    continue
                face_vertices.append(connection_vertices.pop())
                face_vertices.append(v_j)
                face_vertices.append(connection_vertices.pop())
                faces_vertices.append(face_vertices)
                face_edges = []
                for i in range(len(face_vertices)):
                    fe = self.get_edge_from_two_vertices(mesh, face_vertices[i],
                                                         face_vertices[(i + 1) % 4])
                    assert (fe is not None)
                    face_edges.append(fe)
                faces.append(face_edges)

        return faces, faces_vertices

    def build_edges_hood_and_sides(self, mesh, faces_vertices):
        edge_nb = []
        sides = []
        edge2key = dict()
        abs_edges = dict()
        edges = []
        edges_count = 0
        nb_count = []
        for face_id, face in enumerate(faces_vertices):
            face = list(face)
            faces_edges = []
            for i in range(4):
                cur_edge = (face[i], face[(i + 1) % 4])
                faces_edges.append(cur_edge)
            for idx, edge in enumerate(faces_edges):
                edge = tuple(sorted(list(edge)))
                faces_edges[idx] = edge
                if edge not in edge2key:
                    edge2key[edge] = edges_count
                    abs_edges[edges_count] = self.get_edge_from_two_vertices(
                        mesh, edge[0], edge[1])
                    edges.append(list(edge))
                    edge_nb.append([-1, -1, -1, -1, -1, -1])
                    sides.append([-1, -1, -1, -1, -1, -1])
                    nb_count.append(0)
                    edges_count += 1
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                edge_nb[edge_key][nb_count[edge_key]] = abs_edges[edge2key[
                    faces_edges[(idx + 1) % 4]]]
                edge_nb[edge_key][nb_count[edge_key] + 1] = abs_edges[edge2key[
                    faces_edges[(idx + 2) % 4]]]
                edge_nb[edge_key][nb_count[edge_key] + 2] = abs_edges[edge2key[
                    faces_edges[(idx + 3) % 4]]]
                nb_count[edge_key] += 3
            for idx, edge in enumerate(faces_edges):
                edge_key = edge2key[edge]
                sides[edge_key][nb_count[edge_key] - 3] = nb_count[edge2key[
                    faces_edges[(idx + 1) % 4]]] - 1
                sides[edge_key][nb_count[edge_key] - 2] = nb_count[edge2key[
                    faces_edges[(idx + 2) % 4]]] - 2
                sides[edge_key][nb_count[edge_key] - 1] = nb_count[edge2key[
                    faces_edges[(idx + 3) % 4]]] - 3

        for idx, edges in enumerate(edge_nb):
            if mesh.edges[edges[2], 0] in mesh.edges[edges[3]]:
                continue
            elif mesh.edges[edges[2], 1] in mesh.edges[edges[3]]:
                continue
            else:
                edges[5], edges[3] = edges[3], edges[5]
                sides[idx][5], sides[idx][3] = sides[idx][3], sides[idx][5]

        # Put new data in mesh data structure
        for i in range(len(abs_edges)):
            abs_edge = abs_edges[i]
            old_gemm = mesh.gemm_edges[abs_edge]
            if -1 in edge_nb[i] and -1 not in old_gemm:
                new_gemm = edge_nb[i]
                old_pos = np.where(old_gemm == new_gemm[0])[0].flatten()

                if old_pos == 0:
                    new_gemm = old_gemm
                elif old_pos == 2:
                    new_gemm = [old_gemm[2], old_gemm[1], old_gemm[0],
                                old_gemm[5], old_gemm[4], old_gemm[3]]
                elif old_pos == 3:
                    new_gemm = [old_gemm[3], old_gemm[4], old_gemm[5],
                                old_gemm[0], old_gemm[1], old_gemm[2]]
                elif old_pos == 5:
                    new_gemm = old_gemm[::-1]
                else:
                    assert(False)

                mesh.gemm_edges[abs_edge] = new_gemm
                if mesh.edges[new_gemm[2], 0] in mesh.edges[new_gemm[3]]:
                    continue
                elif mesh.edges[new_gemm[2], 1] in mesh.edges[new_gemm[3]]:
                    continue
                else:
                    new_gemm[5], new_gemm[3] = new_gemm[3], new_gemm[5]
                    mesh.gemm_edges[abs_edge] = new_gemm

            else:
                mesh.gemm_edges[abs_edge] = edge_nb[i]
                mesh.sides[abs_edge] = sides[i]

        # I. fix sides
        for i in range(len(abs_edges)):
            abs_edge = abs_edges[i]
            new_sides = [-1, -1, -1, -1, -1, -1]
            for i_en, en in enumerate(mesh.gemm_edges[abs_edge]):
                if en == -1:
                    continue
                side = np.where(mesh.gemm_edges[en] == abs_edge)[0]
                if len(side) > 0:
                    new_sides[i_en] = side[0]

            mesh.sides[abs_edge] = new_sides

        return

    @staticmethod
    def get_edge_from_two_vertices(mesh, v1, v2):
        e1 = mesh.ve[v1]
        e2 = mesh.ve[v2]
        edge = None
        for e in e1:
            if e in e2:
                edge = e
                break
        return edge

    @staticmethod
    def combine_edge_features(mesh, in_features, new_features_combination_dict, edge_groups):
        """
        Creates new edge features according to features combinations dictionary
        """
        updated_fe = in_features.clone()
        for k in new_features_combination_dict.keys():
            combination = new_features_combination_dict[k]
            assert (len(combination) > 0)
            new_feature = torch.sum(in_features[:, combination, :],
                                    axis=1) / len(combination)
            for c in combination:
                if c is not k:
                    MeshPool.__union_groups(mesh, edge_groups, c, k)

            # new_feature = new_feature.reshape(new_feature.shape[0], 1, -1)
            updated_fe[:, int(k), :] = new_feature

        return updated_fe


    def find_double_faces(self, mesh, face, faces, face_vertices, faces_vertices):
        all_vn = []
        removed_faces = []
        removed_faces_vertices = []
        for v in face_vertices:
            vn, _ = self.get_all_vertices_of_edges_connected_to_vertex(mesh, v)
            all_vn = all_vn + vn

        outer_vertices = set(all_vn) - set(face_vertices)
        if len(outer_vertices) == 4:
            faces_copy = faces.copy()
            faces_vertices_copy = faces_vertices.copy()
            for fi, f in enumerate(faces_copy):
                if f == face:
                    continue
                else:
                    is_outer_face = len(outer_vertices.difference(set(faces_vertices_copy[fi]))) == 0
                    if is_outer_face:
                        removed_faces.append(f)
                        removed_faces_vertices.append(faces_vertices_copy[fi])
                        faces.remove(f)
                        faces_vertices.remove(faces_vertices_copy[fi])
                        # print(f)
                        # print('double face was found!')

        return removed_faces, removed_faces_vertices

    @staticmethod # TODO move from here
    def find_doublets(mesh):
        doublet_pair_edges = []
        doublet_vertices = np.where(np.array([len(mesh.ve[j]) for j in range(len(mesh.ve))]) == 2)[0]
        if len(doublet_vertices) > 0:
            doublet_pair_edges = [mesh.ve[v].copy() for v in doublet_vertices]
        return doublet_vertices, doublet_pair_edges

    def clear_doublets(self, mesh, mask, edge_groups):
        doublet_vertices, doublet_pairs_edges = self.find_doublets(mesh)
        if len(doublet_vertices) == 0:
            return [], [], []

        pairs_edges_vertices = np.array([mesh.edges[e].copy() for e in doublet_pairs_edges])

        for i, doublet_pair_edges in enumerate(doublet_pairs_edges):
            vertex = doublet_vertices[i]
            for e in doublet_pair_edges:
                u, v = mesh.edges[e]

                MeshPool.__remove_group(mesh, edge_groups, e)  # TODO needed?
                mask[e] = False
                mesh.remove_edge(e)
                mesh.edges[e] = [-1, -1]  # TODO needed?
                mesh.gemm_edges[e] = [-1, -1, -1, -1, -1, -1]  # TODO needed?
                mesh.edges_count -= 1

                # mesh.ve[u].remove(e)
                # mesh.ve[v].remove(e)
                # e_ns = mesh.gemm_edges[e].copy()
                # for j, e_n in enumerate(e_ns):
                #     side = mesh.sides[e, j]
                #     mesh.gemm_edges[e_n, side] = -1

                mesh.gemm_edges[e] = [-1, -1, -1, -1, -1, -1]

        out = self.clear_doublets(mesh, mask, edge_groups)
        if len(out) > 0:
            doublet_vertices = list(doublet_vertices) + list(out[0])
            pairs_edges_vertices = list(pairs_edges_vertices) + list(out[1])
            doublet_pairs_edges = list(doublet_pairs_edges) + list(out[2])
        return doublet_vertices, pairs_edges_vertices, doublet_pairs_edges

    def find_doublets2(self, mesh, vertices):
        doublet_pair_edges = []
        # doublet_vertices = np.where(np.array([len(mesh.ve[j]) for j in range(len(mesh.ve))]) == 2)[0]
        doublet_vertices_indices = np.where(np.array([len(mesh.ve[v]) for v in vertices]) == 2)[0]
        doublet_vertices = [vertices[i] for i in doublet_vertices_indices]
        if len(doublet_vertices) > 0:
            doublet_pair_edges = [mesh.ve[v].copy() for v in doublet_vertices]

        # check if doublet has boundaries - if it has do not clear this doublet
        doublet_pair_edges_copy = doublet_pair_edges.copy()
        doublet_vertices_copy = doublet_vertices.copy()
        for i, doublet_pair in enumerate(doublet_pair_edges_copy):
            if np.any([self.has_boundaries2(mesh, d) for d in doublet_pair]):
                doublet_pair_edges.remove(doublet_pair)
                doublet_vertices.remove(doublet_vertices_copy[i])

        return doublet_vertices, doublet_pair_edges

    def clear_doublets2(self, vertices, mesh, mask, edge_groups):
        doublet_vertices, doublet_pairs_edges = self.find_doublets2(mesh, vertices)
        if len(doublet_vertices) == 0:
            return [], [], [], []

        pairs_edges_vertices = np.array([mesh.edges[e].copy() for e in doublet_pairs_edges])

        for i, doublet_pair_edges in enumerate(doublet_pairs_edges):
            vertex = doublet_vertices[i]
            for e in doublet_pair_edges:
                u, v = mesh.edges[e]

                MeshPool.__remove_group(mesh, edge_groups, e)  # TODO needed?
                mask[e] = False
                mesh.remove_edge(e)
                mesh.edges[e] = [-1, -1]  # TODO needed?
                mesh.edges_count -= 1

                # e_ns = mesh.gemm_edges[e].copy()
                # for j, e_n in enumerate(e_ns):
                #     side = mesh.sides[e, j]
                #     mesh.gemm_edges[e_n, side] = -1

                mesh.gemm_edges[e] = [-1, -1, -1, -1, -1, -1]

        # find the new connections instead of the old doublets
        replaced_doublets_edges = []
        replaced_doublets_vertices = []
        for p_i, pair_edges_vertices in enumerate(pairs_edges_vertices):
            doublet_vertex = doublet_vertices[p_i]
            other_vertices = set(pair_edges_vertices.flatten())
            other_vertices.remove(doublet_vertex)
            if len(other_vertices.intersection(set(doublet_vertices))) > 0:
                continue
            else:
                assert(len(other_vertices) == 2)
                # find other edges connected to the other vertices
                edges_1 = mesh.ve[other_vertices.pop()]
                edges_2 = mesh.ve[other_vertices.pop()]
                # find if there is a common vertex connected to it (remove u)
                # mutual_v = set(np.array(
                #     [mesh.edges[e] for e in e1]).flatten()).intersection(
                #     np.array([mesh.edges[e] for e in e2]).flatten())
                # mutual_v.remove(u)
                replaced_doublet = []
                for e1 in edges_1:
                    v_e1 = set(mesh.edges[e1])
                    if len(replaced_doublet) == 2:
                        break
                    for e2 in edges_2:
                        v_e2 = set(mesh.edges[e2])
                        mutual_v = v_e1.intersection(v_e2)
                        if len(mutual_v) == 1:
                            mutual_v = mutual_v.pop()
                            if mutual_v is not u:
                                replaced_doublet = [e1, e2]
                                break

                replaced_doublets_edges.append(replaced_doublet)
                if len(replaced_doublet) > 0:
                    replaced_doublets_vertices.append(mesh.edges[replaced_doublet[0]])
                    replaced_doublets_vertices.append(mesh.edges[replaced_doublet[1]])

        replaced_doublets_vertices = list(set(np.array(replaced_doublets_vertices).flatten()))
        replaced_doublets_edges = list(
            np.array(replaced_doublets_edges).flatten())

        out = self.clear_doublets2(replaced_doublets_vertices, mesh, mask, edge_groups)
        if len(out) > 0:
            doublet_vertices = list(doublet_vertices) + list(out[0])
            pairs_edges_vertices = list(pairs_edges_vertices) + list(out[1])
            doublet_pairs_edges = list(doublet_pairs_edges) + list(out[2])
            if len(out[2]) > 0:
                doublet_edges2 = out[2][0]

                replaced_doublets_edges_copy = replaced_doublets_edges.copy()
                for e in replaced_doublets_edges_copy:
                # for pair in replaced_doublets_edges:
                    if e in doublet_edges2:
                        replaced_doublets_edges.remove(e)

        return doublet_vertices, pairs_edges_vertices, doublet_pairs_edges, replaced_doublets_edges

    def find_diag_vertices(self, mesh, u, e_u, v_e_u):

        # for each pair of vertices in v_e_u find a connection via another edge
        diag_vertices = []  # diagonal vertices to vertex u
        # dictionary of optional new connections of vertex to edges :
        diag_vertex_to_edges_dict = dict()
        checked_vertex = []  # already checked vertex list
        for i, v_i in enumerate(v_e_u):
            checked_vertex.append(i)
            for j, v_j in enumerate(v_e_u):
                if j in checked_vertex:
                    continue  # do not check vertex twice

                # find mutual vertices via another edges start from v_i and v_j
                connection_vertices = \
                    self.get_vertices_connections_via_another_edge(mesh, v_i,
                                                                   v_j)
                # save only the new vertex that is not the original vertex u
                for con_vertex in connection_vertices:
                    if not con_vertex == u and con_vertex not in diag_vertices:
                        diag_vertices.append(con_vertex)
                        diag_vertex_to_edges_dict[str(con_vertex)] = \
                            list([e_u[i], e_u[j]])

        return diag_vertices, diag_vertex_to_edges_dict

    @staticmethod
    def find_all_one_ring_vertices(u, v_e_u, diag_vertices, diag_vertices_v,
                                   ve_to_reconnect_with_u):
        v_ring = set()
        v_ring.add(u)
        v_ring = v_ring.union(v_e_u)
        v_ring = v_ring.union(diag_vertices)
        v_ring = v_ring.union(diag_vertices_v)
        v_ring = v_ring.union(ve_to_reconnect_with_u)
        return v_ring

    def check_u_v_boundaries(self, mesh, u, v_e_u, e_u, v, v_e_v, e_v):
        correct_config = True
        # check if any edge comes from vertex u has boundary
        switch_u_v = np.any([self.has_boundaries2(mesh, e) for e in e_u])

        if switch_u_v:
            correct_config = not np.any([self.has_boundaries2(mesh, e) for e in e_v])
            if correct_config:
                # swap u and v
                u, v = v, u
                v_e_u, v_e_v = v_e_v, v_e_u
                e_u, e_v = e_v, e_u

        return correct_config, u, v_e_u, e_u, v, v_e_v, e_v

    def get_edge_hood_info(self, mesh, edge_id):
        # get vertices of edge with edge_id
        u, v = mesh.edges[edge_id]

        # get all edges connected to vertex u and all the vertices of them
        v_e_u, e_u = self.get_all_vertices_of_edges_connected_to_vertex(mesh,
                                                                        u)
        # get all edges connected to vertex v and all the vertices of them
        v_e_v, e_v = self.get_all_vertices_of_edges_connected_to_vertex(mesh,
                                                                        v)
        if len(e_u) > len(e_v):
            # swap u and v
            u, v = v, u
            v_e_u, v_e_v = v_e_v, v_e_u
            e_u, e_v = e_v, e_u

        return u, v_e_u, e_u, v, v_e_v, e_v

    @staticmethod
    def build_faces_from_edges_hood(mesh, edges):
        faces = []
        for edge in edges:
            face = [edge] + list(mesh.gemm_edges[edge, 0:3])
            faces.append(face)
            face = [edge] + list(mesh.gemm_edges[edge, 3:6])
            faces.append(face)
        return faces

    @staticmethod
    def restore_data(mesh, restored_mesh, features, restored_features,
                     mask, restored_mask, edge_groups, restored_groups):
        # restore mesh
        mesh.edge_areas = restored_mesh.edge_areas
        mesh.edge_lengths = restored_mesh.edge_lengths
        mesh.edges = restored_mesh.edges
        mesh.edges_count = restored_mesh.edges_count
        mesh.features = restored_mesh.features
        mesh.sides = restored_mesh.sides
        mesh.gemm_edges = restored_mesh.gemm_edges
        mesh.ve = restored_mesh.ve
        mesh.vs = restored_mesh.vs

        # restore features
        features.data = restored_features.data

        # restore mask
        mask[:] = restored_mask[:]

        # restore edge_groups
        edge_groups.groups = restored_groups.groups

        return mesh, features, mask, edge_groups

    @staticmethod
    def remove_edge(mesh, e, edge_groups, mask):
        MeshPool.__remove_group(mesh, edge_groups, e)
        mask[e] = False
        mesh.remove_edge(e)
        mesh.edges[e] = [-1, -1]
        mesh.edges_count -= 1

        # e_ns = mesh.gemm_edges[e].copy()
        # for j, e_n in enumerate(e_ns):
        #     side = mesh.sides[e, j]
        #     mesh.gemm_edges[e_n, side] = -1

        mesh.gemm_edges[e] = [-1, -1, -1, -1, -1, -1]

    @staticmethod
    def build_new_hood_for_diag_collapse(mesh, edge, removed_edges, new_faces):
        old_hood = mesh.gemm_edges[edge]

        # find the edge in the new faces to create the new hood
        if edge in new_faces[0]:
            new_face = new_faces[0]
            pos_in_new_face = np.where(np.array(new_face) == edge)[0][0]
        elif edge in new_faces[1]:
            new_face = new_faces[1]
            pos_in_new_face = np.where(np.array(new_face) == edge)[0][0]
        else:
            assert(False)

        # find in which side we should build the new hood in the old hood
        if np.any([r in old_hood[0:3] for r in removed_edges]):
            old_hood[0:3] = [new_face[(pos_in_new_face + 1) % 4],
                             new_face[(pos_in_new_face + 2) % 4],
                             new_face[(pos_in_new_face + 3) % 4]]
        elif np.any([r in old_hood[3:6] for r in removed_edges]):
            old_hood[3:6] = [new_face[(pos_in_new_face + 1) % 4],
                             new_face[(pos_in_new_face + 2) % 4],
                             new_face[(pos_in_new_face + 3) % 4]]
        else:
            assert(False)

        # fix hood order:
        if mesh.edges[old_hood[2], 0] not in mesh.edges[old_hood[3]] and \
            mesh.edges[old_hood[2], 1] not in mesh.edges[old_hood[3]]:
            old_hood[5], old_hood[3] = old_hood[3], old_hood[5]

        return

    def diagonal_collapse(self, mesh, edges_group, mask, features, origin_v, edge_id):
        # 1. Find edges to remove and reconnect
        v_edges = mesh.ve[origin_v].copy()
        v_edges.remove(edge_id)
        edge_to_remove = v_edges[0]
        edge_to_reconnect = v_edges[1]

        # 2. Find the diagonal vertex to origin vertex
        hood = mesh.gemm_edges[edge_id].copy()
        first_side = hood[0:3]
        second_side = hood[3:6]
        if edge_to_remove in first_side:
            hood = list(first_side)
        else:
            hood = list(second_side)

        edge_to_remove_pos = np.where(np.array(hood) == edge_to_remove)[0][0]
        hood.remove(edge_to_remove)
        if edge_to_remove_pos == 2:
            hood[0], hood[1] = hood[1], hood[0]

        # build new hood edges
        new_hood = mesh.gemm_edges[edge_to_reconnect]
        feature_combine = dict()
        for i, e in enumerate(mesh.gemm_edges[edge_to_reconnect]):
            if e == edge_id:
                mesh.gemm_edges[edge_to_reconnect, i] = hood[1]
                feature_combine[hood[1]] = [hood[1], edge_id]
            elif e == edge_to_remove:
                mesh.gemm_edges[edge_to_reconnect, i] = hood[0]
                feature_combine[hood[0]] = [hood[0], edge_to_remove]

        updated_fe = self.combine_edge_features(mesh, features,
                                                feature_combine,
                                                edges_group)

        v0 = set(mesh.edges[hood[0]].copy())
        v1 = set(mesh.edges[hood[1]].copy())
        v_to_reconnect = (v0.intersection(v1)).pop()

        # reconnect other edge to diagonal vertex
        orig_vertices = mesh.edges[edge_to_reconnect].copy()
        if orig_vertices[0] == origin_v:
            mesh.edges[edge_to_reconnect, 0] = v_to_reconnect
        else:
            mesh.edges[edge_to_reconnect, 1] = v_to_reconnect
        mesh.ve[origin_v].remove(edge_to_reconnect)  # remove old vertex connection
        mesh.ve[v_to_reconnect].append(edge_to_reconnect)  # add new vertex connection

        # build hood for other edges
        removed_edges = [edge_id, edge_to_remove]
        new_faces = [list(new_hood[0:3]) + [edge_to_reconnect],
                     list(new_hood[3:6]) + [edge_to_reconnect]]

        for edge in new_hood:
            self.build_new_hood_for_diag_collapse(mesh, edge, removed_edges, new_faces)

        # remove_edges
        self.remove_edge(mesh, edge_id, edges_group, mask)
        self.remove_edge(mesh, edge_to_remove, edges_group, mask)

        edges_list = [edge_to_reconnect] + list(new_hood)
        for edge in edges_list:
            new_sides = [-1, -1, -1, -1, -1, -1]
            for i_en, en in enumerate(mesh.gemm_edges[edge]):
                if en == -1:
                    continue
                side = np.where(mesh.gemm_edges[en] == edge)[0]
                if len(side) > 0:
                    new_sides[i_en] = side[0]

            mesh.sides[edge] = new_sides

        return updated_fe

    def mesh_decimation(self, mesh, fe, edge_id, mask, edge_groups):

        mesh_copy = deepcopy(mesh)
        fe_copy = fe.clone()
        mask_copy = deepcopy(mask)
        edge_groups_copy = deepcopy(edge_groups)

        # check if already removed this edge
        if not mask[edge_id]:
            print('edge_id %d already removed' % edge_id)
            return fe

        # 1. Get info
        u, v_e_u, e_u, v, v_e_v, e_v = self.get_edge_hood_info(mesh, edge_id)

        if len(e_u) == 3 and len(e_v) == 3:
            return fe

        if len(e_u) == 3:
            update_fe = self.diagonal_collapse(mesh, edge_groups, mask, fe, u, edge_id)
            return update_fe
        elif len(e_v) == 3:
            update_fe = self.diagonal_collapse(mesh, edge_groups, mask, fe, v, edge_id)
            return update_fe



        # check if u and v edges are with boundaries
        correct_config, u, v_e_u, e_u, v, v_e_v, e_v = \
            self.check_u_v_boundaries(mesh, u, v_e_u, e_u, v, v_e_v, e_v)

        if not correct_config:
            return fe

        # get outer edges by building faces
        faces_u = self.build_faces_from_edges_hood(mesh, e_u)
        faces_v = self.build_faces_from_edges_hood(mesh, e_v)
        all_faces = faces_u + faces_v
        all_edges = set().union(*all_faces)
        outer_edges = list(all_edges - set(e_u) - set(e_v))
        outer_vertices = list(
            set(np.array([mesh.edges[e] for e in outer_edges]).flatten()))

        # 2. Edges rotation
        # a. find diagonal vertices
        diag_vertices, diag_vertex_to_edges_dict = \
            self.find_diag_vertices(mesh, u, e_u, v_e_u)
        diag_vertices_v, diag_vertex_to_edges_dict_v = \
            self.find_diag_vertices(mesh, v, e_v, v_e_v)

        # b. rotate - for each edge goes from u - change the original
        # connection to the optional diagonal connection (direction should be
        # consistent for all the rotated edges)
        mesh, new_features_combination_dict = \
            self.rotate_edges_and_connections(mesh, u, e_u, diag_vertices,
                                              diag_vertex_to_edges_dict)

        # c. collapse another 2 edges connected to the other vertex v and
        # reconnect other edges from v connection to u connection
        e_v = mesh.ve[v].copy()  # edges connected to vertex v
        e_to_collapse, e_to_reconnect_with_u = \
            self.collapse_other_vertex_v(mesh, u, v, e_v, diag_vertices,
                                         new_features_combination_dict,
                                         edge_groups, mask)

        # 3. Clear all doublets in the mesh
        doublet_vertices, pairs_edges_vertices, doublet_pairs_edges, replaced_doublets_edges2 = \
            self.clear_doublets2(outer_vertices, mesh, mask, edge_groups)

        # return and restore old connections if we completely removed
        # edge_id vertices
        if len(mesh.ve[u]) == 0 and len(mesh.ve[v]) == 0:
        # if len(doublet_pairs_edges) > 1:
            mesh, features, mask, edge_groups = \
                self.restore_data(mesh, mesh_copy, fe, fe_copy, mask,
                                  mask_copy, edge_groups, edge_groups_copy)
            return fe

        # find the new connections instead of the old doublets
        replaced_doublets_edges = []
        replaced_doublets_vertices = []
        for p_i, pair_edges_vertices in enumerate(pairs_edges_vertices):
            doublet_vertex = doublet_vertices[p_i]
            other_vertices = set(pair_edges_vertices.flatten())
            other_vertices.remove(doublet_vertex)
            if len(other_vertices.intersection(set(doublet_vertices))) > 0:
                continue
            else:
                assert(len(other_vertices) == 2)
                # find other edges connected to the other vertices
                edges_1 = mesh.ve[other_vertices.pop()]
                edges_2 = mesh.ve[other_vertices.pop()]
                # find if there is a common vertex connected to it (remove u)
                replaced_doublet = []
                for e1 in edges_1:
                    v_e1 = set(mesh.edges[e1])
                    if len(replaced_doublet) == 2:
                        break
                    for e2 in edges_2:
                        v_e2 = set(mesh.edges[e2])
                        mutual_v = v_e1.intersection(v_e2)
                        if len(mutual_v) == 1:
                            mutual_v = mutual_v.pop()
                            if mutual_v is not u:
                                replaced_doublet = [e1, e2]
                                break

                replaced_doublets_edges.append(replaced_doublet)
                if len(replaced_doublet) > 0:
                    replaced_doublets_vertices.append(mesh.edges[replaced_doublet[0]])
                    replaced_doublets_vertices.append(mesh.edges[replaced_doublet[1]])

        replaced_doublets_edges = list(np.array(replaced_doublets_edges).flatten()) + replaced_doublets_edges2
        replaced_doublets_vertices = list(np.array(replaced_doublets_vertices).flatten())

        # 4. create new edge features  # TODO should it be here?
        updated_fe = self.combine_edge_features(mesh, fe,
                                                new_features_combination_dict,
                                                edge_groups)

        # 5. Re-build edges neighborhood
        # a. Get new faces - inner ring
        faces, faces_vertices = \
            self.get_all_faces_connected_to_vertex(mesh, u)

        # b. update outer_edges
        outer_edges = set(outer_edges).union(set(replaced_doublets_edges)) -\
                      set(set().union(*doublet_pairs_edges))  # consider removing the inner e_u edges

        # c. keep only faces with outer edges:
        keep = []
        for f_id, face in enumerate(faces):
            if sum([e in outer_edges for e in face]) > 1:
                keep.append(f_id)
        faces = (np.array(faces)[keep]).tolist()
        faces_vertices = (np.array(faces_vertices)[keep]).tolist()

        # d. find new outer ring faces (only if it have one edge from the
        # 1-ring outer edges
        vers = list(set().union(*faces_vertices))
        for ver in vers:
            f, fv = \
                self.get_all_faces_connected_to_vertex(mesh, ver)
            for i, fv_i in enumerate(fv):

                # first ring faces can't have u or v vertices
                if u in fv_i or v in fv_i:
                    continue

                faces_vertices_copy = faces_vertices.copy()
                is_new = True
                for cfv in faces_vertices_copy:
                    diff = len(set(cfv).difference(set(fv_i)))
                    if diff == 0:
                        is_new = False
                        break
                without_boundary = True
                if is_new and np.any([e in outer_edges for e in f[i]]) and without_boundary:
                    faces_vertices.append(fv_i)
                    faces.append(f[i])

        # e. find second-ring faces
        vers = list(set().union(*faces_vertices))
        faces_vertices_second_ring = faces_vertices.copy()
        faces_second_ring = faces.copy()
        for ver in vers:
            f, fv = \
                self.get_all_faces_connected_to_vertex(mesh, ver)
            for i, fv_i in enumerate(fv):

                # second ring faces can't have u or v vertices
                if u in fv_i or v in fv_i:
                    continue

                # check if this is a new face or not
                faces_vertices_copy = faces_vertices.copy()
                is_new = True  # init
                for cfv in faces_vertices_copy:
                    diff = len(set(cfv).difference(set(fv_i)))
                    if diff == 0:
                        is_new = False
                        break
                if is_new:
                    faces_vertices_second_ring.append(fv_i)
                    faces_second_ring.append(f[i])

        # f. Remove double faces from list
        for fi, face in enumerate(faces):
            self.find_double_faces(mesh, face, faces, faces_vertices[fi], faces_vertices)
        for fi, face in enumerate(faces_second_ring):
            removed_faces, removed_faces_vertices = self.find_double_faces(mesh, face, faces_second_ring, faces_vertices_second_ring[fi], faces_vertices_second_ring)
            for irf, rf in enumerate(removed_faces):
                faces_copy = faces.copy()
                if rf in faces_copy:
                    faces.remove(rf)
                    faces_vertices.remove(removed_faces_vertices[irf])

        # g. Re-build edges hood and sides
        self.build_edges_hood_and_sides(mesh, faces_vertices)

        return updated_fe

    def __is_valid_config_one_vertex(self, mesh, edge_id, vertex_k):
        """
        checks that each face has only one mutual edge
        """
        e_k = mesh.ve[mesh.edges[edge_id, vertex_k]]
        e_k_hood = []

        faces, faces_vertices = \
            self.get_all_faces_connected_to_vertex(mesh, mesh.edges[edge_id, vertex_k])

        for e in e_k:
            tmp = set(mesh.gemm_edges[e, 0:3])
            tmp.add(e)
            e_k_hood.append(tmp)
            tmp = set(mesh.gemm_edges[e, 3:6])
            tmp.add(e)
            e_k_hood.append(tmp)

        for i, e_k_i in enumerate(faces):
            for j, e_k_j in enumerate(faces):
                if i == j:
                    continue
                count = len(set.intersection(set(e_k_i), set(e_k_j)))
                if count == 2:  # case of 1 is OK, case of 4 is the whole face
                    return False
                elif count == 3:
                    raise ValueError('Number of mutual edges can not be 3!')

        return True

    def __is_valid_config(self, mesh, edge_id):
        # checks that all faces connected with edge_id has only one mutual edge
        return self.__is_valid_config_one_vertex(mesh, edge_id, 0) and \
               self.__is_valid_config_one_vertex(mesh, edge_id, 1)

    def __clean_side(self, mesh, edge_id, mask, edge_groups, side):
        if mesh.edges_count <= self.__out_target:
            return False
        invalid_edges = MeshPool.__get_invalids(mesh, edge_id, edge_groups,
                                                side)
        while len(invalid_edges) != 0 and mesh.edges_count > self.__out_target:
            self.__remove_triplete(mesh, mask, edge_groups, invalid_edges)
            if mesh.edges_count <= self.__out_target:
                return False
            if self.has_boundaries(mesh, edge_id):
                return False
            invalid_edges = self.__get_invalids(mesh, edge_id, edge_groups,
                                                side)
        return True

    @staticmethod
    def has_boundaries(mesh, edge_id):
        for edge in mesh.gemm_edges[edge_id]:
            if edge == -1 or -1 in mesh.gemm_edges[edge]:
                return True
        return False

    @staticmethod
    def has_boundaries2(mesh, edge_id):
        if -1 in mesh.gemm_edges[edge_id]:
            return True
        return False

    @staticmethod
    def __get_v_n(mesh, edge_id):
        return set(mesh.edges[mesh.ve[mesh.edges[edge_id, 0]]].reshape(-1)), \
               set(mesh.edges[mesh.ve[mesh.edges[edge_id, 1]]].reshape(-1)),

    def __is_one_ring_valid(self, mesh, edge_id):
        # v_a, v_b = get_v_n(mesh, edge_id)
        e_a = mesh.ve[mesh.edges[edge_id, 0]]
        e_b = mesh.ve[mesh.edges[edge_id, 1]]

        v_a = set()  # set of all neighbor + diagonal vertices of first edge vertex
        v_b = set()  # set of all neighbor + diagonal vertices of second edge vertex
        for e in e_a:
            if not e == edge_id:
                v_aa, v_ab = self.__get_v_n(mesh, e)
                v_a = set.union(set.union(v_aa, v_ab), v_a)

        for e in e_b:
            if not e == edge_id:
                v_ba, v_bb = self.__get_v_n(mesh, e)
                v_b = set.union(set.union(v_ba, v_bb), v_b)

        shared = v_a & v_b - set(mesh.edges[edge_id])
        return len(shared) == 4

    def __pool_side(self, mesh, edge_id, mask, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, side_a, side_b, _, other_side_b, _, other_keys_b = info
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2,
                              other_keys_b[0], mesh.sides[key_b, other_side_b])
        self.__redirect_edges(mesh, key_a, side_a - side_a % 2 + 1,
                              other_keys_b[1],
                              mesh.sides[key_b, other_side_b + 1])
        MeshPool.__union_groups(mesh, edge_groups, key_b, key_a)
        MeshPool.__union_groups(mesh, edge_groups, edge_id, key_a)
        mask[key_b] = False
        MeshPool.__remove_group(mesh, edge_groups, key_b)
        mesh.remove_edge(key_b)
        mesh.edges_count -= 1
        return key_a

    @staticmethod
    def __get_invalids(mesh, edge_id, edge_groups, side):
        info = MeshPool.__get_face_info(mesh, edge_id, side)
        key_a, key_b, key_c, side_a, side_b, side_c, \
        other_side_a, other_side_b, other_side_c, \
        other_keys_a, other_keys_b, other_keys_c = info
        shared_items_ab = MeshPool.__get_shared_items(other_keys_a, other_keys_b)
        shared_items_ac = MeshPool.__get_shared_items(other_keys_a, other_keys_c)
        shared_items_bc = MeshPool.__get_shared_items(other_keys_b, other_keys_c)
        if len(shared_items_ab) <= 2 and len(shared_items_ac) <= 2 and \
                len(shared_items_bc) <= 2:
            return []
        else:
            if len(shared_items_ab) > 2:
                shared_items = shared_items_ab
            elif len(shared_items_ac) > 2:
                shared_items = shared_items_ac
            else:
                shared_items = shared_items_bc

            assert (len(shared_items) == 4)
            # TODO - shold change all of the following:
            middle_edge = other_keys_a[shared_items[0]]
            update_key_a = other_keys_a[1 - shared_items[0]]
            update_key_b = other_keys_b[1 - shared_items[1]]
            update_side_a = mesh.sides[
                key_a, other_side_a + 1 - shared_items[0]]
            update_side_b = mesh.sides[
                key_b, other_side_b + 1 - shared_items[1]]
            MeshPool.__redirect_edges(mesh, edge_id, side, update_key_a,
                                      update_side_a)
            MeshPool.__redirect_edges(mesh, edge_id, side + 1, update_key_b,
                                      update_side_b)
            MeshPool.__redirect_edges(mesh, update_key_a,
                                      MeshPool.__get_other_side(update_side_a),
                                      update_key_b,
                                      MeshPool.__get_other_side(update_side_b))
            MeshPool.__union_groups(mesh, edge_groups, key_a, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_b, edge_id)
            MeshPool.__union_groups(mesh, edge_groups, key_a, update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge,
                                    update_key_a)
            MeshPool.__union_groups(mesh, edge_groups, key_b, update_key_b)
            MeshPool.__union_groups(mesh, edge_groups, middle_edge,
                                    update_key_b)
            return [key_a, key_b, middle_edge]

    @staticmethod
    def __redirect_edges(mesh, edge_a_key, side_a, edge_b_key, side_b):
        mesh.gemm_edges[edge_a_key, side_a] = edge_b_key
        mesh.gemm_edges[edge_b_key, side_b] = edge_a_key
        mesh.sides[edge_a_key, side_a] = side_b
        mesh.sides[edge_b_key, side_b] = side_a

    @staticmethod
    def __get_shared_items(list_a, list_b):
        shared_items = []
        for i in range(len(list_a)):
            for j in range(len(list_b)):
                if list_a[i] == list_b[j]:
                    shared_items.extend([i, j])
        return shared_items

    @staticmethod
    def __get_other_side(side):
        return side + 1 - 2 * (side % 2)

    @staticmethod
    def __get_face_info(mesh, edge_id, side):
        key_a = mesh.gemm_edges[edge_id, side]
        key_b = mesh.gemm_edges[edge_id, side + 1]
        key_c = mesh.gemm_edges[edge_id, side + 2]
        side_a = mesh.sides[edge_id, side]
        side_b = mesh.sides[edge_id, side + 1]
        side_c = mesh.sides[edge_id, side + 2]
        other_side_a = (side_a - (side_a % 3) + 3) % 6
        other_side_b = (side_b - (side_b % 3) + 3) % 6
        other_side_c = (side_c - (side_c % 3) + 3) % 6
        other_keys_a = [mesh.gemm_edges[key_a, other_side_a],
                        mesh.gemm_edges[key_a, other_side_a + 1],
                        mesh.gemm_edges[key_a, other_side_a + 2]]
        other_keys_b = [mesh.gemm_edges[key_b, other_side_b],
                        mesh.gemm_edges[key_b, other_side_b + 1],
                        mesh.gemm_edges[key_b, other_side_b + 2]]
        other_keys_c = [mesh.gemm_edges[key_c, other_side_c],
                        mesh.gemm_edges[key_c, other_side_c + 1],
                        mesh.gemm_edges[key_c, other_side_c + 2]]
        return key_a, key_b, key_c, side_a, side_b, side_c, \
               other_side_a, other_side_b, other_side_c, \
               other_keys_a, other_keys_b, other_keys_c

    @staticmethod
    def __remove_triplete(mesh, mask, edge_groups, invalid_edges):
        vertex = set(mesh.edges[invalid_edges[0]])
        for edge_key in invalid_edges:
            vertex &= set(mesh.edges[edge_key])
            mask[edge_key] = False
            MeshPool.__remove_group(mesh, edge_groups, edge_key)
        mesh.edges_count -= 3
        vertex = list(vertex)
        assert (len(vertex) == 1)
        mesh.remove_vertex(vertex[0])

    @staticmethod
    def __build_queue(features, edges_count):
        # delete edges with smallest norm
        squared_magnitude = torch.sum(features * features, 0)
        if squared_magnitude.shape[-1] != 1:
            squared_magnitude = squared_magnitude.unsqueeze(-1)
        edge_ids = torch.arange(edges_count, device=squared_magnitude.device,
                                dtype=torch.float32).unsqueeze(-1)
        heap = torch.cat((squared_magnitude, edge_ids), dim=-1).tolist()
        heapify(heap)
        return heap

    @staticmethod
    def __union_groups(mesh, edge_groups, source, target):
        edge_groups.union(source, target)
        mesh.union_groups(source, target)

    @staticmethod
    def __remove_group(mesh, edge_groups, index):
        edge_groups.remove_group(index)
        mesh.remove_group(index)
