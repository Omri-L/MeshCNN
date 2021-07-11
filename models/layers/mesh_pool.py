import torch
import torch.nn as nn
from threading import Thread
from models.layers.mesh_union import MeshUnion
import numpy as np
from heapq import heappop, heapify
import random


class MeshPool(nn.Module):

    def __init__(self, target, multi_thread=False):
        super(MeshPool, self).__init__()
        self.__out_target = target
        self.__multi_thread = multi_thread
        self.__fe = None
        self.__updated_fe = None
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

        while mesh.edges_count > self.__out_target:
            value, edge_id = heappop(queue)
            edge_id = int(edge_id)
            print('pool edge_id %d' % edge_id)
            if mask[edge_id]:
                _, fe = self.__pool_edge(mesh, edge_id, fe, mask, edge_groups)
        mesh.clean(mask, edge_groups)
        # fe = edge_groups.rebuild_features(self.__fe[mesh_index], mask,
        #                                   self.__out_target)  # TODO use features build here?
        self.__updated_fe[mesh_index] = fe

    def __pool_edge(self, mesh, edge_id, fe, mask, edge_groups):

        if not np.all(mesh.gemm_edges[mesh.gemm_edges[edge_id],
                                      mesh.sides[edge_id]] == edge_id):
            new_sides = np.where(mesh.gemm_edges[mesh.gemm_edges[edge_id]]
                                 == edge_id)[1]
            mesh.sides[edge_id] = new_sides

        if self.has_boundaries(mesh, edge_id):
            return False, fe
        elif self.__clean_side(mesh, edge_id, mask, edge_groups, 0) \
                and self.__clean_side(mesh, edge_id, mask, edge_groups, 3) \
                and self.__is_one_ring_valid(mesh, edge_id):

            fe = self.rotate_edges(mesh, fe, edge_id, mask, edge_groups)
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
                if len(connection_vertices) == 1:
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
            if -1 in edge_nb[i]:
                old_gemm = mesh.gemm_edges[abs_edge]
                new_gemm = edge_nb[i]
                old_pos = np.where(old_gemm == new_gemm[0])[0].flatten()

                # new_gemm = old_gemm.tolist()
                # if old_pos >= 0:
                #     while old_pos > 0:
                #         new_gemm = new_gemm[-1:] + new_gemm[:-1]
                #         old_pos -= 1
                # else:
                #     assert (0)
                # new_gemm = np.array(new_gemm)

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
            else:
                mesh.gemm_edges[abs_edge] = edge_nb[i]
                mesh.sides[abs_edge] = sides[i]

        # I. fix sides
        for abs_edge in abs_edges:
            if not np.all(mesh.gemm_edges[
                              mesh.gemm_edges[abs_edge], mesh.sides[
                                  abs_edge]] == abs_edge):
                new_sides = np.where(
                    mesh.gemm_edges[mesh.gemm_edges[abs_edge]] == abs_edge)[1]
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
            for fi, f in enumerate(faces):
                if f == face:
                    continue
                else:
                    is_outer_face = len(outer_vertices.difference(set(faces_vertices[fi]))) == 0
                    if is_outer_face:
                        removed_faces.append(f)
                        removed_faces_vertices.append(faces_vertices[fi])
                        faces.remove(f)
                        faces_vertices.remove(faces_vertices[fi])
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
                e_ns = mesh.gemm_edges[e].copy()
                for j, e_n in enumerate(e_ns):
                    side = mesh.sides[e, j]
                    mesh.gemm_edges[e_n, side] = -1

                mesh.gemm_edges[e] = [-1, -1, -1, -1, -1, -1]

        out = self.clear_doublets(mesh, mask, edge_groups)
        if len(out) > 0:
            doublet_vertices = list(doublet_vertices) + list(out[0])
            pairs_edges_vertices = list(pairs_edges_vertices) + list(out[1])
            doublet_pairs_edges = list(doublet_pairs_edges) + list(out[2])
        return doublet_vertices, pairs_edges_vertices, doublet_pairs_edges

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

    def rotate_edges(self, mesh, fe, edge_id, mask, edge_groups):

        # check if already removed this edge
        if not mask[edge_id]:
            print('edge_id %d already removed' % edge_id)
            return fe

        # 1. Get info
        # get vertices of edge with edge_id
        u, v = mesh.edges[edge_id]

        # get all edges connected to vertex u and all the vertices of them
        v_e_u, e_u = self.get_all_vertices_of_edges_connected_to_vertex(mesh,
                                                                        u)
        # get all edges connected to vertex v and all the vertices of them
        v_e_v, e_v = self.get_all_vertices_of_edges_connected_to_vertex(mesh,
                                                                        v)

        # get outer edges
        faces_u, _ = self.get_all_faces_connected_to_vertex(mesh, u)
        faces_v, _ = self.get_all_faces_connected_to_vertex(mesh, v)
        all_faces = faces_u + faces_v
        all_edges = set().union(*all_faces)
        outer_edges = list(all_edges - set(e_u) - set(e_v))

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
        doublet_vertices, pairs_edges_vertices, doublet_pairs_edges = \
            self.clear_doublets(mesh, mask, edge_groups)

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

        replaced_doublets_vertices = list(np.array(replaced_doublets_vertices).flatten())

        # 4. create new edge features  # TODO should it be here?
        updated_fe = self.combine_edge_features(mesh, fe,
                                                new_features_combination_dict,
                                                edge_groups)

        # 5. Re-build edges neighborhood
        # a. Get new faces - inner ring
        faces, faces_vertices = \
            self.get_all_faces_connected_to_vertex(mesh, u)

        # b. find outer vertices
        # outer_edges = list(set().union(*faces) - set(e_u))
        v_e_u2, e_u2 = self.get_all_vertices_of_edges_connected_to_vertex(mesh, u)
        outer_vertices = set(v_e_u2 + v_e_u + v_e_v + \
                             diag_vertices_v + replaced_doublets_vertices) - \
                         set(doublet_vertices) - set([u, v])

        # new - update outer_edges
        outer_edges = set(outer_edges).union(set(set().union(*replaced_doublets_edges))) -\
                      set(set().union(*doublet_pairs_edges))


        # c. keep only faces with outer edges:
        keep = []
        for f_id, face in enumerate(faces):
            # if np.any([len(set(mesh.edges[e]).intersection(outer_vertices))
            #            == 2 for e in face]):
            if np.any([e in outer_edges for e in face]):
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
                faces_vertices_copy = faces_vertices.copy()
                is_new = True
                for cfv in faces_vertices_copy:
                    diff = len(set(cfv).difference(set(fv_i)))
                    if diff == 0:
                        is_new = False
                        break
                # if not edge_id == 13254000:
                #     if is_new and np.any([o in f[i] for o in outer_edges]):
                #         faces_vertices.append(fv_i)
                #         faces.append(f[i])
                # else:
                #     if is_new and np.any([len(set(mesh.edges[efi]).intersection(outer_vertices)) == 2 for efi in f[i]]):
                #         faces_vertices.append(fv_i)
                #         faces.append(f[i])
                # if is_new and np.any([len(
                #         set(mesh.edges[efi]).intersection(outer_vertices)) == 2
                #                       for efi in f[i]]):
                if is_new and np.any([efi in outer_vertices for efi in f[i]]):
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
                faces_vertices_copy = faces_vertices.copy()
                is_new = True
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
                if rf in faces:
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
