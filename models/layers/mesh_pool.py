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
        # if self.__out_target == 430:
        #     ids = [261, 199]
        # elif self.__out_target == 380:
        #     ids = [403, 405]
        # elif self.__out_target == 300:
        #     ids = [357, 377]
        # elif self.__out_target == 250:
        #     ids = [276, 278]
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
                status = self.__pool_edge(mesh, edge_id, mask, edge_groups)
        mesh.clean(mask, edge_groups)
        fe = edge_groups.rebuild_features(self.__fe[mesh_index], mask, self.__out_target)
        self.__updated_fe[mesh_index] = fe
        print('finish pooling')

    def clean_mesh_operations(self, mesh, mask, edge_groups):

        # clear doublets and build new hood
        doublet_cleared = self.clear_doublets(mesh, mask, edge_groups)
        while doublet_cleared:
            doublet_cleared = self.clear_doublets(mesh, mask, edge_groups)

        # clear singlets and build new hood
        # self.clear_singlets(mesh, mask, edge_groups)
        return

    def __pool_edge(self, mesh, edge_id, mask, edge_groups):

        # 1. clean mesh operations
        self.clean_mesh_operations(mesh, mask, edge_groups)

        # 2. check if edge_id have boundaries
        if self.has_boundaries(mesh, edge_id):
            return False

        # TODO check if we really need this if
        if self.__clean_side(mesh, edge_id, mask, edge_groups, 0) \
                and self.__clean_side(mesh, edge_id, mask, edge_groups, 3) \
                and self.__is_one_ring_valid(mesh, edge_id):

            self.edge_collapse(edge_id, mesh, mask, edge_groups)
            # mesh.merge_vertices(edge_id)  # TODO should update this method
            return True
        else:
            return False

    def edge_collapse(self, edge_id, mesh, mask, edge_groups):
        # 1. Get edge info
        u, v_e_u, e_u, v, v_e_v, e_v = mesh.get_edge_hood_info(edge_id)

        # check if u and v edges are with boundaries
        correct_config, u, v_e_u, e_u, v, v_e_v, e_v = \
            self.check_u_v_boundaries(mesh, u, v_e_u, e_u, v, v_e_v, e_v)

        if not correct_config:
            return

        # 2. Edges rotations around vertex u
        mesh, new_features_combination_dict, diag_vertices = \
            self.edge_rotations(edge_id, u, e_u, v_e_u, mesh)

        # 3. collapse another 2 edges connected to the other vertex v and
        # reconnect other edges from v connection to u connection
        e_v = mesh.ve[v].copy()  # edges connected to vertex v
        self.collapse_other_vertex_v(mesh, u, v, e_v, diag_vertices,
                                     new_features_combination_dict,
                                     edge_groups, mask)

        # 4. union edge groups
        MeshPool.__union_groups_at_once(mesh, edge_groups, new_features_combination_dict)

    def check_u_v_boundaries(self, mesh, u, v_e_u, e_u, v, v_e_v, e_v):
        correct_config = True
        # check if any edge comes from vertex u has boundary
        switch_u_v = np.any([self.has_boundaries_edge_only(mesh, e) for e in e_u])

        if switch_u_v:
            correct_config = not np.any([self.has_boundaries_edge_only(mesh, e) for e in e_v])
            if correct_config:
                # swap u and v
                u, v = v, u
                v_e_u, v_e_v = v_e_v, v_e_u
                e_u, e_v = e_v, e_u

        return correct_config, u, v_e_u, e_u, v, v_e_v, e_v

    def edge_rotations(self, edge_id, u, e_u, v_e_u, mesh):
        # 1. Find diagonal vertices
        diag_vertices, diag_vertex_to_edges_dict = \
            self.find_diag_vertices(mesh, edge_id, u, e_u, v_e_u)

        # 2. Rotate edges - for each edge goes from u - change the original
        # connection to the optional diagonal connection (direction should be
        # consistent for all the rotated edges)
        mesh, new_features_combination_dict = \
            self.rotate_edges_and_connections(mesh, u, e_u, diag_vertices,
                                              diag_vertex_to_edges_dict)

        return mesh, new_features_combination_dict, diag_vertices

    def find_diag_vertices(self, mesh, edge_id, u, e_u, v_e_u):

        # find outer edges
        all_edges = list(
            set(np.array([mesh.gemm_edges[e] for e in e_u]).flatten()))
        outer_edges = list(set(all_edges) - set(e_u))

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

                # find mutual vertices via another outer edge start from v_i and v_j
                connection_vertices = \
                    self.get_vertices_connections_via_another_edge(mesh, v_i,
                                                                   v_j, outer_edges)
                # save only the new vertex that is not the original vertex u
                for con_vertex in connection_vertices:
                    if not con_vertex == u and con_vertex not in diag_vertices:
                        diag_vertices.append(con_vertex)
                        diag_vertex_to_edges_dict[str(con_vertex)] = \
                            list([e_u[i], e_u[j]])

        return diag_vertices, diag_vertex_to_edges_dict

    def rotate_edges_and_connections(self, mesh, origin_vertex, edges_to_change,
                                     optional_vertices,
                                     optional_vertex_to_edges_dict):
        """
        For each edge goes from original vertex: change the original connection
        to another optional vertex connection in a way that all vertices have
        a new connection (it will make sure the rotation for all will be CCW or
        CW).
        Re-build edge neighborhoods according to the edge rotation.
        """
        old_mesh = deepcopy(mesh)
        edges_to_change_cpy = edges_to_change.copy()  # make a copy
        optional_vertices_cpy = optional_vertices.copy()  # make a copy
        e = edges_to_change_cpy[0]  # first edge in list (init)
        new_features_combination_dict = dict()
        edge_to_new_vertex_connection_dict = dict()
        while len(edges_to_change_cpy) > 0:
            edges_to_change_cpy.remove(e)
            # check which connection is possible according to connectivity
            optional_new_v = set.intersection(set(optional_vertices_cpy), set(
                mesh.edges[mesh.gemm_edges[e]].reshape(-1)))
            # choose one vertex randomly
            random.seed(0)
            v_e_new = random.choice(list(optional_new_v))
            edge_to_new_vertex_connection_dict[e] = v_e_new
            # remove it from options
            optional_vertices_cpy.remove(v_e_new)
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

        edge_hood_built = []
        for e in edge_to_new_vertex_connection_dict:
            # build the new hood for rotated edges
            mesh.gemm_edges[e] = self.edge_rotation_new_hood(mesh, old_mesh, e,
                                                             edge_to_new_vertex_connection_dict[e])
            edge_hood_built.append(e)

        # re-build new hood for the other hood edges of rotated edges
        for e in edge_to_new_vertex_connection_dict:
            new_hood = list(mesh.gemm_edges[e])
            new_faces = [new_hood[0:3] + [e], new_hood[3:6] + [e]]
            for edge in new_hood:
                if edge in edge_hood_built:
                    continue
                else:
                    self.build_new_hood_for_diag_collapse(mesh, edge, edges_to_change,
                                                      new_faces)
                    edge_hood_built.append(edge)

        # fix sides
        self.__fix_mesh_sides(mesh, edge_hood_built)

        return mesh, new_features_combination_dict

    @staticmethod
    def edge_rotation_new_hood(mesh, old_mesh, e, new_vertex):

        old_hood = old_mesh.gemm_edges[e]
        # find the two edges in e hood which connected to the new_vertex
        new_vertex_old_optional_edges_pairs = [[0, 1], [1, 2], [3, 4], [4, 5]]

        for pair_case, pair in enumerate(new_vertex_old_optional_edges_pairs):
            v0 = set(old_mesh.edges[old_hood[pair[0]]])
            v1 = set(old_mesh.edges[old_hood[pair[1]]])
            mutual_v = v0.intersection(v1)
            if len(mutual_v) == 1:
                if new_vertex == mutual_v.pop():
                    break

        if pair_case == 0:
            new_hood = [old_hood[1], -1, old_hood[2], old_hood[3], old_hood[5],
                        old_hood[0]]
            # find the missed edge
            v0 = set(mesh.edges[new_hood[0]]).difference(
                set(old_mesh.edges[new_hood[0]]))
            if len(v0) == 0:
                v0 = set(mesh.edges[new_hood[2]]).difference(
                    set(old_mesh.edges[new_hood[2]]))
            v1 = set(old_mesh.edges[new_hood[0]]).intersection(
                set(old_mesh.edges[new_hood[2]]))
            missed_e = set(old_mesh.ve[v0.pop()]).intersection(set(old_mesh.ve[v1.pop()]))
            new_hood[1] = missed_e.pop()
        elif pair_case == 1:
            new_hood = [old_hood[0], -1, old_hood[1], old_hood[2], old_hood[3],
                        old_hood[5]]
            # find the missed edge
            v0 = set(mesh.edges[new_hood[0]]).difference(
                set(old_mesh.edges[new_hood[0]]))
            if len(v0) == 0:
                v0 = set(mesh.edges[new_hood[2]]).difference(
                    set(old_mesh.edges[new_hood[2]]))
            v1 = set(old_mesh.edges[new_hood[0]]).intersection(
                set(old_mesh.edges[new_hood[2]]))
            missed_e = set(old_mesh.ve[v0.pop()]).intersection(set(old_mesh.ve[v1.pop()]))
            new_hood[1] = missed_e.pop()
        elif pair_case == 2:
            new_hood = [old_hood[0], old_hood[2], old_hood[3], old_hood[4], -1,
                        old_hood[5]]
            # find the missed edge
            v0 = set(mesh.edges[new_hood[3]]).difference(
                set(old_mesh.edges[new_hood[3]]))
            if len(v0) == 0:
                v0 = set(mesh.edges[new_hood[5]]).difference(
                    set(old_mesh.edges[new_hood[5]]))
            v1 = set(old_mesh.edges[new_hood[3]]).intersection(
                set(old_mesh.edges[new_hood[5]]))
            missed_e = set(old_mesh.ve[v0.pop()]).intersection(set(old_mesh.ve[v1.pop()]))
            new_hood[4] = missed_e.pop()
        elif pair_case == 3:
            new_hood = [old_hood[5], old_hood[0], old_hood[2], old_hood[3], -1,
                        old_hood[4]]
            # find the missed edge
            v0 = set(mesh.edges[new_hood[3]]).difference(
                set(old_mesh.edges[new_hood[3]]))
            if len(v0) == 0:
                v0 = set(mesh.edges[new_hood[5]]).difference(
                    set(old_mesh.edges[new_hood[5]]))
            v1 = set(old_mesh.edges[new_hood[3]]).intersection(
                set(old_mesh.edges[new_hood[5]]))
            missed_e = set(old_mesh.ve[v0.pop()]).intersection(set(old_mesh.ve[v1.pop()]))
            new_hood[4] = missed_e.pop()
        else:
            assert(False)

        return new_hood

    def collapse_other_vertex_v(self, mesh, u, v, e_v, diag_vertices,
                                new_features_combination_dict, edge_groups,
                                mask):
        if self.clear_doublets(mesh, mask, edge_groups, [v]):
            return

        old_mesh = deepcopy(mesh)

        e_to_collapse = []  # edges we should remove
        collapsed_e_to_orig_e_dict = dict()  # to which edge the collpased edge combined with
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
                        collapsed_e_to_orig_e_dict[e] = key
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

        # fix hood for edges which re-connected to u
        already_built_edges_hood = []
        for e in e_to_reconnect_with_u:
            hood = old_mesh.gemm_edges[e]
            edges_to_check = [e] + list(hood)
            for edge in edges_to_check:
                if edge in already_built_edges_hood or edge in e_to_collapse:
                    continue

                already_built_edges_hood.append(edge)
                old_hood = old_mesh.gemm_edges[edge]
                new_hood = mesh.gemm_edges[edge]
                # replace any e_to_collapse edge by the matched edge
                for e_collapse in e_to_collapse:
                    if np.any([h == e_collapse for h in old_hood]):
                        e_collapse_pos = \
                        np.where([h == e_collapse for h in old_hood])[0][0]
                        new_hood[e_collapse_pos] = collapsed_e_to_orig_e_dict[e_collapse]

        # now fix hood for the rotated edges
        for key in collapsed_e_to_orig_e_dict:
            edge = collapsed_e_to_orig_e_dict[key]
            old_hood = old_mesh.gemm_edges[edge]
            new_hood = mesh.gemm_edges[edge]
            already_built_edges_hood.append(edge)
            if key in old_hood[0:3]:
                if edge not in old_mesh.gemm_edges[key, 0:3]:
                    new_hood[0:3] = old_mesh.gemm_edges[key, 0:3]
                else:
                    new_hood[0:3] = old_mesh.gemm_edges[key, 3:6]
            elif key in old_hood[3:6]:
                if edge not in old_mesh.gemm_edges[key, 0:3]:
                    new_hood[3:6] = old_mesh.gemm_edges[key, 0:3]
                else:
                    new_hood[3:6] = old_mesh.gemm_edges[key, 3:6]
            else:
                assert(False)

            for i, e in enumerate(new_hood):
                if e in collapsed_e_to_orig_e_dict.keys():
                    new_hood[i] = collapsed_e_to_orig_e_dict[e]

        # fix hood order:
        self.__fix_mesh_hood_order(mesh, already_built_edges_hood)

        # fix sides
        self.__fix_mesh_sides(mesh, already_built_edges_hood)
        return

    @staticmethod
    def get_vertices_connections_via_another_edge(mesh, v1, v2, allowed_edges=None):
        """
        Find mutual connection (vertex) via another edges from vertices v1 and
        v2.
        """
        # get all edges connected to v1 or v2
        e_v1 = mesh.ve[v1]
        e_v2 = mesh.ve[v2]
        if allowed_edges is not None:
            e_v1 = [e for e in e_v1 if e in allowed_edges]
            e_v2 = [e for e in e_v2 if e in allowed_edges]
        # get all vertices of edges e_v1 or e_v2
        v_e_v1 = set(mesh.edges[e_v1].reshape(-1))
        v_e_v2 = set(mesh.edges[e_v2].reshape(-1))
        # get the vertex intersection
        inter_vertices = set.intersection(v_e_v1, v_e_v2)
        return inter_vertices

    def find_doublets(self, mesh, vertices):
        doublet_pair_edges = []
        if vertices is None:
            doublet_vertices = np.where(np.array([len(mesh.ve[j]) for j in range(len(mesh.ve))]) == 2)[0]
            doublet_vertices = list(doublet_vertices)
        else:
            doublet_vertices_indices = np.where(np.array([len(mesh.ve[v]) for v in vertices]) == 2)[0]
            doublet_vertices = [vertices[i] for i in doublet_vertices_indices]

        if len(doublet_vertices) > 0:
            doublet_pair_edges = [mesh.ve[v].copy() for v in doublet_vertices]

        # check if doublet has boundaries - if it has do not clear this doublet
        doublet_pair_edges_copy = doublet_pair_edges.copy()
        doublet_vertices_copy = doublet_vertices.copy()
        for i, doublet_pair in enumerate(doublet_pair_edges_copy):
            if np.any([self.has_boundaries_edge_only(mesh, d) for d in doublet_pair]):
                doublet_pair_edges.remove(doublet_pair)
                doublet_vertices.remove(doublet_vertices_copy[i])

        return doublet_vertices, doublet_pair_edges

    def clear_doublets(self, mesh, mask, edge_groups, vertices=None):
        doublet_vertices, doublet_pairs_edges = self.find_doublets(mesh, vertices)
        if len(doublet_vertices) == 0:
            return False

        for pair in doublet_pairs_edges:
            old_mesh = deepcopy(mesh)
            old_hood = old_mesh.gemm_edges[pair[0]]
            replaced_edges = [e for e in old_hood[0:3] if e not in pair]
            other_edges_to_fix = [e for e in old_hood[3:6] if e not in pair]
            new_vertex = set(old_mesh.edges[replaced_edges[0]]).intersection(set(old_mesh.edges[replaced_edges[1]])).pop()

            # re-build hood
            # match doublet edge with replaced edge
            doubelt_to_replaced_edge = dict()
            v_e = set(old_mesh.edges[pair[0]])  # vertices of doublet edge
            v_r = set(old_mesh.edges[replaced_edges[0]])  # vertices of potential replaced edge
            mutual_v = v_e.intersection(v_r)
            if len(mutual_v) == 1:
                doubelt_to_replaced_edge[pair[0]] = replaced_edges[0]
                doubelt_to_replaced_edge[pair[1]] = replaced_edges[1]
            else:
                doubelt_to_replaced_edge[pair[0]] = replaced_edges[1]
                doubelt_to_replaced_edge[pair[1]] = replaced_edges[0]

            # union groups for features
            for key in doubelt_to_replaced_edge.keys():
                MeshPool.__union_groups(mesh, edge_groups, key,
                                        doubelt_to_replaced_edge[key])

            # fix other edges hood
            for edge in other_edges_to_fix:
                new_hood = mesh.gemm_edges[edge]
                for i, e in enumerate(new_hood):
                    if e in doubelt_to_replaced_edge.keys():
                        new_hood[i] = doubelt_to_replaced_edge[e]

            # fix other side hood
            doubelt_to_replaced_edge_other_side = dict()
            v_e = set(old_mesh.edges[pair[0]])  # vertices of doublet edge
            v_r = set(old_mesh.edges[other_edges_to_fix[0]])  # vertices of potential replaced edge
            mutual_v = v_e.intersection(v_r)
            if len(mutual_v) == 1:
                doubelt_to_replaced_edge_other_side[pair[0]] = other_edges_to_fix[0]
                doubelt_to_replaced_edge_other_side[pair[1]] = other_edges_to_fix[1]
            else:
                doubelt_to_replaced_edge_other_side[pair[0]] = other_edges_to_fix[1]
                doubelt_to_replaced_edge_other_side[pair[1]] = other_edges_to_fix[0]

            # union groups for features
            for key in doubelt_to_replaced_edge_other_side.keys():
                MeshPool.__union_groups(mesh, edge_groups, key,
                                        doubelt_to_replaced_edge_other_side[key])

            for edge in replaced_edges:
                new_hood = mesh.gemm_edges[edge]
                for i, e in enumerate(new_hood):
                    if e in pair:
                        new_hood[i] = doubelt_to_replaced_edge_other_side[e]

            # fix hood order:
            edges_list = replaced_edges + other_edges_to_fix
            self.__fix_mesh_hood_order(mesh, edges_list)

            # fix sides
            self.__fix_mesh_sides(mesh, edges_list)

            # remove doublet from mesh:
            for e in pair:
                u, v = mesh.edges[e]
                MeshPool.__remove_group(mesh, edge_groups, e)  # TODO needed?
                mask[e] = False
                mesh.remove_edge(e)
                mesh.edges[e] = [-1, -1]  # TODO needed?
                mesh.edges_count -= 1
                mesh.gemm_edges[e] = [-1, -1, -1, -1, -1, -1]

        return True

    @staticmethod
    def remove_edge(mesh, e, edge_groups, mask):
        MeshPool.__remove_group(mesh, edge_groups, e)
        mask[e] = False
        mesh.remove_edge(e)
        mesh.edges[e] = [-1, -1]
        mesh.edges_count -= 1
        mesh.gemm_edges[e] = [-1, -1, -1, -1, -1, -1]

    def build_new_hood_for_diag_collapse(self, mesh, edge, removed_edges, new_faces):
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
        self.__fix_mesh_hood_order(mesh, [edge])

        return

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
    def has_boundaries_edge_only(mesh, edge_id):
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
    def __union_groups_at_once(mesh, edge_groups, targets_to_sources_dict):
        edge_groups.union_groups(targets_to_sources_dict)
        for target in targets_to_sources_dict.keys():
            for source in targets_to_sources_dict[target]:
                if target is not source:
                    mesh.union_groups(source, target)

    @staticmethod
    def __remove_group(mesh, edge_groups, index):
        edge_groups.remove_group(index)
        mesh.remove_group(index)

    @staticmethod
    def __fix_mesh_sides(mesh, edges):
        for edge in edges:
            new_sides = [-1, -1, -1, -1, -1, -1]
            for i_en, en in enumerate(mesh.gemm_edges[edge]):
                if en == -1:
                    continue
                side = np.where(mesh.gemm_edges[en] == edge)[0]
                if len(side) > 0:
                    new_sides[i_en] = side[0]

            mesh.sides[edge] = new_sides

    @staticmethod
    def __fix_mesh_hood_order(mesh, edges):
        for edge in edges:
            hood = mesh.gemm_edges[edge]
            if mesh.edges[hood[2], 0] not in mesh.edges[hood[3]] and \
                    mesh.edges[hood[2], 1] not in mesh.edges[hood[3]]:
                hood[5], hood[3] = hood[3], hood[5]

    # @staticmethod
    # def get_all_vertices_of_edges_connected_to_vertex(mesh, vertex_u):
    #     """
    #     Get all the vertices of edges which are connected to vertex u,
    #     exclude u itself. Another output is the edges themselves.
    #     """
    #     v_e_u = []
    #     e_u = mesh.ve[vertex_u].copy()  # edges connected to vertex u
    #
    #     for e in e_u:
    #         v1, v2 = mesh.edges[e]
    #         if v1 == vertex_u:
    #             v_e_u.append(v2)
    #         else:
    #             v_e_u.append(v1)
    #
    #     return v_e_u, e_u

    # def get_edge_hood_info(self, mesh, edge_id):
    #     # get vertices of edge with edge_id
    #     u, v = mesh.edges[edge_id]
    #
    #     # get all edges connected to vertex u and all the vertices of them
    #     v_e_u, e_u = self.get_all_vertices_of_edges_connected_to_vertex(mesh,
    #                                                                     u)
    #     # get all edges connected to vertex v and all the vertices of them
    #     v_e_v, e_v = self.get_all_vertices_of_edges_connected_to_vertex(mesh,
    #                                                                     v)
    #     if len(e_u) > len(e_v):
    #         # swap u and v
    #         u, v = v, u
    #         v_e_u, v_e_v = v_e_v, v_e_u
    #         e_u, e_v = e_v, e_u
    #
    #     return u, v_e_u, e_u, v, v_e_v, e_v
    # @staticmethod
    # def get_edge_from_two_vertices(mesh, v1, v2):
    #     e1 = mesh.ve[v1]
    #     e2 = mesh.ve[v2]
    #     edge = None
    #     for e in e1:
    #         if e in e2:
    #             edge = e
    #             break
    #     return edge

    # def find_double_faces(self, mesh, face, faces, face_vertices, faces_vertices):
    #     all_vn = []
    #     removed_faces = []
    #     removed_faces_vertices = []
    #     for v in face_vertices:
    #         vn, _ = self.get_all_vertices_of_edges_connected_to_vertex(mesh, v)
    #         all_vn = all_vn + vn
    #
    #     outer_vertices = set(all_vn) - set(face_vertices)
    #     if len(outer_vertices) == 4:
    #         faces_copy = faces.copy()
    #         faces_vertices_copy = faces_vertices.copy()
    #         for fi, f in enumerate(faces_copy):
    #             if f == face:
    #                 continue
    #             else:
    #                 is_outer_face = len(outer_vertices.difference(set(faces_vertices_copy[fi]))) == 0
    #                 if is_outer_face:
    #                     removed_faces.append(f)
    #                     removed_faces_vertices.append(faces_vertices_copy[fi])
    #                     faces.remove(f)
    #                     faces_vertices.remove(faces_vertices_copy[fi])
    #                     # print(f)
    #                     # print('double face was found!')
    #
    #     return removed_faces, removed_faces_vertices


