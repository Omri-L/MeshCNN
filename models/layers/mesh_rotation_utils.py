import numpy as np
import random
from copy import deepcopy


def edge_rotations(u, e_u, v_e_u, mesh):
    """
    This function implements the edge rotation algorithm.:
    1. Find all diagonal connections from vertex u to all vertices called
    "diagonal vertices". In addition, find the optional edges to connect
    with a vertex in the diagonal vertices.
    2. Rotate the edges according to the optional new connections (all
    edges rotate to the same direction). Re-build the edges neighborhoods.
    """
    # 1. Find diagonal vertices
    diag_vertices, diag_vertex_to_edges_dict = \
        find_diag_vertices(mesh, u, e_u, v_e_u)

    if len(diag_vertices) < len(e_u):  # not correct config
        return mesh, None, None

    # 2. Rotate edges - for each edge goes from u - change the original
    # connection to the optional diagonal connection (direction should be
    # consistent for all the rotated edges)
    old_mesh = deepcopy(mesh)

    mesh, new_features_combination_dict = \
        rotate_edges_and_connections(old_mesh, mesh, u, e_u, diag_vertices,
                                     diag_vertex_to_edges_dict)

    return mesh, new_features_combination_dict, diag_vertices


def find_diag_vertices(mesh, u, e_u, v_e_u):
    """
    Find diagonl connections for all e_u edges from u to another vertex via
    another edge. The "other edge" must be in outer_edges list.
    """
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

            # find mutual vertices via another outer edge start from v_i
            # and v_j
            connection_vertices = \
                get_vertices_connections_via_another_edge(mesh, v_i,
                                                          v_j, outer_edges)
            # save only the new vertex that is not the original vertex u
            for con_vertex in connection_vertices:
                if not con_vertex == u and con_vertex not in diag_vertices:
                    diag_vertices.append(con_vertex)
                    diag_vertex_to_edges_dict[str(con_vertex)] = \
                        list([e_u[i], e_u[j]])

    return diag_vertices, diag_vertex_to_edges_dict


def get_vertices_connections_via_another_edge(mesh, v1, v2,
                                              allowed_edges=None):
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


def rotate_edges_and_connections(old_mesh, mesh, origin_vertex, edges_to_change,
                                 optional_vertices,
                                 optional_vertex_to_edges_dict):
    """
    For each edge goes from origin vertex: change the original connection
    to another optional vertex connection in a way that all vertices have
    a new connection (it will make sure the rotation for all will be CCW or
    CW).
    Re-build edge neighborhoods according to the edge rotation.
    """
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
        other_e_matched = optional_vertex_to_edges_dict[str(v_e_new)].copy()
        other_e_matched.remove(e)
        e = other_e_matched[0]

    edge_hood_built = []
    for e in edge_to_new_vertex_connection_dict:
        # build the new hood for rotated edges
        mesh.gemm_edges[e] = edge_rotation_new_hood(mesh, old_mesh, e,
                                                    edge_to_new_vertex_connection_dict[
                                                        e])
        edge_hood_built.append(e)

    # re-build new hood for the other hood edges of rotated edges
    for e in edge_to_new_vertex_connection_dict:
        new_hood = list(mesh.gemm_edges[e])
        new_faces = [new_hood[0:3] + [e], new_hood[3:6] + [e]]
        for edge in new_hood:
            if edge in edge_hood_built:
                continue
            else:
                rebuild_hood_for_edges_according_to_new_faces(mesh,
                                                              edge,
                                                              edges_to_change,
                                                              new_faces)
                edge_hood_built.append(edge)

    # fix sides
    fix_mesh_sides(mesh, edge_hood_built)

    return mesh, new_features_combination_dict


def edge_rotation_new_hood(mesh, old_mesh, e, new_vertex):
    """
    This function fixes the old edge neighborhood according to the new
    connection. This function is used after edge rotations.
    Returns: new edge neighborhood.

    """
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
        missed_e = set(old_mesh.ve[v0.pop()]).intersection(
            set(old_mesh.ve[v1.pop()]))
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
        missed_e = set(old_mesh.ve[v0.pop()]).intersection(
            set(old_mesh.ve[v1.pop()]))
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
        missed_e = set(old_mesh.ve[v0.pop()]).intersection(
            set(old_mesh.ve[v1.pop()]))
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
        missed_e = set(old_mesh.ve[v0.pop()]).intersection(
            set(old_mesh.ve[v1.pop()]))
        new_hood[4] = missed_e.pop()
    else:
        assert (False)

    return new_hood


def rebuild_hood_for_edges_according_to_new_faces(mesh, edge,
                                                  removed_edges,
                                                  new_faces):
    """
    The function re-builds edge neighborhood of an edge according to
    correct new faces exsit in the mesh strcutre.
    """
    old_hood = mesh.gemm_edges[edge]

    # find the edge in the new faces to create the new hood
    if edge in new_faces[0]:
        new_face = new_faces[0]
        pos_in_new_face = np.where(np.array(new_face) == edge)[0][0]
    elif edge in new_faces[1]:
        new_face = new_faces[1]
        pos_in_new_face = np.where(np.array(new_face) == edge)[0][0]
    else:
        assert (False)

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
        assert (False)

    # fix hood order:
    fix_mesh_hood_order(mesh, [edge])

    return


def check_u_v_boundaries(mesh, u, v_e_u, e_u, v, v_e_v, e_v):
    """
    This function checks that if any edge which comes from vertex  u has
    boundary. If yes - it switches the "roles" of u and v, and check
    boundaries again with the "new u" vertex (originally v). If there is,
    it calls this configuration invalid and returns False.
    """
    correct_config = True
    # check if any edge comes from vertex u has boundary
    switch_u_v = np.any([has_boundaries_edge_only(mesh, e) for e in e_u])

    if switch_u_v:
        correct_config = not np.any(
            [has_boundaries_edge_only(mesh, e) for e in e_v])
        if correct_config:
            # swap u and v
            u, v = v, u
            v_e_u, v_e_v = v_e_v, v_e_u
            e_u, e_v = e_v, e_u

    return correct_config, u, v_e_u, e_u, v, v_e_v, e_v


def has_boundaries_edge_only(mesh, edge_id):
    if -1 in mesh.gemm_edges[edge_id]:
        return True
    return False


def rotate_edges_around_vertex(mesh, edge_id):
    """
    This function implements edge rotation algorithm inspired by the paper:
    "Practical quad mesh simplification" Tarini et al.
    The algorithm goes as follows:
    1. Extract edge mesh information (for each vertex extract edge
        connections and their vertices).
    2. Check if the edges connected to u and v have boundaries.
    3. Rotate the edges connected to u and re-build their neighborhood.
    """

    # 1. Get edge info
    u, v_e_u, e_u, v, v_e_v, e_v = get_edge_hood_info(mesh, edge_id)

    # 2. Check if u and v edges are with boundaries
    correct_config, u, v_e_u, e_u, v, v_e_v, e_v = \
        check_u_v_boundaries(mesh, u, v_e_u, e_u, v, v_e_v, e_v)

    if not correct_config:
        return mesh

    # 3. Edges rotations around vertex u
    mesh, _, diag_vertices = \
        edge_rotations(u, e_u, v_e_u, mesh)

    return mesh


def fix_mesh_sides(mesh, edges):
    for edge in edges:
        new_sides = [-1, -1, -1, -1, -1, -1]
        for i_en, en in enumerate(mesh.gemm_edges[edge]):
            if en == -1:
                continue
            side = np.where(mesh.gemm_edges[en] == edge)[0]
            if len(side) > 0:
                new_sides[i_en] = side[0]

        mesh.sides[edge] = new_sides


def fix_mesh_hood_order(mesh, edges):
    for edge in edges:
        hood = mesh.gemm_edges[edge]
        if mesh.edges[hood[2], 0] not in mesh.edges[hood[3]] and \
                mesh.edges[hood[2], 1] not in mesh.edges[hood[3]]:
            hood[5], hood[3] = hood[3], hood[5]


def get_edge_hood_info(mesh, edge_id):
    """
    Get all edge neighborhood information.
    Args:
        edge_id (int): edge identification number to extract information.
    Returns:
        u (int): one edge vertex.
        v_e_u (list of ints): list of all vertices of all the edges
                              connected to vertex u.
        e_u (list of ints): list of all edge ids connected to vertex u.
        v (int): second edge vertex.
        v_e_v (list of ints): list of all vertices of all the edges
                              connected to vertex v.
        e_v (list of ints): list of all edge ids connected to vertex v.
    """
    # get vertices of edge with edge_id
    u, v = mesh.edges[edge_id]

    # get all edges connected to vertex u and all the vertices of them
    v_e_u, e_u = get_all_vertices_of_edges_connected_to_vertex(mesh, u)
    # get all edges connected to vertex v and all the vertices of them
    v_e_v, e_v = get_all_vertices_of_edges_connected_to_vertex(mesh, v)
    if len(e_u) > len(e_v):
        # swap u and v
        u, v = v, u
        v_e_u, v_e_v = v_e_v, v_e_u
        e_u, e_v = e_v, e_u

    return u, v_e_u, e_u, v, v_e_v, e_v


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


def clean_mesh_operations(mesh, mask, edge_groups=None):
    """
    This function implements the mesh cleaning process. In-order to keep
    mesh with valid connectivity and without edge neighborhoods ambiguities
    we keep mesh clear from "doublet" and "singlet" (TBD) edges.
    """
    # clear doublets and build new hood
    doublet_cleared = clear_doublets(mesh, mask, edge_groups)
    while doublet_cleared:
        doublet_cleared = clear_doublets(mesh, mask, edge_groups)

    # TBD
    # clear singlets and build new hood
    # clear_singlets(mesh, mask, edge_groups)
    return


def find_doublets(mesh, vertices):
    """
    Find doublet edges in the mesh structure.
    If vertices list is not None - check only this list, otherwise - all
    vertices in the mesh.
    Definition: If a vertex v has only two non-boundary edges connected to
    it, these two edges called "doublet edges".
    """
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
        if np.any([has_boundaries_edge_only(mesh, d) for d in doublet_pair]):
            doublet_pair_edges.remove(doublet_pair)
            doublet_vertices.remove(doublet_vertices_copy[i])

    return doublet_vertices, doublet_pair_edges


def clear_doublet_pair(mesh, mask, doublet_pair):

    old_mesh = deepcopy(mesh)
    old_hood = old_mesh.gemm_edges[doublet_pair[0]]
    replaced_edges = [e for e in old_hood[0:3] if e not in doublet_pair]
    other_edges_to_fix = [e for e in old_hood[3:6] if e not in doublet_pair]
    new_vertex = set(old_mesh.edges[replaced_edges[0]]).intersection(
        set(old_mesh.edges[replaced_edges[1]])).pop()

    # re-build hood
    # match doublet edge with replaced edge
    doubelt_to_replaced_edge = dict()
    v_e = set(old_mesh.edges[doublet_pair[0]])  # vertices of doublet edge
    v_r = set(old_mesh.edges[
                  replaced_edges[0]])  # vertices of potential replaced edge
    mutual_v = v_e.intersection(v_r)
    if len(mutual_v) == 1:
        doubelt_to_replaced_edge[doublet_pair[0]] = replaced_edges[0]
        doubelt_to_replaced_edge[doublet_pair[1]] = replaced_edges[1]
    else:
        doubelt_to_replaced_edge[doublet_pair[0]] = replaced_edges[1]
        doubelt_to_replaced_edge[doublet_pair[1]] = replaced_edges[0]

    # fix other edges hood
    for edge in other_edges_to_fix:
        new_hood = mesh.gemm_edges[edge]
        for i, e in enumerate(new_hood):
            if e in doubelt_to_replaced_edge.keys():
                new_hood[i] = doubelt_to_replaced_edge[e]

    # fix other side hood
    doubelt_to_replaced_edge_other_side = dict()
    v_e = set(old_mesh.edges[doublet_pair[0]])  # vertices of doublet edge
    v_r = set(old_mesh.edges[other_edges_to_fix[
        0]])  # vertices of potential replaced edge
    mutual_v = v_e.intersection(v_r)
    if len(mutual_v) == 1:
        doubelt_to_replaced_edge_other_side[doublet_pair[0]] = other_edges_to_fix[0]
        doubelt_to_replaced_edge_other_side[doublet_pair[1]] = other_edges_to_fix[1]
    else:
        doubelt_to_replaced_edge_other_side[doublet_pair[0]] = other_edges_to_fix[1]
        doubelt_to_replaced_edge_other_side[doublet_pair[1]] = other_edges_to_fix[0]

    for edge in replaced_edges:
        new_hood = mesh.gemm_edges[edge]
        for i, e in enumerate(new_hood):
            if e in doublet_pair:
                new_hood[i] = doubelt_to_replaced_edge_other_side[e]

    # fix hood order:
    edges_list = replaced_edges + other_edges_to_fix
    fix_mesh_hood_order(mesh, edges_list)

    # fix sides
    fix_mesh_sides(mesh, edges_list)

    # remove doublet from mesh:
    for e in doublet_pair:
        mask[e] = False
        mesh.remove_edge(e)
        mesh.edges[e] = [-1, -1]
        mesh.edges_count -= 1
        mesh.gemm_edges[e] = [-1, -1, -1, -1, -1, -1]

    return doubelt_to_replaced_edge, doubelt_to_replaced_edge_other_side


def clear_doublets(mesh, mask, vertices=None):
    """
    This function finds doublet configuration and removes it from the mesh.
    Args:
        mesh (Mesh): mesh structure
        mask (ndarray): array of boolean which indicates which edge removed
        edge_groups (MeshUnion): mesh union strcture contain all edges
                                 groups of edge features combinations.
        vertices (list, optional): if not None, check only this list of
                                   vertices in the mesh.
                                   Otherwise - check all mesh.

    Returns:
        boolean - True if doublet found and removed.
                  False - otherwise.
    """
    doublet_vertices, doublet_pairs_edges = find_doublets(mesh, vertices)
    if len(doublet_vertices) == 0:
        return False

    for pair in doublet_pairs_edges:
        clear_doublet_pair(mesh, mask, pair)

    return True