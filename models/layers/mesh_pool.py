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
        # ids = [10248,13982,9061,9947,13409,1385,1726,1465,9006,14759,10240,13301,1375,1359,8560,3710,14757,970,14782,13408,2908,14751,11352,13550,1461,1727,11007,488,6073,13580,14789,9056,15796,9016,13114,9340,1734,3321,13426,14755,7049,7034,7271,12911,9907,1730,13142,4765,13981,7770,7032,13141,10606,13582,12917,13527,7051,4190,9008,13591,7036,11350,3375,13289,13316,4521,13558,8018,13634,7778,4199,3712,12012,9273,13596,7399,10743,9063,13588,3313,10734,14900,3326,13525,15531,9093,5482,7211,11259,3317,14784,5478,1350,10741,2582,5284,9091,9945,6125,32,7038,13294,2895,3225,3330,1335,5931,13148,9338,6336,9095,12016,7605,9334,9269,6128,12881,14793,3652,5592,6130,9097,10745,13600,13110,13632,3219,9080,10698,8016,12905,6338,4775,7192,14906,9277,3235,14786,5934,9060,4936,10136,14281,1728,10730,3249,4781,8173,6597,10700,9949,9955,9078,3714,7030,1363,13615,4767,15255,14461,7776,12919,3395,9101,10738,13150,12742,14390,3708,14005,3393,10657,14740,8193,1193,13554,9087,5281,10354,4777,9940,3391,9105,13584,13584,8304,6131,3387,3387,13107,9012,13585,7194,7194,3222,5819,4785,6127,4779,5476,5476,2900,10747,13225,5474,5474,14475,9058,4789,4760,4760,3109,9113,5928,10140,13101,13101,1469,9109,10294,9099,9099,5283,5283,14742,8172,7766,6123,6123,4793,2899,8315,10795,10733,10733,8317,14275,3113,6045,4527,3322,3322,12740,8302,13298,13599,13599,1392,2893,8313,9135,6023,4773,1420,6134,9121,14931,3397,7329,9117,9117,10799,13593,14745,6397,13232,1355,13412,2927,13300,13300,8305,12005,1356,13942,5480,5591,5591,9089,9089,14287,14287,9130,9125,9125,7046,12241,4769,4769,10049,10049,3324,3324,14283,12752,1483,4763,13805,13805,4783,13626,13626,12807,10963,14925,15247,5479,5479,6041,10144,9305,14896,14896,9411,9980,9980,9141,9103,9103,12822,13138,13636,4787,4787,14908,13803,13803,6330,13230,3371,14795,14285,14285,7223,6334,9138,13382,13382,6332,9951,9951,14792,14792,13280,15245,3239,10706,9111,9111,7205,7843,9140,8321,8321,3401,3401,13305,13305,9115,9115,6053,3379,3379,12879,14009,8021,10602,12738,12738,9119,9119,8517,3656,1487,7031,3711,8511,6391,9263,8329,7847,15012,13595,13595,3247,14660,7048,10047,9261,8022,8022,8174,8174,10699,9107,9107,8325,10726,10726,2921,10296,978,978,13601,3325,9133,5667,12588,3226,3226,1382,3403,3403,3399,3399,8327,7819,13145,9946,9946,6389,12247,14477,13597,13597,8179,8179,4563,9137,9137,10350,8311,8311,3713,3713,8319,9415,9357,3778,6712,8012,8307,8323,8323,10658,10705,10705,6710,15199,8309,8309,12291,1479,12699,14902,14902,13152,13152,2898,2898,10803,7784,7784,14747,14747,2889,7842,10714,10727,10727,8983,13552,4934,8515,7037,7037,13099,13099,10731,10731,14277,12020,10696,10696,3128,6020,6840,12243,12243,11011,3385,9976,8024,8024,1731,10300,9332,11253,490,490,9011,9011,2894,2894,13602,13602,10695,10695,969,7225,14783,14783,13384,10612,5590,14018,12483,6037,13974,13441,13441,9323,9330,8471,10967,11257,4938,10116,1877,1879,1879,9281,9281,4771,4771,6025,1485,9487,1885,15541,8512,8512,9987,9987,11251,6832,6194,3245,13413,13413,8180,11263,1873,10043,10043,7398,11833,10148,13985,10298,14930,8014,8014,1477,9333,9333,6016,6016,1360,1360,6559,3477,13628,13628,9881,3780,8475,8475,3220,3220,13547,13547,13952,10897,14328,14767,8507,5923,6043,3768,7045,7045,7209,7209,4761,4761,7772,5707,14279,13598,8031,6049,6133,6133,12914,12914,1310,4944,6561,1893,10704,10704,1489,1489,3132,10965,5708,8670,12805,9992,4567,4567,5484,5484,3323,3323,13146,13146,5585,7328,9977,3241,3241,12956,8036,3319,12480,14394,9481,9941,9941,31,31,3525,8562,10346,1308,6828,8519,8519,3253,3253,10002,13589,14282,14282,7198,13380,13380,9361,2901,2901,1345,8017,8020,2250,2250,10028,10028,11998,5103,5103,1887,2580,9070,14773,1881,1881,9066,11265,11265,11068,15527,15527,14790,5929,5929,9879,13958,3407,3389,3389,489,489,5926,10777,6621,6621,8520,5821,10807,10801,10801,3254,3254,3405,3405,6604,1318,1318,1413,14752,14752,3110,3110,1214,11017,10421,14332,8525,7322,7322,13553,13553,14765,14765,13147,13147,11247,7024,5586,5586,13286,14327,14327,9082,8563,8563,13105,13105,2254,14280,14280,3409,3774,10712,6328,10961,2896,1467,1493,1493,1495,15532,15532,5930,5930,6664,15561,2252,2252,8333,15529,6136,6136,7042,7042,5933,1323,13279,10708,14918,9321,11066,9336,9336,1720,1720,7200,7200,9957,15203,6199,6199,12478,8331,8331,3752,12871,8028,8028,1344,1344,13439,6019,9147,9147,8521,11346,1480,1480,13010,14011,1886,1886,1565,4184,7033,7033,1388,3248,3248,11013,6563,11003,3413,13809,11060,8335,1384,1384,10423,10423,13976,13976,10045,13576,13576,14289,3280,12189,15248,15248,9959,8503,8503,9984,9984,13149,13149,6039,6124,6124,10702,10702,9145,9145,13501,1491,1491,5477,5477,3637,6060,9143,15249,15249,10033,3411,3411,11058,11005,11005,5815,6064,6059,6059,3479,4565,13307,13307,13103,13103,2707,10152,5813,6055,6055,13624,8533,8023,8023,10974,2258,2258,13372,13372,15088,7791,5596,3329,3329,14273,13423,13423,3244,3659,3659,13100,13100,7974,14794,14794,5728,5728,9877,10692,10692,3314,3314,6024,6024,13096,13096,6200,6200,3227,3227,8674,8508,14293,12598,8675,8675,10360,10360,1327,10735,10735,5594,10711,10711,14912,14912,8176,8176,13388,13388,13798,7848,7848,13060,3415,2536,7327,7327,14796,14796,5486,15785,10618,15784,12889,13084,9007,3377,3292,13640,8177,8177,7043,7043,3250,3250,9493,9493,3283,9878,9878,12024,5022,5022] #,2916]
        # for edge_id in ids:  # TODO just for test
        #     assert(self.__is_valid_config(mesh, edge_id))
        #     fe = self.rotate_edges(mesh, fe, edge_id)
        #
        # # # # TODO actual
        # while mesh.edges_count > self.__out_target:
        #     value, edge_id = heappop(queue)
        #     edge_id = int(edge_id)
        #     assert(self.__is_valid_config(mesh, edge_id))
        #     print(edge_id)
        #     fe = self.rotate_edges(mesh, fe, edge_id)

        # TODO remove after check mask and mesh.clean staff
        count = -1
        ids = [2070, 2638, 2823, 2640, 2642, 2498, 2821, 2496, 2819, 2062, 3944, 4383, 4379, 4381, 4649, 4660, 3942, 4662, 3940, 3938, 5183, 8791, 1997, 1999, 2001, 4619, 4617, 4651, 4427, 4429, 8789, 2003, 4431, 2428, 3878, 3880, 8787, 5791, 2064, 8910, 8906, 3876, 4083, 4077, 8914, 2500, 4071, 3874, 9553, 15703, 10142, 2426, 4073, 4652, 5138, 10146, 4350, 2430, 4346, 8743, 13009, 4300, 5181, 9551, 4348, 5793, 9555, 4009, 8785, 4007, 3930, 4013, 2704, 8741, 9810, 4019, 4298, 15748, 8739, 5134, 8783, 15764, 9812, 2432, 8858, 3932, 2054, 4302, 8862, 1991, 1989, 2434, 13456, 1993, 3936, 4615, 3934, 4389, 1995, 8866, 9549, 4694, 9816, 4613, 8733, 5785, 3868, 8735, 4433, 3870, 4435, 4437, 15719, 11444, 3866, 15766, 3872, 8737, 4387, 4692, 3208, 10138, 8781, 12427, 8786, 5136, 11446, 9748, 2134, 8784, 9547, 4091, 3935, 3933, 3931, 12051, 8912, 4385, 4294, 4292, 10963, 4085, 4081, 2502, 8782, 2632, 2128, 4079, 2424, 3945, 12423, 4656, 15710, 4296, 9746, 2061, 8916, 2122, 8887, 4040, 11397, 9545, 8920, 4052, 4698, 2827, 9754, 4380, 4390, 2825, 10382, 9760, 4382, 9619, 9779, 2124, 5787, 4646, 9756, 4650, 5133, 5189, 9752, 8891, 2636, 15717, 9766, 4696, 15701, 8872, 10551, 2634, 8535, 4645, 8868, 2829, 10559, 9543, 2142, 15749, 4430, 9811, 2706, 3881, 2633, 9785, 9267, 4428, 4438, 11212, 1992, 2004, 1994, 2435, 15562, 9263, 4614, 15750, 9758, 2136, 11216, 4616, 8864, 1905, 15702, 4317, 1990, 2643, 10555, 13626, 5784, 4681, 3871, 12644, 8492, 12180, 4648, 4058, 10860, 4321, 4325, 4340, 10380, 2130, 2427, 5790, 4027, 2132, 12047, 4679, 5190, 3869, 4886, 4021, 4344, 5789, 11448, 4342, 4017, 4015, 5788, 5187, 1907, 10545, 2353, 3867, 2497, 9820, 13498, 9541, 10298, 2635, 2347, 8738, 2351, 15765, 8919, 11401, 8895, 10296, 8734, 13452, 2425, 2491, 9817, 9620, 8736, 12035, 2763, 2345, 8915, 2820, 2830, 5132, 11220, 9822, 2816, 2822, 12176, 10376, 10578, 1903, 2708, 2493, 9791, 2641, 8911, 2639, 12178, 13458, 2814, 5786, 10549, 12039, 4046, 4374, 10582, 10553, 2824, 4090, 2069, 9611, 2649, 10626, 4084, 4666, 2109, 13460, 9610, 2343, 13004, 9556, 2499, 1320, 10620, 5179, 4659, 2103, 8673, 9317, 5780, 4647, 15659, 9617, 8295, 11814, 9818, 2349, 5024, 4664, 5028, 4078, 10574, 3212, 9482, 4376, 11818, 11405, 10300, 5182, 14461, 5022, 2765, 8488, 5387, 4384, 13055, 3899, 13064, 5020, 5137, 5094, 5135, 5383, 5129, 9823, 11206, 12031, 3953, 11210, 9127, 3939, 9123, 10374, 9325, 8496, 3941, 12431, 11561, 3943, 9546, 3216, 8792, 5184, 10628, 4799, 13056, 9797, 5783, 2097, 12432, 2503, 9815, 2705, 4072, 2627, 8790, 10378, 4791, 4795, 2709, 2504, 5099, 6657, 11816, 11563, 11214, 5093, 9608, 8788, 4432, 3901, 14295, 2078, 8867, 11877, 5131, 6250, 8863, 1904, 5766, 2056, 13454, 3889, 2258, 5779, 6246, 4026, 15718, 8529, 4414, 12642, 5128, 9809, 13234, 8871, 13090, 4424, 3913, 4020, 4406, 3903, 4293, 2000, 2629, 4653, 9814, 4014, 4422, 14339, 13451, 1998, 12027, 6655, 11808, 1328, 4404, 4620, 8531, 4618, 9747, 10622, 2501, 11393, 6044, 6048, 11882, 9094, 4008, 4630, 3879, 2701, 8301, 2002, 5185, 15747, 15655, 13078, 2492, 9829, 3215, 12045, 9609, 4341, 2012, 2646, 4345, 2490, 10593]
        while mesh.edges_count > self.__out_target:
            value, edge_id = heappop(queue)
            count = count + 1
            edge_id = ids[count]
            edge_id = int(edge_id)
            print('pool edge_id %d' % edge_id)
            if mask[edge_id]:
                _, fe = self.__pool_edge(mesh, edge_id, fe, mask, edge_groups)
        # mesh.clean(mask, edge_groups)
        # fe = edge_groups.rebuild_features(self.__fe[mesh_index], mask,
        #                                   self.__out_target)
        self.__updated_fe[mesh_index] = fe

    def __pool_edge(self, mesh, edge_id, fe, mask, edge_groups):

        if not np.all(mesh.gemm_edges[mesh.gemm_edges[edge_id], mesh.sides[edge_id]] == edge_id):
            print('problem!!!!!!')
            new_sides = np.where(
                mesh.gemm_edges[mesh.gemm_edges[edge_id]] == edge_id)[1]
            mesh.sides[edge_id] = new_sides

        if self.has_boundaries(mesh, edge_id):
            return False, fe
        elif self.__clean_side(mesh, edge_id, mask, edge_groups, 0) \
                and self.__clean_side(mesh, edge_id, mask, edge_groups, 3) \
                and self.__is_one_ring_valid(mesh, edge_id):

            # for id in range(mesh.edges_count):
            #     mesh_copy = mesh
            #     mask_copy = mask
            #     self.__merge_edges = self.__pool_side(mesh_copy, id, mask_copy, edge_groups, 0)

            fe = self.rotate_edges(mesh, fe, edge_id, mask)

            # mesh.merge_vertices(edge_id)
            # mask[edge_id] = False
            # MeshPool.__remove_group(mesh, edge_groups, edge_id)
            # mesh.edges_count -= 1
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

        return faces, face_edges, faces_vertices

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

        return edge_nb, sides, abs_edges

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
    def combine_edge_features(in_features, new_features_combination_dict):
        """
        Creates new edge features according to features combinations dictionary
        """
        updated_fe = in_features.clone()
        for k in new_features_combination_dict.keys():
            combination = new_features_combination_dict[k]
            assert (len(combination) > 0)
            new_feature = torch.sum(in_features[:, combination, :],
                                    axis=1) / len(combination)
            # new_feature = new_feature.reshape(new_feature.shape[0], 1, -1)
            updated_fe[:, int(k), :] = new_feature

        return updated_fe


    def find_double_faces(self, mesh, face, faces, face_vertices, faces_vertices):
        all_vn = []
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
                        faces.remove(f)
                        faces_vertices.remove(faces_vertices[fi])
                        print('double face was found!')

    def find_double_faces2(self, mesh, face, face_vertices):

        faces_vertices = []
        faces = []
        face_to_remove = None
        face_vertices_to_remove = None

        for ver in face_vertices:
            f, _, fv = \
                self.get_all_faces_connected_to_vertex(mesh, ver)
            for i, fv_i in enumerate(fv):
                face_vertices_copy = [face_vertices.copy()]
                is_new = True
                # for cfv in face_vertices_copy:
                    # diff = len(set(cfv).difference(set(fv_i)))
                    # if diff == 0:
                    #     is_new = False
                    #     break
                if is_new:
                    faces_vertices.append(fv_i)
                    faces.append(f[i])

        for fi, inner_face in enumerate(faces):
            self.find_double_faces(mesh, inner_face, faces, faces_vertices[fi], faces_vertices)

        # all_vn = []
        # for v in face_vertices:
        #     vn, _ = self.get_all_vertices_of_edges_connected_to_vertex(mesh, v)
        #     all_vn = all_vn + vn
        #
        # outer_vertices = set(all_vn) - set(face_vertices)
        # if len(outer_vertices) == 4:
        #     for fi, f in enumerate(faces):
        #         if f == face:
        #             continue
        #         else:
        #             is_outer_face = len(outer_vertices.difference(set(faces_vertices[fi]))) == 0
        #             if is_outer_face:
        #                 face_to_remove = f
        #                 face_vertices_to_remove = faces_vertices[fi]
        #                 print('double face was found!')

        return face_to_remove, face_vertices_to_remove




    @staticmethod # TODO move from here
    def find_doublets(mesh):
        doublet_pair_edges = []
        doublet_vertices = np.where(np.array([len(mesh.ve[j]) for j in range(len(mesh.ve))]) == 2)[0]
        if len(doublet_vertices) > 0:
            doublet_pair_edges = [mesh.ve[v].copy() for v in doublet_vertices]
        return doublet_vertices, doublet_pair_edges

    def clear_doublets(self, mesh, mask):
        doublet_vertices, doublet_pairs_edges = self.find_doublets(mesh)
        if len(doublet_vertices) == 0:
            return []
        for i, doublet_pair_edges in enumerate(doublet_pairs_edges):
            vertex = doublet_vertices[i]
            for e in doublet_pair_edges:
                u, v = mesh.edges[e]

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

        doublet_vertices = list(doublet_vertices) + list(self.clear_doublets(mesh, mask))
        return doublet_vertices

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

    def rotate_edges(self, mesh, fe, edge_id, mask):
        # get vertices of edge with edge_id
        u, v = mesh.edges[edge_id]

        if u == -1 or v == -1: # TODO replace it by the mask
            print('edge_id %d already removed' % edge_id)
            return fe

        # get all edges connected to vertex u and all the vertices of them
        v_e_u, e_u = self.get_all_vertices_of_edges_connected_to_vertex(mesh,
                                                                        u)
        v_e_v, e_v = self.get_all_vertices_of_edges_connected_to_vertex(mesh,
                                                                        v)

        # Edges rotation
        # 1. find diagonal vertices
        diag_vertices, diag_vertex_to_edges_dict = \
            self.find_diag_vertices(mesh, u, e_u, v_e_u)
        diag_vertices_v, diag_vertex_to_edges_dict_v = \
            self.find_diag_vertices(mesh, v, e_v, v_e_v)

        # 2. rotate - for each edge goes from u - change the original
        # connection to the optional diagonal connection (direction should be
        # consistent for all the rotated edges
        mesh, new_features_combination_dict = \
            self.rotate_edges_and_connections(mesh, u, e_u, diag_vertices,
                                              diag_vertex_to_edges_dict)

        # 3. collapse another 2 edges connected to the other vertex v
        e_v = mesh.ve[v].copy()  # edges connected to vertex v
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

                # MeshPool.__remove_group(mesh, edge_groups, e)  # TODO needed?
                mask[e] = False
                mesh.remove_edge(e)
                mesh.edges[e] = [-1, -1]  # TODO needed?
                mesh.gemm_edges[e] = [-1, -1, -1, -1, -1, -1]  # TODO needed?
                mesh.edges_count -= 1


                # mesh.ve[u_e].remove(e)
                # mesh.ve[v_e].remove(e)

            else:
                e_to_reconnect_with_u.append(e)
                mesh.ve[v].remove(e)
                mesh.ve[u].append(e)
                if mesh.edges[e, 0] == v:
                    mesh.edges[e, 0] = u
                else:
                    mesh.edges[e, 1] = u

        ##### test
        ve_to_reconnect_with_u = set(np.array([mesh.edges[e].flatten() for e in e_to_reconnect_with_u]).flatten())

        one_ring_v = self.find_all_one_ring_vertices(u, set(v_e_u), set(diag_vertices),
                                        set(diag_vertices_v),
                                        ve_to_reconnect_with_u)
        ##### test

        doublet_vertices = self.clear_doublets(mesh, mask)

        # create new edge features
        updated_fe = self.combine_edge_features(fe,
                                                new_features_combination_dict)

        # re-build edges neighborhood
        # 1. Get new faces
        faces, _, faces_vertices = \
            self.get_all_faces_connected_to_vertex(mesh, u)

        outer_edges = list(set().union(*faces) - set(e_u))

        ##### test
        # clear faces out of one-ring
        # for face_v in faces_vertices:
        #     if len(set(face_v).intersection(one_ring_v)) is not 4:
        #         print('not ok!')
        ##### test

        ##### test
        # for fv_i, face_vertices in enumerate(faces_vertices):
        #     for fv_j, face_vertices2 in enumerate(faces_vertices):
        #         if fv_i == fv_j:
        #             continue;
        #         edges_i = []
        #         edges_ver_i = []
        #         for fv_i in face_vertices:
        #             e_fv, v_e_fv = self.get_all_vertices_of_edges_connected_to_vertex(fv_i)
        #             edges_i.append(e_fv)
        #             edges_ver_i.append(v_e_fv)
        #         edges_j = []
        #         edges_ver_j = []
        #         for fv in face_vertices:
        #             e_fv, v_e_fv = self.get_all_vertices_of_edges_connected_to_vertex(fv)
        #             edges.append(e_fv)
        #             edges_ver.append(v_e_fv)
        ##### test

        vers = list(set().union(*faces_vertices))
        for ver in vers:
            f, _, fv = \
                self.get_all_faces_connected_to_vertex(mesh, ver)
            for i, fv_i in enumerate(fv):
                faces_vertices_copy = faces_vertices.copy()
                is_new = True
                for cfv in faces_vertices_copy:
                    diff = len(set(cfv).difference(set(fv_i)))
                    if diff == 0:
                        is_new = False
                        break
                if is_new and np.any(outer_edges in f[i]):
                    faces_vertices.append(fv_i)
                    faces.append(f[i])

        if edge_id == 10593111111:
            for fi, face in enumerate(faces):
                face_to_remove, face_vertices_to_remove = self.find_double_faces2(mesh, face, faces_vertices[fi])
        else:
            for fi, face in enumerate(faces):
                self.find_double_faces(mesh, face, faces, faces_vertices[fi], faces_vertices)



        # 2. Re-build edges hood and sides
        edge_nb, sides, abs_edges = self.build_edges_hood_and_sides(mesh,
                                                                    faces_vertices)
        # 3. Fix to absolute edge values
        # for i, e_nb in enumerate(edge_nb):
        #     for j, e in enumerate(e_nb):
        #         if e is not -1:
        #             edge_nb[i][j] = abs_edges[e]
        #         else:  # if e==-1 - take the other edges from the original hood
        #             continue
                    # other_side_edges = []
                    # other_side_edges_sides = []
                    # for k, other_side_edge in \
                    #         enumerate(mesh.gemm_edges[abs_edges[i]]):
                    #     # if other_side_edge not in np.array(edge_nb[i]).reshape(-1) \
                    #     #         and other_side_edge not in e_to_collapse:
                    #     if len(other_side_edges) == 3:
                    #         break
                    #     if not other_side_edge == -1 \
                    #             and other_side_edge not in np.array(mesh.ve[u]).reshape(-1) \
                    #             and other_side_edge not in e_to_collapse \
                    #             and other_side_edge not in edge_nb[i]:
                    #         other_side_edges.append(other_side_edge)
                    #         other_side_edges_sides.append(
                    #             mesh.sides[abs_edges[i]][k])
                    # edge_nb[i][j:j + 3] = other_side_edges
                    # sides[i][j:j + 3] = other_side_edges_sides
                    # break

        # 4. Put new data in mesh data structure
        for i in range(len(abs_edges)):
            abs_edge = abs_edges[i]
            if -1 in edge_nb[i]:
                old_gemm = mesh.gemm_edges[abs_edge]
                new_gemm = edge_nb[i]
                old_pos = np.where(old_gemm == new_gemm[0])[0].flatten()

                if old_pos == 5:
                    new_gemm = old_gemm[::-1]
                elif old_pos == 3:
                    new_gemm = [old_gemm[3], old_gemm[4], old_gemm[5],
                                old_gemm[0], old_gemm[1], old_gemm[2]]
                elif old_pos == 2:
                    new_gemm = [old_gemm[2], old_gemm[1], old_gemm[0],
                                old_gemm[5], old_gemm[4], old_gemm[3]]
                elif old_pos == 0:
                    new_gemm = old_gemm
                else:
                    assert(False)

                mesh.gemm_edges[abs_edge] = new_gemm
            else:
                mesh.gemm_edges[abs_edge] = edge_nb[i]
                mesh.sides[abs_edge] = sides[i]

        # fix sides
        # for abs_edge in abs_edges:
        #     new_sides = np.where(
        #         mesh.gemm_edges[mesh.gemm_edges[abs_edge]] == abs_edge)[1]
        #     mesh.sides[abs_edge] = new_sides

        for i in range(len(abs_edges)):
            abs_edge = abs_edges[i]
            if not np.all(mesh.gemm_edges[mesh.gemm_edges[abs_edge], mesh.sides[abs_edge]] == abs_edge):
                new_sides = np.where(
                    mesh.gemm_edges[mesh.gemm_edges[abs_edge]] == abs_edge)[1]
                mesh.sides[abs_edge] = new_sides
                # print('problem!!!!!!')

        return updated_fe

    def __is_valid_config_one_vertex(self, mesh, edge_id, vertex_k):
        """
        checks that each face has only one mutual edge
        """
        e_k = mesh.ve[mesh.edges[edge_id, vertex_k]]
        e_k_hood = []

        faces, _, faces_vertices = \
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
