import numpy as np
import os


class Img2MeshGenerator:

    def __init__(self, img_size, output_res):

        self.img_size = img_size
        self.res = output_res

        # create vetrices grid
        self.vx = np.linspace(start=0, stop=img_size[0]-1,
                              num=output_res[0]).astype(int)
        self.vy = np.linspace(start=0, stop=img_size[1]-1,
                              num=output_res[1]).astype(int)

        self.vxy_pairs = self._create_vertices_pairs()
        self.faces = self._create_faces()

        self.save_mesh_obj('E:\Omri\FinalProject\QuadMesh\MeshCNN\datasets\img2mesh_test')

    def _create_vertices_pairs(self):
        pairs = []
        for jvy in self.vy:
            for ivx in self.vx:
                pair = (ivx, jvy)
                pairs.append(pair)
        return pairs

    def _create_faces(self):
        faces = []
        for jvy in range(len(self.vy)-1):
            for ivx in range(len(self.vx)-1):
                face = [(ivx, jvy), (ivx + 1, jvy),
                        (ivx + 1, jvy + 1), (ivx, jvy + 1)]
                faces.append(face)

        return faces

    def save_mesh_obj(self, path):
        file_path = os.path.join(path, 'test.obj')
        with open(file_path, 'w') as file:
            for pair in self.vxy_pairs:
                line_format = 'v %f %f 0.0\n' % (pair[0], pair[1])
                file.write(line_format)
            for face in self.faces:
                pair_ind0 = face[0][0] + face[0][1]*self.res[0] + 1
                pair_ind1 = face[1][0] + face[1][1]*self.res[0] + 1
                pair_ind2 = face[2][0] + face[2][1]*self.res[0] + 1
                pair_ind3 = face[3][0] + face[3][1]*self.res[0] + 1
                line_format = 'f %d//%d %d//%d %d//%d %d//%d\n' % (pair_ind0,
                                                                   pair_ind0,
                                                                   pair_ind1,
                                                                   pair_ind1,
                                                                   pair_ind2,
                                                                   pair_ind2,
                                                                   pair_ind3,
                                                                   pair_ind3)
                file.write(line_format)

            file.close()

    # def img2mesh(self, img):


if __name__ == '__main__':
    img_size = [32, 32]
    output_res = [16, 16]
    gen = Img2MeshGenerator(img_size, output_res)

