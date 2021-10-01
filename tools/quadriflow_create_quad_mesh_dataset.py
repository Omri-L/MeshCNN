import os
import subprocess

quadriflow_exe_path = r'E:\Omri\quadriflow\build\Debug\quadriflow.exe'
dataset_path = r'E:\Omri\FinalProject\QuadMesh\train_meshMNIST\train_meshMNIST'
output_res = 750
objs_path = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path)
             for f in filenames if os.path.splitext(f)[1] == '.obj']

for i, path in enumerate(objs_path):
    print('------- obj number %d -------' % i)
    dir = os.path.dirname(path)
    file_name = os.path.basename(path).replace('.obj', '_quad.obj')
    # file_name2 = os.path.basename(path).replace('.obj', '_quad_sharp.obj')
    out_path = os.path.join(dir, file_name)
    # out_path2 = os.path.join(dir, file_name2)
    command = quadriflow_exe_path + ' -i %s -o %s -f %d' % (
    path, out_path, output_res)

    # command2 = quadriflow_exe_path + ' -sharp -i %s -o %s -f %d' % (
    # path, out_path2, output_res)
    subprocess.run(command)
    # subprocess.run(command2)

print('end')
