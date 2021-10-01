import os
import subprocess
from tqdm import tqdm


def run_process(command):
    r = subprocess.run(command)


if __name__ == '__main__':
    exe_path = r'E:\Omri\instant-meshes-windows\InstantMeshes.exe'
    # dataset_path = r'E:\Omri\FinalProject\QuadMesh\meshMNIST\Orig\t10k_meshMNIST_100V'
    dataset_path = r'E:\Omri\FinalProject\QuadMesh\meshMNIST\Orig\train_meshMNIST_100V\train_meshMNIST'
    output_res = 175
    objs_path = [os.path.join(dp, f) for dp, dn, filenames in os.walk(dataset_path)
                 for f in filenames if os.path.splitext(f)[1] == '.obj' and
                 '_quad' not in f]

    output_dir = 'instant_meshes'

    pbar = tqdm(total=len(objs_path))
    files_to_run_again = []
    for i, path in enumerate(objs_path):
        print('------- obj number %d -------' % i)
        dir = os.path.join(os.path.dirname(path), output_dir)
        file_name = os.path.basename(path).replace('.obj',
                                                   '_instantmesh_quad.obj')
        if not os.path.exists(dir):
            os.mkdir(dir)
        out_path = os.path.join(dir, file_name)
        if os.path.exists(out_path):
            pbar.update(1)
            continue
        command = exe_path + ' %s -f %s -o %s -t 4' % (
        path, output_res, out_path)
        try:
            subprocess.run(command, timeout=3)
        except subprocess.TimeoutExpired:
            files_to_run_again.append(i)
            continue

        pbar.update(1)
        print('\n\n\n\n\n\n\n\n')

    pbar.close()

    while len(files_to_run_again) > 0:
        print('now re-run again %d samples' % len(files_to_run_again))
        files_to_run_again_copy = files_to_run_again.copy()
        files_to_run_again = []
        for i in files_to_run_again_copy:
            path = objs_path[i]
            dir = os.path.join(os.path.dirname(path), output_dir)
            file_name = os.path.basename(path).replace('.obj',
                                                       '_instantmesh_quad.obj')
            if not os.path.exists(dir):
                os.mkdir(dir)
            out_path = os.path.join(dir, file_name)
            if os.path.exists(out_path):
                pbar.update(1)
                continue
            command = exe_path + ' %s -f %s -o %s -t 4' % (
                path, output_res, out_path)
            try:
                subprocess.run(command, timeout=3)
            except subprocess.TimeoutExpired:
                files_to_run_again.append(i)
                continue

    print('end')
