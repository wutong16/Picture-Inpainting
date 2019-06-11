from My_Inplement_Inpainting import *

parser = argparse.ArgumentParser()
parser.add_argument('--picture_path', type=str, default='pictures/hill.jpg', help='file path to the complete picture')
parser.add_argument('--mask_path', type=str, default='pictures/hill_mask.jpg', help='file path for the mask')
parser.add_argument('--save_dir', type=str, default='./results', help='path to save the results')
parser.add_argument('--patch_size', type=int, default=65, help='side-length for a square patch')
parser.add_argument('--step', type=int, default=16, help='step interval to fetch the patches')
parser.add_argument('--alpha', type=float, default=0.001, help='alpha value for Lasso')
parser.add_argument('--max_iter', type=int, default=10000, help='max_iteration time for Lasso')
parser.add_argument('--tolerance', type=float, default=0.0001, help='tolerance value for Lasso')
parser.add_argument('--local', type=bool, default=True, help='whether to build the dictionary locally')
args = parser.parse_args()


PI = PictureInpainting(picture_path=args.picture_path, mask_path = args.mask_path, save_dir=args.save_dir,
                       patch_size=args.patch_size, step=args.step ,
                       alpha = args.alpha,tolerance=args.tolerance, max_iter = args.max_iter, local = args.local)

PI.inpainting()
PI.measure()
PI.report_results()