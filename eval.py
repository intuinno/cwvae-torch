import argparse
import torch
import pathlib
import pickle

from cwvae import CWVAE
from data_loader import *
from datetime import datetime
import pytz
import ruamel.yaml as yaml
import tools
import os

yaml = yaml.YAML(typ='unsafe')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=None, 
        type=str,
        required=True,
        help='path to dir containing model and config.yml'
    )
    parser.add_argument(
        "--num-examples", default=100, type=int, help="number of examples to eval on"
    )
    parser.add_argument(
        "--eval-seq-len",
        default=None,
        type=int,
        help="total length of evaluation sequences",
    )
    parser.add_argument("--datadir", default=None, type=str)
    parser.add_argument(
        "--num-samples", default=1, type=int, help="samples to generate per example"
    )
    parser.add_argument(
        "--open-loop-ctx", default=36, type=int, help="number of context frames"
    )
    parser.add_argument(
        "--use-obs",
        default=None,
        type=str,
        help="string of T/Fs per level, e.g. TTF to skip obs at the top level",
    )
    parser.add_argument(
        "--no-save-grid",
        action="store_true",
        default=False,
        help="to prevent saving grids of images",
    )

    args = parser.parse_args()
    
    model_path = pathlib.Path(args.model).expanduser()
    exp_rootdir = model_path.parent.absolute()
    
    assert exp_rootdir.is_dir()
    print(f"Reading log dir {exp_rootdir}")
    
    tz = pytz.timezone("US/Central")
    now = datetime.now(tz)
    date_time = now.strftime("%Y%m%d_%H%M%S")
    eval_logdir = exp_rootdir / f"eval_{date_time}"
    eval_logdir.mkdir()
    print(f"Saving eval results at {eval_logdir}")
    configs = yaml.load((exp_rootdir / 'config.yml').read_text())
    configs.batch_size = 1 
    
    if args.use_obs is not None:
        assert len(args.use_obs) == configs.levels
        configs.use_obs = [dict(T=True, F=False)[c] for c in args.use_obs.upper()]
    else:
        configs.use_obs = True
    
    _, val_dataloader = load_dataset(configs)
    configs.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = CWVAE(configs).to(configs.device)

    # Restore model
    print(f"========== Loading saved model from {model_path} ===========")
    checkpoint = torch.load(model_path, map_location=torch.device(configs.device))
    model.load_state_dict(checkpoint)
    
    # Evaluating.
    ssim_all = []
    psnr_all = []
    for i_ex in range(args.num_examples):
    
        x = next(iter(val_dataloader))
        gts = torch.tile(
            x,
            [args.num_samples, 1, 1, 1, 1],
        )
        preds = model.pred(gts.to(configs.device), args.open_loop_ctx)        

        gts = gts / 255.0
        gts = gts.numpy()
        
        # Computing metrics.
        ssim, psnr = tools.compute_metrics(gts[:, args.open_loop_ctx :], preds)

        # Getting arrays save-ready
        gts = np.uint8(np.clip(gts, 0, 1) * 255)
        preds = np.uint8(np.clip(preds, 0, 1) * 255)

        # Finding the order within samples wrt avg metric across time.
        order_ssim = np.argsort(np.mean(ssim, -1))
        order_psnr = np.argsort(np.mean(psnr, -1))

        # Setting aside the best metrics among all samples for plotting.
        ssim_all.append(np.expand_dims(ssim[order_ssim[-1]], 0))
        psnr_all.append(np.expand_dims(psnr[order_psnr[-1]], 0))

        # Storing gt for prediction and the context.
        path = os.path.join(eval_logdir, "sample" + str(i_ex) + "_gt/")
        os.makedirs(path, exist_ok=True)
        np.savez(path + "gt_ctx.npz", gts[0, : args.open_loop_ctx])
        np.savez(path + "gt_pred.npz", gts[0, args.open_loop_ctx :])
        if not args.no_save_grid:
            tools.save_as_grid(gts[0, : args.open_loop_ctx], path, "gt_ctx.png")
            tools.save_as_grid(gts[0, args.open_loop_ctx :], path, "gt_pred.png")

        # Storing best and random samples.
        path = os.path.join(eval_logdir, "sample" + str(i_ex) + "/")
        os.makedirs(path, exist_ok=True)
        np.savez(path + "random_sample_1.npz", preds[0])
        if args.num_samples > 1:
            np.savez(path + "best_ssim_sample.npz", preds[order_ssim[-1]])
            np.savez(path + "best_psnr_sample.npz", preds[order_psnr[-1]])
            np.savez(path + "random_sample_2.npz", preds[1])
        if not args.no_save_grid:
            tools.save_as_grid(preds[0], path, "random_sample_1.png")
            if args.num_samples > 1:
                tools.save_as_grid(
                    preds[order_ssim[-1]], path, "best_ssim_sample.png"
                )
                tools.save_as_grid(
                    preds[order_psnr[-1]], path, "best_psnr_sample.png"
                )
                tools.save_as_grid(preds[1], path, "random_sample_2.png")

    # Plotting.
    tools.plot_metrics(ssim_all, eval_logdir, "ssim")
    tools.plot_metrics(psnr_all, eval_logdir, "psnr")
    metrics = { "ssim": ssim_all,
               "psnr": psnr_all}
    with open( eval_logdir / "metrics.pickle", 'wb') as f:
        pickle.dump(metrics, f)

    
    
    
    
