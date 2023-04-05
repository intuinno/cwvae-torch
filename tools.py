import ruamel.yaml as yaml
from pathlib import Path
import numpy as np
import imageio
import os
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter


class AttrDict(dict):

    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__


# class Module(tf.Module):
#     def __init__(self):
#         super().__init__()
#         self._modules = {}

#     def get(self, name, Tf_layer, *args, **kwargs):
#         if name not in self._modules.keys():
#             self._modules[name] = Tf_layer(*args, **kwargs)
#         return self._modules[name]


class Step:
    def __init__(self, session):
        self._session = session
        self._step = tf.get_variable("step", initializer=lambda: 0, trainable=False)

    def increment(self):
        self._session.run(self._step.assign_add(1))

    def __call__(self):
        return self._session.run(self._step)


def exp_name(cfg, model_dir_prefix=None):
    exp_name = "{}_cwvae_{}".format(cfg.dataset, cfg.cell_type.lower())
    exp_name += "_{}l_f{}".format(cfg.levels, cfg.tmp_abs_factor)
    exp_name += "_decsd{}".format(cfg.dec_stddev)
    exp_name += "_enchl{}_ences{}_edchnlmult{}".format(
        cfg.enc_dense_layers, cfg.enc_dense_embed_size, cfg.channels_mult
    )
    exp_name += "_ss{}_ds{}_es{}".format(
        cfg.cell_stoch_size, cfg.cell_deter_size, cfg.cell_embed_size
    )
    exp_name += "_seq{}_lr{}_bs{}".format(cfg.seq_len, cfg.lr, cfg.batch_size)
    return exp_name


def validate_config(cfg):
    assert (
        cfg.channels is not None and cfg.channels > 0
    ), "Incompatible channels = {} found in config.".format(cfg.config)
    assert (
        cfg.open_loop_ctx % (cfg.tmp_abs_factor ** (cfg.levels - 1)) == 0
    ), "Incompatible open-loop context length {} and temporal abstraction factor {} for levels {}".format(
        cfg.open_loop_ctx, cfg.tmp_abs_factor, cfg.levels
    )
    assert cfg.datadir is not None, "data root directory cannot be None."
    assert cfg.logdir is not None, "log root directory cannot be None."


def read_configs(config_path, base_config_path=None, **kwargs):
    if base_config_path is not None:
        base_config = yaml.safe_load(Path(base_config_path).read_text())
        config = base_config.copy()
        config.update(yaml.safe_load(Path(config_path).read_text()))
        assert (
            len(set(config).difference(base_config)) == 0
        ), "Found new keys in config. Make sure to set them in base_config first."
    else:
        with open(config_path, "r") as f:
            config = yaml.load(f)
    config = AttrDict(config)

    if kwargs.get("datadir", None) is not None:
        config.datadir = kwargs["datadir"]
    if kwargs.get("logdir", None) is not None:
        config.logdir = kwargs["logdir"]

    validate_config(config)
    return config


def scan(cell, inputs, use_obs, initial):
    assert initial is not None, "initial cannot be None. Pass zero_state instead."
    # Transpose inputs to (T, B, ...)
    inputs = tf.nest.map_structure(
        lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape.as_list())))),
        inputs,
    )
    outputs = tf.scan(
        lambda agg, inp: cell(agg, inp, use_obs=use_obs), inputs, initializer=initial
    )
    # Transpose outputs back to (B, T, ...)
    outputs = tf.nest.map_structure(
        lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape.as_list())))),
        outputs,
    )
    return outputs["out"], tf.nest.map_structure(
        lambda x: x[:, -1, ...], outputs["state"]
    )


def _to_padded_strip(images):
    if len(images.shape) <= 3:
        images = np.expand_dims(images, -1)
    c_dim = images.shape[-1]
    x_dim = images.shape[-3]
    y_dim = images.shape[-2]
    padding = 1
    result = np.zeros(
        (x_dim, y_dim * images.shape[0] + padding * (images.shape[0] - 1), c_dim),
        dtype=np.uint8,
    )
    for i in range(images.shape[0]):
        result[:, i * y_dim + i * padding : (i + 1) * y_dim + i * padding, :] = images[
            i
        ]
    if result.shape[-1] == 1:
        result = np.reshape(result, result.shape[:2])
    return result


def save_as_grid(images, save_dir, filename, strip_width=50):
    # Creating a grid of images.
    # images shape: (T, ...)
    results = list([])
    if images.shape[0] < strip_width:
        results.append(_to_padded_strip(images))
    else:
        for i in range(0, images.shape[0], strip_width):
            if i + strip_width < images.shape[0]:
                results.append(_to_padded_strip(images[i : i + strip_width]))
    grid = np.concatenate(results, 0)
    imageio.imwrite(os.path.join(save_dir, filename), grid)
    print("Written grid file {}".format(os.path.join(save_dir, filename)))


def compute_metrics(gt, pred):
    gt = np.transpose(gt, [0, 1, 4, 2, 3])
    pred = np.transpose(pred, [0, 1, 4, 2, 3])
    bs = gt.shape[0]
    T = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            for c in range(gt[i][t].shape[0]):
                ssim[i, t] += ssim_metric(gt[i][t][c], pred[i][t][c])
                psnr[i, t] += psnr_metric(gt[i][t][c], pred[i][t][c])
            ssim[i, t] /= gt[i][t].shape[0]
            psnr[i, t] /= gt[i][t].shape[0]

    return ssim, psnr


def plot_metrics(metrics, logdir, name):
    mean_metric = np.squeeze(np.mean(metrics, 0))
    stddev_metric = np.squeeze(np.std(metrics, 0))
    np.savez(os.path.join(logdir, "{}_mean.npz".format(name)), mean_metric)
    np.savez(os.path.join(logdir, "{}_stddev.npz".format(name)), stddev_metric)

    plt.figure()
    axes = plt.gca()
    axes.yaxis.grid(True)
    plt.plot(mean_metric, color="blue")
    axes.fill_between(
        np.arange(0, mean_metric.shape[0]),
        mean_metric - stddev_metric,
        mean_metric + stddev_metric,
        color="blue",
        alpha=0.4,
    )
    plt.savefig(os.path.join(logdir, "{}_range.png".format(name)))

    

def args_type(default):
  def parse_string(x):
    if default is None:
      return x
    if isinstance(default, bool):
      return bool(['False', 'True'].index(x))
    if isinstance(default, int):
      return float(x) if ('e' in x or '.' in x) else int(x)
    if isinstance(default, (list, tuple)):
      return tuple(args_type(default[0])(y) for y in x.split(','))
    return type(default)(x)
  def parse_object(x):
    if isinstance(default, (list, tuple)):
      return tuple(x)
    return x
  return lambda x: parse_string(x) if isinstance(x, str) else parse_object(x)


class Optimizer():

    def __init__(
        self, name, parameters, lr, eps=1e-4, clip=None, wd=None, wd_pattern=r'.*',
        opt='adam', use_amp=False):
        assert 0 <= wd < 1
        assert not clip or 1 <= clip
        self._name = name
        self._parameters = parameters
        self._clip = clip
        self._wd = wd
        self._wd_pattern = wd_pattern
        self._opt = {
            'adam': lambda: torch.optim.Adam(parameters,
                                lr=lr,
                                eps=eps),
            'nadam': lambda: NotImplemented(
                                f'{config.opt} is not implemented'),
            'adamax': lambda: torch.optim.Adamax(parameters,
                                lr=lr,
                                eps=eps),
            'sgd': lambda: torch.optim.SGD(parameters,
                            lr=lr),
            'momentum': lambda: torch.optim.SGD(parameters,
                                    lr=lr,
                                    momentum=0.9),
        }[opt]()
        self._scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    def __call__(self, loss, params, retain_graph=False):
        assert len(loss.shape) == 0, loss.shape
        metrics = {}
        metrics[f'{self._name}_loss'] = loss.detach().cpu().numpy()
        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self._opt)
        #loss.backward(retain_graph=retain_graph)
        norm = torch.nn.utils.clip_grad_norm_(params, self._clip)
        if self._wd:
            self._apply_weight_decay(params)
        self._scaler.step(self._opt)
        self._scaler.update()
        #self._opt.step()
        self._opt.zero_grad()
        metrics[f'{self._name}_grad_norm'] = norm.item()
        return metrics

    def _apply_weight_decay(self, varibs):
        nontrivial = (self._wd_pattern != r'.*')
        if nontrivial:
            raise NotImplementedError
        for var in varibs:
            var.data = (1 - self._wd) * var.data
        

def schedule(string, step):
  try:
    return float(string)
  except ValueError:
    match = re.match(r'linear\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, duration = [float(group) for group in match.groups()]
      mix = torch.clip(torch.Tensor([step / duration]), 0, 1)[0]
      return (1 - mix) * initial + mix * final
    match = re.match(r'warmup\((.+),(.+)\)', string)
    if match:
      warmup, value = [float(group) for group in match.groups()]
      scale = torch.clip(step / warmup, 0, 1)
      return scale * value
    match = re.match(r'exp\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, halflife = [float(group) for group in match.groups()]
      return (initial - final) * 0.5 ** (step / halflife) + final
    match = re.match(r'horizon\((.+),(.+),(.+)\)', string)
    if match:
      initial, final, duration = [float(group) for group in match.groups()]
      mix = torch.clip(step / duration, 0, 1)
      horizon = (1 - mix) * initial + mix * final
      return 1 - 1 / horizon
    raise NotImplementedError(string)

    
class RequiresGrad:

  def __init__(self, model):
    self._model = model

  def __enter__(self):
    self._model.requires_grad_(requires_grad=True)

  def __exit__(self, *args):
    self._model.requires_grad_(requires_grad=False)

def static_scan(fn, inputs, start):
  last = start
  indices = range(inputs[0].shape[0])
  flag = True
  for index in indices:
    inp = lambda x: (_input[x] for _input in inputs)
    last = fn(last, *inp(index))
    if flag:
      if type(last) == type({}):
        outputs = {key: value.clone().unsqueeze(0) for key, value in last.items()}
      else:
        outputs = []
        for _last in last:
          if type(_last) == type({}):
            outputs.append({key: value.clone().unsqueeze(0) for key, value in _last.items()})
          else:
            outputs.append(_last.clone().unsqueeze(0))
      flag = False
    else:
      if type(last) == type({}):
        for key in last.keys():
          outputs[key] = torch.cat([outputs[key], last[key].unsqueeze(0)], dim=0)
      else:
        for j in range(len(outputs)):
          if type(last[j]) == type({}):
            for key in last[j].keys():
              outputs[j][key] = torch.cat([outputs[j][key],
                  last[j][key].unsqueeze(0)], dim=0)
          else:
            outputs[j] = torch.cat([outputs[j], last[j].unsqueeze(0)], dim=0)
  if type(last) == type({}):
    outputs = [outputs]
  return outputs



class ContDist:

  def __init__(self, dist=None):
    super().__init__()
    self._dist = dist
    self.mean = dist.mean

  def __getattr__(self, name):
    return getattr(self._dist, name)

  def entropy(self):
    return self._dist.entropy()

  def mode(self):
    return self._dist.mean

  def sample(self, sample_shape=()):
    return self._dist.rsample(sample_shape)

  def log_prob(self, x):
    return self._dist.log_prob(x)



class Logger:

  def __init__(self, logdir, step):
    self._logdir = logdir
    self._writer = SummaryWriter(log_dir=str(logdir), max_queue=1000)
    self._last_step = None
    self._last_time = None
    self._scalars = {}
    self._images = {}
    self._videos = {}
    self.step = step

  def scalar(self, name, value):
    self._scalars[name] = float(value)

  def image(self, name, value):
    self._images[name] = np.array(value)

  def video(self, name, value):
    self._videos[name] = np.array(value)

  def write(self, fps=False):
    scalars = list(self._scalars.items())
    if fps:
      scalars.append(('fps', self._compute_fps(self.step)))
    print(f'[{self.step}]', ' / '.join(f'{k} {v:.1f}' for k, v in scalars))
    with (self._logdir / 'metrics.jsonl').open('a') as f:
      f.write(json.dumps({'step': self.step, ** dict(scalars)}) + '\n')
    for name, value in scalars:
      self._writer.add_scalar('scalars/' + name, value, self.step)
    for name, value in self._images.items():
      self._writer.add_image(name, value, self.step)
    for name, value in self._videos.items():
      name = name if isinstance(name, str) else name.decode('utf-8')
      if np.issubdtype(value.dtype, np.floating):
        value = np.clip(255 * value, 0, 255).astype(np.uint8)
      B, T, H, W, C = value.shape
      value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B*W))
      self._writer.add_video(name, value, self.step, 16)

    self._writer.flush()
    self._scalars = {}
    self._images = {}
    self._videos = {}

  def _compute_fps(self, step):
    if self._last_step is None:
      self._last_time = time.time()
      self._last_step = step
      return 0
    steps = step - self._last_step
    duration = time.time() - self._last_time
    self._last_time += duration
    self._last_step = step
    return steps / duration

  def offline_scalar(self, name, value, step):
    self._writer.add_scalar('scalars/'+name, value, step)

  def offline_video(self, name, value, step):
    if np.issubdtype(value.dtype, np.floating):
      value = np.clip(255 * value, 0, 255).astype(np.uint8)
    B, T, H, W, C = value.shape
    value = value.transpose(1, 4, 2, 0, 3).reshape((1, T, C, H, B*W))
    self._writer.add_video(name, value, step, 16)
