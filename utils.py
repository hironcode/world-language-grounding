# ObservationWrapper, wrap_env, and reward_wrap_env are the implementation of a blog below 
# Reference: https://medium.com/@kaige.yang0110/ray-rllib-how-to-train-dreamerv3-on-vizdoom-and-atari-122c8bd1170b

import gym
import cv2
import numpy as np
import json
from typing import Any, Tuple

IMAGE_SHAPE = (64, 64)
FRAME_SKIP = 4

class DoomWrapper(object):
    def __init__(
        self,
        env: gym.Env,
        config: dict
    ):
        self._env = _ObservationWrapper(env, shape=config['image_shape'], frame_skip=config['frame_skip'])
        self.img_hight = config['image_shape'][0]
        self.img_width = config['image_shape'][1]
    
    def _set_nested_attr(self, env: gym.Env, value: int, attr: str) -> None:
        """
        多重継承の属性に再帰的にアクセスして値を変更する．
        カメラの設定に利用．

        Parameters
        ----------
        value : int
            設定したい値．
        attr : str
            変更したい属性の名前．
        """
        if hasattr(env, attr):
            setattr(env, attr, value)
        else:
            self._set_nested_attr(env.env, value, attr)

    def __getattr(self, name: str) -> Any:
        """
        環境が保持している属性値を取得するメソッド．

        Parameters
        ----------
        name : str
            取得したい属性値の名前．

        Returns
        -------
        _env.name : Any
            環境が保持している属性値．
        """
        return getattr(self._env, name)

    @property
    def observation_space(self) -> gym.spaces.Box:
        """
        観測空間に関する情報を取得するメソッド．

        Returns
        -------
        space : gym.spaces.Box
            観測空間に関する情報（各画素値の最小値，各画素値の最大値，観測データの形状， データの型）．
        """
        width = self._render_width
        height = self._render_height
        return gym.spaces.Box(0, 255, (height, width, 3), dtype=np.uint8)

    @property
    def action_space(self) -> gym.spaces.Box:
        """
        行動空間に関する情報を取得するメソッド．

        Returns
        -------
        space : gym.spaces.Box
            行動空間に関する情報（各行動の最小値，各行動の最大値，行動空間の次元， データの型） ．
        """
        return self._env.action_space

    # 　元の観測（低次元の状態）は今回は捨てて，env.render()で取得した画像を観測とします.
    #  画像，報酬，終了シグナルが得られます.
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        環境に行動を与え次の観測，報酬，終了フラグを取得するメソッド．

        Parameters
        ----------
        action : np.dnarray (action_dim, )
            与える行動．

        Returns
        -------
        obs : np.ndarray (height, width, 3)
            行動を与えたときの次の観測．
        reward : float
            行動を与えたときに得られる報酬．
        done : bool
            エピソードが終了したかどうか表すフラグ．
        info : dict
            その他の環境に関する情報．
        """
        _, reward, done, info = self._env.step(action)
        # stateを画像(observation)に変換
        obs = self._env.render(mode="rgb_array")
        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        """
        環境をリセットするためのメソッド．

        Returns
        -------
        obs : np.ndarray (height, width, 3)
            環境をリセットしたときの初期の観測．
        """
        self._env.reset()
        obs = self._env.render(mode="rgb_array")
        return obs

    def render(self, mode="human", **kwargs) -> np.ndarray:
        """
        観測をレンダリングするためのメソッド．

        Parameters
        ----------
        mode : str
            レンダリング方法に関するオプション． (default='human')

        Returns
        -------
        obs : np.ndarray (height, width, 3)
            観測をレンダリングした結果．
        """
        return self._env.render(mode, **kwargs)

    def close(self) -> None:
        """
        環境を閉じるためのメソッド．
        """
        self._env.close()
        

    

class _ObservationWrapper(gym.ObservationWrapper):
    """
    ViZDoom environments return dictionaries as observations, containing
    the main image as well other info.
    The image is also too large for normal training.

    This wrapper replaces the dictionary observation space with a simple
    Box space (i.e., only the RGB image), and also resizes the image to a
    smaller size.

    NOTE: Ideally, you should set the image size to smaller in the scenario files
          for faster running of ViZDoom. This can really impact performance,
          and this code is pretty slow because of this!
    """

    def __init__(self, env, shape=IMAGE_SHAPE, frame_skip=FRAME_SKIP):
        super().__init__(env)
        self.image_shape = shape
        # print('shape', shape)
        self.image_shape_reverse = shape[::-1]
        # print('image_shape_reverse', self.image_shape_reverse)
        # REPEAT ACTIONというかframe skipはここで既に実装されている？
        # self.env.frame_skip = frame_skip

        # Create new observation space with the new shape
        # print('env.obs', env.observation_space)
        num_channels = env.observation_space["screen"].shape[-1]
        new_shape = (self.image_shape[0], self.image_shape[1], num_channels)
        # print('new_shape', new_shape)
        self.observation_space = gym.spaces.Box(0, 255, shape=new_shape, dtype=np.float32)

    def observation(self, observation):
        # print('observation["screen"].shape', observation["screen"].shape)
        observation = cv2.resize(observation["screen"], self.image_shape_reverse)
        # print('obs.shape', observation.shape)
        observation = observation.astype('float32')
        # print('obs.shape', observation.shape)
        return observation
    
class RepeatAction(gym.Wrapper):
    """
    同じ行動を指定された回数自動的に繰り返すラッパー．観測は最後の行動に対応するものになる
    """

    def __init__(self, env: DoomWrapper, skip: int = 4) -> None:
        """
        コンストラクタ．

        Parameters
        ----------
        env : GymWrapper_PyBullet
            環境のインスタンス．今回は先程定義したラッパーでラップした環境を利用する．
        skip : int
            同じ行動を繰り返す回数．
        """
        gym.Wrapper.__init__(self, env)
        self._skip = skip

    def reset(self) -> np.ndarray:
        """
        環境をリセットするためのメソッド．

        Returns
        -------
        obs : np.ndarray (width, height, 3)
            環境をリセットしたときの初期の観測．
        """
        return self.env.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """
        環境に行動を与え次の観測，報酬，終了フラグを取得するメソッド．
        与えられた行動をskipの回数だけ繰り返した結果を返す．

        Parameters
        ----------
        action : np.ndarray (action_dim, )
            与える行動．

        Returns
        -------
        obs : np.ndarray (width, height, 3)
            行動をskipの回数だけ繰り返したあとの観測．
        total_reawrd : float
            行動をskipの回数だけ繰り返したときの報酬和．
        done : bool
            エピソードが終了したかどうか表すフラグ．
        info : dict
            その他の環境に関する情報．
        """
        total_reward = 0.0
        # repeat the step
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
    

def wrap_env(env, config):
    env = DoomWrapper(env, config)
    env = RepeatAction(env, skip=4)
    env = gym.wrappers.TransformReward(env, lambda r: r * 0.01)
    return env

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


