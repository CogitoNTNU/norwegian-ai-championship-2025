import gymnasium as gym

IM_GLOBAL_STEP = 0  # Global imitation phase step counter


class ImitationPhaseWrapper(gym.Wrapper):
    def __init__(self, env, rule_based_ai, pretrain_steps):
        super().__init__(env)
        self.rule_based_ai = rule_based_ai
        self.pretrain_steps = pretrain_steps
        self.global_step = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        # --- Pick one source of truth for step counting
        # If you want a GLOBAL imitation phase, use IM_GLOBAL_STEP, otherwise per-env:
        imitation_active = self.global_step < self.pretrain_steps
        print(
            f"[IMITATION] step={self.global_step}, action={action} ({self.env.action_map[action]})"
        )

        if imitation_active:
            action = self.rule_based_ai.get_action(self._last_obs, self.global_step)
            action_source = "rule_based"
        else:
            action_source = "ppo"

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs
        self.global_step += 1

        # Optional: if you want a truly global counter, you could use IM_GLOBAL_STEP
        # global IM_GLOBAL_STEP
        # IM_GLOBAL_STEP += 1

        info["imitation_phase"] = imitation_active
        info["action_source"] = action_source

        return obs, reward, terminated, truncated, info
