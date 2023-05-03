import numpy
import torch
import torch.nn.functional as F

from torch_ac.algos.base import BaseAlgo

import os

class PPOAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, device=None, preprocess_obss=None, cfg=None):
        cfg.frames_per_proc = cfg.frames_per_proc or 128

        super().__init__(envs, acmodel, device, preprocess_obss, cfg)

        self.clip_eps = cfg.clip_eps
        self.epochs = cfg.epochs
        self.batch_size = cfg.batch_size

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.AdamW(self.acmodel.parameters(), cfg.lr, eps=cfg.optim_eps)
        self.batch_num = 0

    def update_parameters(self, exps, bc_mode, store_data=False, lr_down=False, cur_num_frames=0):
        # Collect experiences
        if lr_down:
            for g in self.optimizer.param_groups:
                g['lr'] = g['lr']*0.1
        # arr_list = []
        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []

            for inds in self._get_batches_starting_indexes():
                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                cur_exp = []
                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss
                    cur_exp.append([sb.obs.image[0, :, :, 0], sb.action[0]])

                    if self.acmodel.recurrent:
                        if self.cfg.use_ext_mem:
                            dist, value, memory_w_dist = self.acmodel(sb.obs, memory * sb.mask)
                        else:
                            dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        if self.cfg.use_ext_mem:
                            dist, value, memory_w_dist = self.acmodel(sb.obs, None)
                        else:
                            dist, value = self.acmodel(sb.obs)

                    entropy = dist.entropy().mean()

                    if not bc_mode:
                        if self.cfg.use_ext_mem:
                            sb.action = sb.action.reshape(-1, 2)
                            sb.log_prob = sb.log_prob.reshape(-1, 2)
                            breakpoint()
                            ratio = torch.exp(dist.log_prob(sb.action[:, 0]) - sb.log_prob[:, 0])
                            surr1 = ratio * sb.advantage
                            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                            policy_loss_act = -torch.min(surr1, surr2).mean()
                            ratio = torch.exp(memory_w_dist.log_prob(sb.action[:, 1]) - sb.log_prob[:, 1])
                            surr1 = ratio * sb.advantage
                            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                            policy_loss_memw = -torch.min(surr1, surr2).mean()
                            policy_loss = (policy_loss_act + policy_loss_memw)/2
                        else:
                            ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                            surr1 = ratio * sb.advantage
                            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                            policy_loss = -torch.min(surr1, surr2).mean()
                    else:
                        policy_loss = -dist.log_prob(sb.action).mean()
                        # print(sb.action)
                        # if not torch.all(sb.action == 2):
                        #     # policy_loss += dist.entropy().mean()
                        #     print("probability going left (top): ", dist.log_prob(torch.full((sb.obs.image.shape[0],), 0).cuda()).mean())
                        #     print("probability going right (bottom): ", dist.log_prob(torch.full((sb.obs.image.shape[0],), 1).cuda()).mean())
                        #     print("probability of correct action: ", dist.log_prob(sb.action).mean())
                        #     if dist.log_prob(sb.action).mean() > -0.05 and not os.path.isfile("current_run_working.txt"):
                        #         with open("current_run_working.txt", 'w') as f:
                        #             f.write(str(cur_num_frames))
                        #             f.write("\n")
                        #             f.write(str(dist.log_prob(sb.action).mean()))

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()
                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # arr_list.append(cur_exp)
                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward()
                grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms)
        }
        
        # if store_data:
        #     with open("obs_list_S9linear.txt", 'a') as f:
        #         for exp in arr_list:
        #             for tup in exp:
        #                 obs, gt = tup[0], tup[1]
        #                 f.write("Observation:\n")
        #                 numpy.savetxt(f, obs.cpu().detach().numpy(), fmt='%d', delimiter='\t')
        #                 f.write("\nGt action: ")
        #                 f.write(str(int(gt)))
        #                 f.write("\n\n")
        #             f.write("-----one recurrent exp end-------\n\n")

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes
