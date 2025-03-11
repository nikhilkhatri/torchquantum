"""
MIT License

Copyright (c) 2020-present TorchQuantum Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import torch
import torchquantum as tq

from torchpack.utils.logging import logger


__all__ = [
    "NoiseModelTQActivation",
    "NoiseModelTQPhase",
]


def cos_adjust_noise(
    current_epoch,
    n_epochs,
    prob_schedule,
    prob_schedule_separator,
    orig_noise_total_prob,
):
    """
        Adjust the noise probability based on the current epoch and a cosine schedule.

        Args:
            current_epoch (int): The current epoch.
            n_epochs (int): The total number of epochs.
            prob_schedule (str): The probability schedule type. Possible values are:
                - None: No schedule, use the original noise probability.
                - "increase": Increase the noise probability using a cosine schedule.
                - "decrease": Decrease the noise probability using a cosine schedule.
                - "increase_decrease": Increase the noise probability until a separator epoch,
                  then decrease it using cosine schedules.
            prob_schedule_separator (int): The epoch at which the schedule changes for
                "increase_decrease" mode.
            orig_noise_total_prob (float): The original noise probability.

        Returns:
            float: The adjusted noise probability based on the schedule.

        Note:
            The adjusted noise probability is returned as a float between 0 and 1.

        Raises:
            None.

        """

    if prob_schedule is None:
        noise_total_prob = orig_noise_total_prob
    elif prob_schedule == "increase":
        # scale the cos
        if current_epoch <= prob_schedule_separator:
            noise_total_prob = orig_noise_total_prob * (
                -np.cos(current_epoch / prob_schedule_separator * np.pi) / 2 + 0.5
            )
        else:
            noise_total_prob = orig_noise_total_prob
    elif prob_schedule == "decrease":
        if current_epoch >= prob_schedule_separator:
            noise_total_prob = orig_noise_total_prob * (
                np.cos(
                    (current_epoch - prob_schedule_separator)
                    / (n_epochs - prob_schedule_separator)
                    * np.pi
                )
                / 2
                + 0.5
            )
        else:
            noise_total_prob = orig_noise_total_prob
    elif prob_schedule == "increase_decrease":
        # if current_epoch <= self.prob_schedule_separator:
        #     self.noise_total_prob = self.orig_noise_total_prob * \
        #         1 / (1 + np.exp(-(current_epoch - (
        #             self.prob_schedule_separator / 2)) / 10))
        # else:
        #     self.noise_total_prob = self.orig_noise_total_prob * \
        #         1 / (1 + np.exp((current_epoch - (
        #             self.n_epochs + self.prob_schedule_separator) / 2) /
        #                 10))
        if current_epoch <= prob_schedule_separator:
            noise_total_prob = orig_noise_total_prob * (
                -np.cos(current_epoch / prob_schedule_separator * np.pi) / 2 + 0.5
            )
        else:
            noise_total_prob = orig_noise_total_prob * (
                np.cos(
                    (current_epoch - prob_schedule_separator)
                    / (n_epochs - prob_schedule_separator)
                    * np.pi
                )
                / 2
                + 0.5
            )
    else:
        logger.warning(
            f"Not implemented schedule{prob_schedule}, " f"will not change prob!"
        )
        noise_total_prob = orig_noise_total_prob

    return noise_total_prob


def apply_readout_error_func(x, c2p_mapping, measure_info):
    """
        Apply readout error to the measurement outcomes.

        Args:
            x (torch.Tensor): The measurement outcomes, represented as a tensor of shape (batch_size, num_qubits).
            c2p_mapping (dict): Mapping from qubit indices to physical wire indices.
            measure_info (dict): Measurement information dictionary containing the probabilities for different outcomes.

        Returns:
            torch.Tensor: The measurement outcomes after applying the readout error, represented as a tensor of the same shape as x.

        Note:
            The readout error is applied based on the given mapping and measurement information.
            The measurement information dictionary should have the following structure:
            {
                (wire_1,): {"probabilities": [[p_0, p_1], [p_0, p_1]]},
                (wire_2,): {"probabilities": [[p_0, p_1], [p_0, p_1]]},
                ...
            }
            where wire_1, wire_2, ... are the physical wire indices, and p_0 and p_1 are the probabilities of measuring 0 and 1, respectively,
            for each wire.

        Raises:
            None.

        """
    # add readout error
    noise_free_0_probs = (x + 1) / 2
    noise_free_1_probs = 1 - (x + 1) / 2

    noisy_0_to_0_prob_all = []
    noisy_0_to_1_prob_all = []
    noisy_1_to_0_prob_all = []
    noisy_1_to_1_prob_all = []

    for k in range(x.shape[-1]):
        p_wire = [c2p_mapping[k]]
        noisy_0_to_0_prob_all.append(measure_info[tuple(p_wire)]["probabilities"][0][0])
        noisy_0_to_1_prob_all.append(measure_info[tuple(p_wire)]["probabilities"][0][1])
        noisy_1_to_0_prob_all.append(measure_info[tuple(p_wire)]["probabilities"][1][0])
        noisy_1_to_1_prob_all.append(measure_info[tuple(p_wire)]["probabilities"][1][1])

    noisy_0_to_0_prob_all = torch.tensor(noisy_0_to_0_prob_all, device=x.device)
    noisy_0_to_1_prob_all = torch.tensor(noisy_0_to_1_prob_all, device=x.device)
    noisy_1_to_0_prob_all = torch.tensor(noisy_1_to_0_prob_all, device=x.device)
    noisy_1_to_1_prob_all = torch.tensor(noisy_1_to_1_prob_all, device=x.device)

    noisy_measured_0 = (
        noise_free_0_probs * noisy_0_to_0_prob_all
        + noise_free_1_probs * noisy_1_to_0_prob_all
    )

    noisy_measured_1 = (
        noise_free_0_probs * noisy_0_to_1_prob_all
        + noise_free_1_probs * noisy_1_to_1_prob_all
    )
    noisy_expectation = noisy_measured_0 * 1 + noisy_measured_1 * (-1)

    return noisy_expectation


class NoiseCounter:
    """
        A class for counting the occurrences of Pauli error gates.

        Attributes:
            counter_x (int): Counter for Pauli X errors.
            counter_y (int): Counter for Pauli Y errors.
            counter_z (int): Counter for Pauli Z errors.
            counter_X (int): Counter for Pauli X errors (for two-qubit gates).
            counter_Y (int): Counter for Pauli Y errors (for two-qubit gates).
            counter_Z (int): Counter for Pauli Z errors (for two-qubit gates).

        Methods:
            add(error): Adds a Pauli error to the counters based on the error type.
            __str__(): Returns a string representation of the counters.

        """
    def __init__(self):
        self.counter_x = 0
        self.counter_y = 0
        self.counter_z = 0
        self.counter_X = 0
        self.counter_Y = 0
        self.counter_Z = 0

    def add(self, error):
        if error == 'x':
            self.counter_x += 1
        elif error == 'y':
            self.counter_y += 1
        elif error == 'z':
            self.counter_z += 1
        if error == 'X':
            self.counter_X += 1
        elif error == 'Y':
            self.counter_Y += 1
        elif error == 'Z':
            self.counter_Z += 1
        else:
            pass
        
    def __str__(self) -> str:
        return f'single qubit error: pauli x = {self.counter_x}, pauli y = {self.counter_y}, pauli z = {self.counter_z}\n' + \
               f'double qubit error: pauli x = {self.counter_X}, pauli y = {self.counter_Y}, pauli z = {self.counter_Z}'




class NoiseModelTQActivation(object):
    """
        A class for adding noise to the activations.

        Attributes:
            mean (tuple): Mean values of the noise.
            std (tuple): Standard deviation values of the noise.
            n_epochs (int): Number of epochs.
            prob_schedule (list): Probability schedule.
            prob_schedule_separator (str): Separator for probability schedule.
            after_norm (bool): Flag indicating whether noise should be added after normalization.
            factor (float): Factor for adjusting the noise.

        Methods:
            adjust_noise(current_epoch): Adjusts the noise based on the current epoch.
            sample_noise_op(op_in): Samples a noise operation.
            apply_readout_error(x): Applies readout error to the input.
            add_noise(x, node_id, is_after_norm): Adds noise to the activations.

        """


    def __init__(
        self,
        mean=(0.0,),
        std=(1.0,),
        n_epochs=200,
        prob_schedule=None,
        prob_schedule_separator=None,
        after_norm=False,
        factor=None,
    ):
        self.mean = mean
        self.std = std
        self.is_add_noise = True
        self.mode = "train"
        self.after_norm = after_norm

        self.orig_std = std
        self.n_epochs = n_epochs
        self.prob_schedule = prob_schedule
        self.prob_schedule_separator = prob_schedule_separator
        self.factor = factor

    @property
    def noise_total_prob(self):
        return self.std

    @noise_total_prob.setter
    def noise_total_prob(self, value):
        self.std = value

    def adjust_noise(self, current_epoch):
        self.std = cos_adjust_noise(
            current_epoch=current_epoch,
            n_epochs=self.n_epochs,
            prob_schedule=self.prob_schedule,
            prob_schedule_separator=self.prob_schedule_separator,
            orig_noise_total_prob=self.orig_std,
        )

    def sample_noise_op(self, op_in):
        return []

    def apply_readout_error(self, x):
        return x

    def add_noise(self, x, node_id, is_after_norm=False):
        if (self.after_norm and is_after_norm) or (
            not self.after_norm and not is_after_norm
        ):
            if self.mode == "train" and self.is_add_noise:
                if self.factor is None:
                    factor = 1
                else:
                    factor = self.factor

                x = (
                    x
                    + torch.randn(x.shape, device=x.device) * self.std[node_id] * factor
                    + self.mean[node_id]
                )

        return x


class NoiseModelTQPhase(object):
    """
        A class for adding noise to rotation parameters.

        Attributes:
            mean (float): Mean value of the noise.
            std (float): Standard deviation value of the noise.
            n_epochs (int): Number of epochs.
            prob_schedule (list): Probability schedule.
            prob_schedule_separator (str): Separator for probability schedule.
            factor (float): Factor for adjusting the noise.

        Methods:
            adjust_noise(current_epoch): Adjusts the noise based on the current epoch.
            sample_noise_op(op_in): Samples a noise operation.
            apply_readout_error(x): Applies readout error to the input.
            add_noise(phase): Adds noise to the rotation parameters.

        """

    def __init__(
        self,
        mean=0.0,
        std=1.0,
        n_epochs=200,
        prob_schedule=None,
        prob_schedule_separator=None,
        factor=None,
    ):
        self.mean = mean
        self.std = std
        self.is_add_noise = True
        self.mode = "train"

        self.orig_std = std
        self.n_epochs = n_epochs
        self.prob_schedule = prob_schedule
        self.prob_schedule_separator = prob_schedule_separator
        self.factor = factor

    @property
    def noise_total_prob(self):
        return self.std

    @noise_total_prob.setter
    def noise_total_prob(self, value):
        self.std = value

    def adjust_noise(self, current_epoch):
        self.std = cos_adjust_noise(
            current_epoch=current_epoch,
            n_epochs=self.n_epochs,
            prob_schedule=self.prob_schedule,
            prob_schedule_separator=self.prob_schedule_separator,
            orig_noise_total_prob=self.orig_std,
        )

    def sample_noise_op(self, op_in):
        return []

    def apply_readout_error(self, x):
        return x

    def add_noise(self, phase):
        if self.mode == "train" and self.is_add_noise:
            if self.factor is None:
                factor = 1
            else:
                factor = self.factor
            phase = (
                phase
                + torch.randn(phase.shape, device=phase.device) * self.std * factor
                + self.mean
            )

        return phase


