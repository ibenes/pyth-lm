import numpy as np
import copy


def area_under_curve(xs_in, ys_in):
    assert(len(xs_in) == len(ys_in))

    xs = list(copy.deepcopy(xs_in))
    ys = list(copy.deepcopy(ys_in))

    if xs[0] > 0.0:
        xs.insert(0, 0.0)
        ys.insert(0, 1.0)

    if ys[-1] > 0.0:
        xs.append(1.0)
        ys.append(0.0)

    running_sum = 0.0

    for i in range(len(xs)-1):
        x_len = xs[i+1] - xs[i]
        avg_y = (ys[i] + ys[i+1])/2
        running_sum += x_len * avg_y

    return running_sum


def eer(xs, ys):
    assert(len(xs) == len(ys))

    eer = float('nan')
    for i in range(len(xs)-1):
        if xs[i] < ys[i] and xs[i+1] >= ys[i+1]:
            d_i = abs(xs[i] - ys[i])
            d_ip1 = xs[i+1] - ys[i+1]
            lambda_i = d_i/(d_i + d_ip1)
            eer = lambda_i * xs[i] + (1.0-lambda_i)*xs[i+1]

    return eer


class DETCurve:
    def __init__(self, score_tg, baseline, max_det_points=0):
        self._baseline = baseline

        nb_trials = len(score_tg)
        nb_same = sum(s[1] for s in score_tg)
        nb_different = nb_trials - nb_same

        self._max_miss_rate = nb_same / nb_trials
        self._max_fa_rate = nb_different / nb_trials

        print("# positive trials: {} ({:.1f} %)".format(nb_same, 100.0*nb_same/nb_trials))
        print("# negative trials: {} ({:.1f} %)".format(nb_different, 100.0*nb_different/nb_trials))

        score_tg = sorted(score_tg, key=lambda s: s[0])

        mis_fas = []
        nb_correct_same = nb_same
        nb_correct_different = 0
        nb_false_alarms = nb_different
        nb_misses = 0

        for s in score_tg:
            if s[1] == 1:
                nb_misses += 1
                nb_correct_same -= 1
            else:
                nb_false_alarms -= 1
                nb_correct_different += 1

            mis_fas.append([nb_misses/nb_trials, nb_false_alarms/nb_trials])

        if max_det_points > 0:
            assert(max_det_points > 1)
            subsampling_coeff = len(mis_fas) // max_det_points
            mis_fas = mis_fas[::subsampling_coeff] + [mis_fas[-1]]

        mis_fas = np.asarray(mis_fas)
        self._miss_rate = mis_fas[:, 0]
        self._fa_rate = mis_fas[:, 1]

    def textual_report(self):
        report = ""
        area_line_fmt = "Area under DET curve (in linspace): {:.5f}"
        eer_line_fmt = "EER: {:.2f} %"

        system_au_det = area_under_curve(self._miss_rate, self._fa_rate)
        system_eer = eer(self._miss_rate, self._fa_rate)

        if self._baseline:
            area_line_fmt += " / {:.5f} / {:.2f} %"
            eer_line_fmt += " / {:.2f} % / {:.2f} %"

            baseline_au_det = self._max_miss_rate * self._max_fa_rate / 2.0
            baseline_eer = self._max_miss_rate * self._max_fa_rate / (self._max_miss_rate + self._max_fa_rate)

            report += area_line_fmt.format(
                system_au_det,
                baseline_au_det,
                100.0 * (1.0 - system_au_det/baseline_au_det)
            ) + '\n'
            report += eer_line_fmt.format(
                100.0*system_eer,
                100.0*baseline_eer,
                100.0 * (1.0 - system_eer/baseline_eer)
            ) + '\n'

        else:
            report += area_line_fmt.format(system_au_det) + '\n'
            report += eer_line_fmt.format(100.0*system_eer) + '\n'

        return report

    def plot(self, log_axis, scaled_axis, eer_line):
        import matplotlib.pyplot as plt
        plt.figure()

        if log_axis:
            plt_func = plt.loglog
        else:
            plt_func = plt.plot
        plt_func(self._miss_rate, self._fa_rate, label='System')

        if self._baseline:
            xs = np.linspace(0, self._max_miss_rate)
            ys = np.linspace(self._max_fa_rate, 0)
            plt_func(xs, ys, label='Baseline')

            plt.legend()

        if eer_line:
            endpoint = min([self._max_fa_rate, self._max_miss_rate])
            plt_func([0.0, endpoint], [0.0, endpoint], color='k', linestyle='-.', linewidth=0.75)

        if scaled_axis:
            plt.axis('scaled')
        plt.xlim(left=0.0)
        plt.ylim(bottom=0.0)
        plt.xlabel('miss rate')
        plt.ylabel('FA rate')

        plt.show()
